import xarray as xr
import numpy as np
import os
import json
from loguru import logger

from urban_planner.config import CONFIG

def process_temperature(raw_temp_dir:str, processed_temp_dir:str):
    """
    Process CRU temperature data:
    1. Compute baseline mean and std from 1901-1950
    2. For each year 1951-2019, normalize using baseline and save to new NetCDF
    """
    # if file exists, skip processing
    if os.path.exists(os.path.join(processed_temp_dir, "tas_norm_1951.nc")) and os.path.exists(os.path.join(processed_temp_dir, "tas_norm_2019.nc")):
        logger.info("Processed temperature files already exist. Skipping processing.")
        return


    os.makedirs(processed_temp_dir, exist_ok=True)

    # Compute baseline (mean and std per grid point) over 1901-1950
    baseline_years = range(1901, 1951)
    tas_list = []

    logger.debug("Loading baseline years...")
    for year in baseline_years:
        ds = xr.open_dataset(os.path.join(raw_temp_dir, f"CRU_mean_temperature_mon_0.5x0.5_global_{year}_v4.03.nc"))
        tas_list.append(ds['tas'])
    baseline = xr.concat(tas_list, dim='time')
    baseline_mean = baseline.mean(dim='time', skipna=True)
    baseline_std = baseline.std(dim='time', skipna=True)

    # Save baseline metrics
    baseline_metrics = {
        "mean": baseline_mean.values.tolist(),
        "std": baseline_std.values.tolist()
    }
    with open(os.path.join(processed_temp_dir, "baseline_metrics.json"), 'w') as f:
        json.dump(baseline_metrics, f)
    logger.success(f"Baseline metrics saved to {os.path.join(processed_temp_dir, 'baseline_metrics.json')}")


    logger.debug("Baseline computed.")

    # Process 1951-2019 and save normalized files
    for year in range(1951, 2020):
        logger.debug(f"Processing year {year}...")
        ds = xr.open_dataset(os.path.join(raw_temp_dir, f"CRU_mean_temperature_mon_0.5x0.5_global_{year}_v4.03.nc"))
        
        tas_norm = (ds['tas'] - baseline_mean) / baseline_std
        ds_norm = ds.copy()
        ds_norm['tas'] = tas_norm
        ds_norm.to_netcdf(os.path.join(processed_temp_dir, f"tas_norm_{year}.nc"))

    logger.debug("All years processed and saved.")


class TemperatureQuery:
    def __init__(self, processed_dir:str):
        """
        Loads all processed .nc files into memory for fast querying.
        """
        self.start_year = CONFIG.dataset.temporal_start_year
        self.end_year = CONFIG.dataset.temporal_end_year
        self.data = []
        self.lats = None
        self.lons = None
        self.timestamps = [] 
        
        logger.info("Loading normalized temperature data into memory...")
        for year in range(self.start_year, self.end_year + 1):
            ds = xr.open_dataset(os.path.join(processed_dir, f"tas_norm_{year}.nc"))
            tas = ds['tas'].values  # shape: (12, lat, lon)
            if self.lats is None:
                self.lats = ds['lat'].values
                self.lons = ds['lon'].values
            self.data.append(tas)

            if self.lats is None:
                self.lats = ds['lat'].values
                self.lons = ds['lon'].values
            
            # store timestamps for each month
            for month in range(1, 13):
                self.timestamps.append((year, month))
                
        # Stack all data into a single array: (n_years*n_months, lat, lon)
        self.data = np.concatenate(self.data, axis=0) 
        logger.info(f"Data loaded: {self.data.shape[0]} time entry, {len(self.lats)} lat, {len(self.lons)} lon")
        
    def query(self, lat:float, lon:float, max_year:int, max_month:int) -> list[float]:
        """
        Returns the full normalized temperature time series for the closest grid point,
        up to year/month specified.
        """
        # Find nearest grid indices
        # lat_idx = np.abs(self.lats - lat).argmin()
        # lon_idx = np.abs(self.lons - lon).argmin()
        # ts = self.data[:, lat_idx, lon_idx]
        # return ts.tolist()

        lat_idx = (np.abs(self.lats - lat)).argmin()
        lon_idx = (np.abs(self.lons - lon)).argmin()
        ts = self.data[:, lat_idx, lon_idx]  # (n_months,)

        # find the last index to include
        for i, (y, m) in enumerate(self.timestamps):
            if (y > max_year) or (y == max_year and m > max_month):
                ts = ts[:i]
                break

        return ts.tolist()
