"""
This is a modified version of gee_functions.py to focus on forecasting power.
"""

import os
from dotenv import load_dotenv
from loguru import logger

from pandas.core import apply
import rasterio
import numpy as np 
import pandas as pd
import ee
import geemap
from PIL import Image

from urban_planner.config import CONFIG

load_dotenv()

def authenticate():
    try:
        ee.Initialize(project=os.getenv('GEE_PROJECT_ID'))
        print("Earth Engine already authenticated and initialized.")
    except Exception as e:
        print(f"Authentication needed or project not set. Attempting authentication flow...")
        service_account = os.getenv('GEE_SERVICE_ACCOUNT')
        credentials = ee.ServiceAccountCredentials(service_account, '.private-key.json')
        ee.Initialize(credentials)
    print(ee.String('Hello from the Earth Engine servers!').getInfo())

def load_cities(force:bool=False) -> pd.DataFrame:
    # Create directories if they don't exist
    os.makedirs(os.path.join(CONFIG.RAW_DATA_DIR, 'cities'), exist_ok=True)
    os.makedirs(os.path.join(CONFIG.PROCESSED_DATA_DIR, 'cities'), exist_ok=True)

    raw_cities = os.path.join(CONFIG.RAW_DATA_DIR, 'cities', 'worldcities.csv')
    processed_cities = os.path.join(CONFIG.PROCESSED_DATA_DIR, 'cities', 'worldcities_processed.csv')
    
    # Check if raw file exists,
    if not os.path.exists(raw_cities):
        raise FileNotFoundError(f"Raw city data not found. Please place 'worldcities.csv' in '{os.path.join(CONFIG.RAW_DATA_DIR, 'cities')}'")

    if not force and os.path.exists(processed_cities):
        result_df = pd.read_csv(processed_cities)
        logger.info(f"Loaded existing processed file: {processed_cities}")
    else:
        cities = pd.read_csv(raw_cities)
        coords_df = cities[['city', 'lat', 'lng', 'population', 'id']]

        coords_df = coords_df[coords_df['population'] >= CONFIG.dataset.min_population].dropna()
        coords_df_sorted = coords_df.sort_values('population', ascending=False).reset_index(drop=True)

        n = len(coords_df_sorted)
        result_rows = []
        for i in range(n // 2):
            result_rows.append(coords_df_sorted.iloc[i])
            result_rows.append(coords_df_sorted.iloc[-(i+1)])

        if n % 2 == 1:
            result_rows.append(coords_df_sorted.iloc[n // 2])

        result_df = pd.DataFrame(result_rows).reset_index(drop=True)
        result_df.to_csv(processed_cities, index=False)
        logger.info(f"Created and saved processed file: {processed_cities}")

    print("Top 5 cities to be processed:")
    print(result_df.head())
    return result_df

def process_city_chunk(args):
    """
    Worker function to process a chunk of the cities DataFrame.
    Initializes Earth Engine and processes each city in its assigned chunk.
    """
    chunk_id, cities_df_chunk, output_dir = args
    worker_pid = os.getpid()
    logger.info(f"[Worker PID: {worker_pid}, Chunk: {chunk_id}] Starting to process {len(cities_df_chunk)} cities.")

    # Each process must initialize EE on its own.
    try:
        service_account = os.getenv('GEE_SERVICE_ACCOUNT')
        credentials = ee.ServiceAccountCredentials(service_account, '.private-key.json')
        ee.Initialize(credentials)
        logger.info(f"[Worker PID: {worker_pid}] Earth Engine Initialized.")
    except Exception as e:
        logger.error(f"[Worker PID: {worker_pid}] Could not initialize Earth Engine: {e}")
        return

    # GEE assets
    year_min = 2017
    year_max = 2025
    month_min = 7
    month_max = 9
    # List of (year, month) tuples from 2015-07 to 2025-09:
    moments = []
    for year in range(year_min, year_max + 1):
        for month in range(1, 13, 2):
            if (year == year_min and month < month_min) or (year == year_max and month > month_max):
                continue
            moments.append((year, month))

    VALID_PIXEL_THRESHOLD = 0.9



    for city_idx, (city_name, lat_source, lng_source, _ ,cityid) in enumerate(cities_df_chunk.to_numpy()):
        for offset_x, offset_y in [(-0.02, 0), (0.02, 0), (0, -0.02), (0, 0.02), (0,0)]:
            lat = lat_source + offset_y
            lng = lng_source + offset_x
            basename = f"{city_name.lower().replace(' ', '_')}_{cityid}_{lat:.4f}_{lng:.4f}_{offset_x:.4f}_{offset_y:.4f}"
            coords = (lng, lat)
            meters = 1000
            point = ee.Geometry.Point(coords)
            region = point.buffer(meters).bounds()
            logger.debug(f"[Worker PID: {worker_pid}] Processing {city_idx+1}/{len(cities_df_chunk)}: {basename}")

            for year, month in moments:
                name = basename + f"_{year}_{month:02d}"
                start = ee.Date.fromYMD(year, month, 1)
                end = start.advance(30, 'days')

                dw = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
                      .filterBounds(region)
                      .filterDate(start, end)
                      )
                l8Col = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').map(apply_scale_landsat).filter( #filterDate(start, end).filterBounds(region)
                    ee.Filter.lt('CLOUD_COVER_LAND', 10)
                ).filterDate(start, end).filterBounds(region).map(maskL8sr)

                s2Col = ee.ImageCollection('COPERNICUS/S2_HARMONIZED').filter(
                    ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)
                ).filterDate(start, end).filterBounds(region)


                # Check that collections are not empty
                dw_size = dw.size().getInfo()
                s2_size = s2Col.size().getInfo()
                l8_size = l8Col.size().getInfo()

                if dw_size == 0:
                    logger.warning(f"No Dynamic World images for {name} {year}-{month}")
                    continue
                if s2_size == 0:
                    logger.warning(f"No Sentinel-2 images for {name} {year}-{month}")
                    continue
                if l8_size == 0:
                    logger.warning(f"No Landsat-8 images for {name} {year}-{month}")
                    continue

                linked = dw.linkCollection(s2Col, s2Col.first().bandNames())
                # ========================= DYNAMICWORLD ==========================
                dw_mean = linked.select('label').mean()
                # Compute fraction of valid (non-null) pixels
                valid_mask = dw_mean.mask()  # EE mask: 1 for valid, 0 for null
                stats = valid_mask.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=region,
                    scale=10,
                    maxPixels=1e13
                )
                valid_fraction = stats.get('label').getInfo()  # fraction of valid pixels
                if valid_fraction < VALID_PIXEL_THRESHOLD:
                    logger.warning(
                        f"Too many null pixels in Dynamic World for {name} {year}-{month} "
                        f"({valid_fraction*100:.2f}% valid). Skipping export."
                    )
                    continue

                geemap.ee_export_image(
                    dw_mean.clip(region),
                    filename=os.path.join(output_dir, f"{name}_dw.tif"),
                    scale=10,
                    region=region,
                    file_per_band=False
                )

                # If the resulting image do not exist, delete and skip:
                if not os.path.exists(os.path.join(output_dir, f"{name}_dw.tif")):
                    logger.warning(f"Failed to export DW image for {name}, skipping.")
                    continue

                # ========================= Sentinel 2 ==========================
                s2_rgb = linked.select(['B4', 'B3', 'B2']).mean().visualize(min=0, max=3000, bands=['B4', 'B3', 'B2'])
                geemap.ee_export_image(
                    s2_rgb.clip(region),
                    filename=os.path.join(output_dir, f'{name}_rgb.tif'),
                    scale=10,
                    region=region,
                    file_per_band=False
                )

                if not os.path.exists(os.path.join(output_dir, f"{name}_rgb.tif")):
                    os.remove(os.path.join(output_dir, f"{name}_dw.tif"))
                    logger.warning(f"Failed to export RGB image for {name}, skipping.")
                    continue

                # ========================= NDVI ==========================
                ndvi = linked.median().normalizedDifference(['B8', 'B4']).rename('NDVI')
                geemap.ee_export_image(
                    ndvi.clip(region),
                    filename=os.path.join(output_dir, f'{name}_ndvi.tif'),
                    scale=10,
                    region=region,
                    file_per_band=False
                )

                if not os.path.exists(os.path.join(output_dir, f"{name}_ndvi.tif")):
                    os.remove(os.path.join(output_dir, f"{name}_dw.tif"))
                    os.remove(os.path.join(output_dir, f"{name}_rgb.tif"))
                    logger.warning(f"Failed to export NDVI image for {name}, skipping.")
                    continue


                # ========================= Temperature  ==========================
                linkedL8 = dw.linkCollection(l8Col, l8Col.first().bandNames())
                st_kelvin = l8Col.select('ST_B10').mean()
                st_celsius = st_kelvin.subtract(273.15) # Convert to Celsius
                geemap.ee_export_image(
                    st_celsius.clip(region),
                    filename=os.path.join(output_dir, f'{name}_temp.tif'),
                    scale=30,
                    region=region,
                    file_per_band=False, 
                    #crs=dw_mean.projection().crs()
                )
                if not os.path.exists(os.path.join(output_dir, f"{name}_temp.tif")):
                    os.remove(os.path.join(output_dir, f"{name}_dw.tif"))
                    os.remove(os.path.join(output_dir, f"{name}_rgb.tif"))
                    os.remove(os.path.join(output_dir, f"{name}_ndvi.tif"))
                    logger.warning(f"Failed to export Temperature image for {name}, skipping.")
                    continue


    logger.info(f"[Worker PID: {worker_pid}, Chunk: {chunk_id}] Finished processing chunk.")



def apply_scale_landsat(image):
    optical = image.select('SR_B.*').multiply(0.0000275).add(-0.2)
    thermal = image.select('ST_B.*').multiply(0.00341802).add(149.0)
    return image.addBands(optical, overwrite=True).addBands(thermal, overwrite=True)

def maskL8sr(image):
    """
    Mask clouds using the pixel_qa band.
    """
    qa = image.select('QA_PIXEL')
    # Bits 3 and 5 are cloud and cloud shadow
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 5).eq(0))
    return image.updateMask(mask)

