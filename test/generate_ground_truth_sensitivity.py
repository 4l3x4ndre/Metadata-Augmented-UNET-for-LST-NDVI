import os
import json
import numpy as np
import torch
from tqdm import tqdm
from typer import Typer
from urban_planner.config import CONFIG
from src.dataset import create_dataloader

app = Typer()

@app.command()
def main(
    output_dir: str = "reports/tests/sensitivity",
    split: str = "test"
):
    """
    Generates a 'Ground Truth' sensitivity JSON by analyzing the actual dataset distribution.
    Calculates Mean and Std of targets (Temp, NDVI) binned by Latitude and Longitude.
    """
    CONFIG.device = "cpu" 
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Metrics for Un-normalization
    metrics_path = os.path.join(CONFIG.PROCESSED_IMAGE_DATASET, 'normalization_metrics.json')
    if not os.path.exists(metrics_path):
        print(f"Error: Metrics file not found at {metrics_path}")
        return
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    meta_mean = np.array(metrics['meta_mean'])
    meta_std = np.array(metrics['meta_std'])
    # meta indices: 0=Lat, 1=Lon, 2=Pop, 3=DeltaTime
    
    # 2. Load Dataset
    print(f"Loading {split} dataset for Ground Truth analysis...")
    # We use a large batch size as we are only collecting data, not running a model
    loader = create_dataloader(
        split=split,
        dataset_type=CONFIG.dataset.dataset_type,
        batch_size=256,
        shuffle=False,
        num_workers=4
    )
    
    target_channels = CONFIG.dataset.target_channels
    print(f"Target Channels: {target_channels}")
    
    # 3. Collect Data
    lats_all = []
    lons_all = []
    targets_all = [] # List of arrays (B, C)
    
    print("Collecting samples...")
    # batch: input_stack, metadata_orig, temp_series, _, t1_dates, t2_dates, targets
    for batch in tqdm(loader, desc="Reading Dataset"):
        _, metadata_orig, _, _, _, _, targets = batch
        
        # Metadata is normalized in the dataset
        # metadata_orig shape: (B, 4) [Lat, Lon, Pop, Time]
        # Un-normalize Lat/Lon
        batch_lats = metadata_orig[:, 0].numpy() * meta_std[0] + meta_mean[0]
        batch_lons = metadata_orig[:, 1].numpy() * meta_std[1] + meta_mean[1]
        
        lats_all.append(batch_lats)
        lons_all.append(batch_lons)
        
        # Targets are normalized
        targets_np = targets.numpy()
        unnorm_targets = np.zeros_like(targets_np)
        
        for i, ch in enumerate(target_channels):
            if 'temp' in ch.lower():
                unnorm_targets[:, i] = targets_np[:, i] * metrics['temp_std'] + metrics['temp_mean']
            else:
                # NDVI is usually [-1, 1], typically not normalized further or just passed through
                unnorm_targets[:, i] = targets_np[:, i]
        
        targets_all.append(unnorm_targets)
        
    lats_all = np.concatenate(lats_all)
    lons_all = np.concatenate(lons_all)
    targets_all = np.concatenate(targets_all, axis=0)
    
    print(f"Total samples collected: {len(lats_all)}")
    
    # 4. Binning and Statistics
    # Use same ranges as metadata_sensitivity.py for consistency
    lat_steps = 50
    lon_steps = 50
    lat_range = np.linspace(-60, 70, lat_steps)
    lon_range = np.linspace(-180, 180, lon_steps)
    
    sweeps_data = {
        "latitude": {"x": lat_range.tolist(), "channels": {}},
        "longitude": {"x": lon_range.tolist(), "channels": {}}
    }
    
    def compute_bin_stats(x_data, y_data, bin_centers):
        """
        Bins x_data according to bin_centers and computes mean/std of y_data in each bin.
        """
        # Create edges as midpoints between centers
        edges = np.concatenate([
            [bin_centers[0] - (bin_centers[1] - bin_centers[0])/2],
            (bin_centers[:-1] + bin_centers[1:]) / 2,
            [bin_centers[-1] + (bin_centers[-1] - bin_centers[-2])/2]
        ])
        
        # np.digitize returns indices where bins[i-1] <= x < bins[i]
        # indices 1..len(bin_centers) correspond to the bins
        indices = np.digitize(x_data, edges)
        
        means = []
        stds = []
        
        for i in range(1, len(bin_centers) + 1):
            mask = (indices == i)
            if np.any(mask):
                vals = y_data[mask]
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
            else:
                # Empty bin
                means.append(float('nan'))
                stds.append(float('nan'))
                
        return means, stds

    # --- Process Latitude ---
    print("Calculating stats vs Latitude...")
    for ch_i, ch_name in enumerate(target_channels):
        means, stds = compute_bin_stats(lats_all, targets_all[:, ch_i], lat_range)
        sweeps_data["latitude"]["channels"][ch_name] = {
            "mean": means,
            "std": stds
        }
        
    # --- Process Longitude ---
    print("Calculating stats vs Longitude...")
    for ch_i, ch_name in enumerate(target_channels):
        means, stds = compute_bin_stats(lons_all, targets_all[:, ch_i], lon_range)
        sweeps_data["longitude"]["channels"][ch_name] = {
            "mean": means,
            "std": stds
        }

    # 5. Export
    export_data = {
        "model_name": "Ground Truth (Dataset)",
        "model_type": "dataset",
        "sweeps": sweeps_data,
        "heatmaps": {} # No heatmaps for global GT
    }
    
    filename = "sensitivity_data_ground_truth.json"
    out_path = os.path.join(output_dir, filename)
    
    # allow_nan=True is default but explicit here for clarity (Standard JSON is null, but Python accepts NaN)
    with open(out_path, 'w') as f:
        json.dump(export_data, f, indent=4)
        
    print(f"Saved Ground Truth sensitivity data to: {out_path}")

if __name__ == "__main__":
    app()
