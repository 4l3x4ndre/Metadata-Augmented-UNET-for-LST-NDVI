import os
from typing import List
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typer import Typer
import json
from tqdm import tqdm
from omegaconf import OmegaConf
import wandb
import random
import seaborn as sns

from urban_planner.config import CONFIG
from src.dataset import create_dataloader
from src.model import UrbanPredictor
from src.utils.plot_utils import DATASET_COLORS_5V5, get_styled_figure_ax, style_legend, convert_label

app = Typer()

def get_unnormalized_data(targets, outputs, metrics):
    """Un-normalizes temperature data for visualization and metric calculation."""
    if not metrics:
        return targets, outputs

    unnorm_targets = np.zeros_like(targets)
    unnorm_outputs = np.zeros_like(outputs)
    target_channel_names = CONFIG.dataset.target_channels

    for i, ch in enumerate(target_channel_names):
        if 'temp' in ch.lower():
            unnorm_targets[:, i] = targets[:, i] * metrics['temp_std'] + metrics['temp_mean']
            unnorm_outputs[:, i] = outputs[:, i] * metrics['temp_std'] + metrics['temp_mean']
        else:  # NDVI [-1, 1] range channel is not normalized
            unnorm_targets[:, i] = targets[:, i]
            unnorm_outputs[:, i] = outputs[:, i]
            
    return unnorm_targets, unnorm_outputs

@app.command()
def main(
    checkpoint_path: str,
    evaluation_csv: str,
    jobid: int = 0,
    device: str = '',
    output_dir: str = "reports/tests/sensitivity",
    wandblog: bool = False,
    study_name: str = "sensitivity_analysis"
):
    # --- Setup ---
    if device:
        if device.lower() == 'gpu' and torch.cuda.is_available():
            CONFIG.device = "cuda:0"
        else:
            CONFIG.device = device
    else:
         CONFIG.device = "cuda:0" if torch.cuda.is_available() else "cpu"
         
    logger_device = CONFIG.device
    print(f"Using device: {logger_device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Load Metrics ---
    metrics_path = os.path.join(CONFIG.PROCESSED_IMAGE_DATASET, 'normalization_metrics.json')
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found at {metrics_path}")
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    meta_mean = np.array(metrics['meta_mean'])
    meta_std = np.array(metrics['meta_std'])
    # meta indices: 0=Lat, 1=Lon, 2=Pop, 3=DeltaTime

    # --- Load Checkpoint ---
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=CONFIG.device)
    hyperparams = checkpoint.get('hyperparameters', {})
    
    # Resolve embedding flags
    if 'temporal_embeddings' in hyperparams:
        temporal_embeddings = hyperparams['temporal_embeddings']
        metadata_embeddings = hyperparams['metadata_embeddings']
    else:
        # Legacy fallback
        default_emb = True
        checkpoint_study = checkpoint.get('study_name', '')
        # Check if 'noemb' is in study_name (arg) or checkpoint study name
        if 'noemb' in study_name or 'noemb' in checkpoint_study:
            default_emb = False

        additional_embeddings = checkpoint.get('additional_embeddings', default_emb)
        metadata_only_embeddings = checkpoint.get('metadata_only_embeddings', False)
        if additional_embeddings:
            temporal_embeddings = True
            metadata_embeddings = True
        elif metadata_only_embeddings:
            temporal_embeddings = False
            metadata_embeddings = True
        else:
            temporal_embeddings = False
            metadata_embeddings = False

    model_type = checkpoint.get('model_type', 'unet')
    metadata_input_length = checkpoint.get('metadata_input_length', 4)
    model_name = ''
    if temporal_embeddings and metadata_embeddings:
        model_name = 'emb'
    elif metadata_embeddings:
        model_name = 'metaemb'
    elif temporal_embeddings:
        model_name = 'tempemb'
    else:
        model_name = 'noemb'
    if '++' in model_type:
        model_name += '++'

    model = UrbanPredictor(
        model_type=model_type,
        spatial_channels=CONFIG.dataset.nb_input_channels,
        seq_len=CONFIG.dataset.temporal_length,
        temporal_dim=hyperparams.get('temporal_dim', 16),
        meta_features=metadata_input_length,
        meta_dim=hyperparams.get('meta_dim', 8),
        lstm_dim=hyperparams.get('lstm_hidden', 32),
        out_channels=len(CONFIG.dataset.target_channels),
        temporal_embeddings=temporal_embeddings,
        metadata_embeddings=metadata_embeddings
    ).to(CONFIG.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # --- Select Samples from CSV ---
    print(f"Loading evaluation report: {evaluation_csv}")
    df_eval = pd.read_csv(evaluation_csv)
    
    # Filter for overall metrics
    df_overall = df_eval[df_eval['dw_class'] == 'overall'].copy()
    
    # 1. Aggregate RMSE by City to find Best/Worst Cities
    df_city_agg = df_overall.groupby('city')['rmse'].mean().reset_index()
    df_city_agg = df_city_agg.sort_values('rmse')
    
    best_cities = df_city_agg.head(5)['city'].tolist()
    worst_cities = df_city_agg.tail(5)['city'].tolist()
    
    # 2. Select one representative sample per city
    target_indices = []
    sample_groups = {}
    
    # Helper to pick a sample for a city
    def get_sample_for_city(city_name):
        # Pick the sample with the median RMSE for this city to be representative
        city_samples = df_overall[df_overall['city'] == city_name]
        # Sort by RMSE within the city
        city_samples = city_samples.sort_values('rmse')
        # Pick middle
        middle_idx = len(city_samples) // 2
        return city_samples.iloc[middle_idx]['sample_idx']

    for city in best_cities:
        s_idx = get_sample_for_city(city)
        target_indices.append(s_idx)
        sample_groups[s_idx] = 'Best'
        
    for city in worst_cities:
        s_idx = get_sample_for_city(city)
        target_indices.append(s_idx)
        sample_groups[s_idx] = 'Worst'
    
    print(f"Selected {len(target_indices)} samples for individual highlighting.")
    
    # --- Prepare for All-Sample Analysis ---
    all_indices = df_overall['sample_idx'].unique().tolist()
    
    # Downsample for average calculation if too many samples
    MAX_AVG_SAMPLES = 1000
    if len(all_indices) > MAX_AVG_SAMPLES:
        print(f"Downsampling from {len(all_indices)} to {MAX_AVG_SAMPLES} samples for average sensitivity plots.")
        # Ensure target_indices are kept
        remaining_indices = list(set(all_indices) - set(target_indices))
        n_to_sample = min(len(remaining_indices), MAX_AVG_SAMPLES - len(target_indices))
        sampled_indices = random.sample(remaining_indices, n_to_sample)
        all_indices = target_indices + sampled_indices

    print(f"Total samples to analyze for average: {len(all_indices)}")

    # --- Load Data Subset ---
    base_loader = create_dataloader(
        split='test',
        dataset_type=CONFIG.dataset.dataset_type,
        batch_size=1,
        shuffle=False, # Must be False to align indices
        num_workers=0
    )
    
    from torch.utils.data import Subset, DataLoader
    from src.dataset import collate_fn
    
    # We iterate over ALL samples now
    subset = Subset(base_loader.dataset, all_indices)
    subset_loader = DataLoader(subset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # --- WandB ---
    if wandblog:
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            name=f"{study_name}_eval",
            group=study_name,
            tags=["sensitivity", model_type],
            config=hyperparams
        )

    # --- Sweep Configurations ---
    lat_steps = 50
    lon_steps = 50
    lat_range = np.linspace(-60, 70, lat_steps) # Avoid extreme poles
    lon_range = np.linspace(-180, 180, lon_steps)

    target_channels = CONFIG.dataset.target_channels
    
    # Store aggregated results for individual highlighted samples
    results_lat = []
    results_lon = []
    
    # Store accumulated results for averaging [sample_idx, step_idx, channel_idx]
    # Structure: lat_accum[channel_name] = list of arrays of shape (lat_steps,)
    lat_accum = {ch: [] for ch in target_channels}
    lon_accum = {ch: [] for ch in target_channels}

    # --- Determine Status Uniformity ---
    unique_statuses = set()
    for idx in target_indices:
        # Get first row for this sample to check status
        val = df_overall[df_overall['sample_idx'] == idx]['is_known_city'].iloc[0] if 'is_known_city' in df_overall.columns else False
        unique_statuses.add(bool(val))
    
    is_mixed_status = len(unique_statuses) > 1
    global_status_str = ""
    if not is_mixed_status and unique_statuses:
         global_status_str = "known" if list(unique_statuses)[0] else "unknown"

    # --- Analysis Loop ---
    
    results_heatmap = [] # Store data for heatmaps

    # subset_loader iterates over all_indices
    for i, batch in enumerate(tqdm(subset_loader, desc="Analyzing Samples")):
        current_sample_idx = all_indices[i]
        
        # Check if this sample is one of the few we want to highlight/plot individually
        is_highlight = current_sample_idx in target_indices
        group_label = sample_groups.get(current_sample_idx, 'Other')
        
        input_stack, metadata_orig, temp_series, _, t1_dates, t2_dates, targets = batch
        
        # Metadata un-normalization
        orig_lat_norm = metadata_orig[0, 0].item()
        orig_lon_norm = metadata_orig[0, 1].item()
        orig_lat = orig_lat_norm * meta_std[0] + meta_mean[0]
        orig_lon = orig_lon_norm * meta_std[1] + meta_mean[1]
        
        if is_highlight:
            # Retrieve city info from CSV for better labeling
            sample_row = df_overall[df_overall['sample_idx'] == current_sample_idx].iloc[0]
            city_name = sample_row['city'].title()
            is_known = bool(sample_row.get('is_known_city', False))
            
            df_agg_sample = df_overall[df_overall['sample_idx'] == current_sample_idx].groupby('sample_idx')['rmse'].mean().reset_index()
            base_rmse = df_agg_sample['rmse'].values[0]
            
            if is_mixed_status:
                known_str = "Known\\ city" if is_known else "Unknown\\ city"
                sample_label = f"{group_label} {city_name} RMSE={base_rmse:.3f}\n$\\mathit{{{known_str}}}$"
            else:
                sample_label = f"{group_label} {city_name} RMSE={base_rmse:.3f}"
        else:
            sample_label = f"Sample {current_sample_idx}"

        # Capture Ground Truth Values
        targets_np = targets.cpu().numpy()
        _, unnorm_targets = get_unnormalized_data(targets_np, targets_np, metrics)
        gt_values = {}
        for ch_i, ch_name in enumerate(target_channels):
            gt_values[ch_name] = np.mean(unnorm_targets[0, ch_i])

        # --- Latitude Sweep ---
        # Prepare batch for sweep to speed up
        # Create (lat_steps, input_channels...)
        
        # Repeat inputs: (lat_steps, ...)
        meta_batch = metadata_orig.repeat(lat_steps, 1)
        # Modify latitude column
        lat_norms = (lat_range - meta_mean[0]) / meta_std[0]
        meta_batch[:, 0] = torch.tensor(lat_norms, device=CONFIG.device, dtype=meta_batch.dtype)
        
        if metadata_input_length == 8:
            t1_batch = t1_dates.repeat(lat_steps, 1)
            t2_batch = t2_dates.repeat(lat_steps, 1)
            meta_full_batch = torch.cat([meta_batch, t1_batch, t2_batch], dim=1)
        else:
            meta_full_batch = meta_batch
            
        input_stack_batch = input_stack.repeat(lat_steps, 1, 1, 1)
        temp_series_batch = temp_series.repeat(lat_steps, 1)
        
        with torch.no_grad():
            outputs = model(input_stack_batch, temp_series_batch, meta_full_batch)
            outputs_np = outputs.cpu().numpy()
            # Dummy targets for unnorm (shape match)
            dummy_targets = np.zeros_like(outputs_np)
            _, unnorm_outputs_batch = get_unnormalized_data(dummy_targets, outputs_np, metrics)
            
            # Store results
            for ch_i, ch_name in enumerate(target_channels):
                # Average over spatial dims if needed (prediction is usually (B, C, H, W) or (B, C))
                # Assuming output is (B, C) or we average over H,W
                if unnorm_outputs_batch.ndim == 4:
                    vals = np.mean(unnorm_outputs_batch[:, ch_i, :, :], axis=(1, 2))
                else:
                    vals = unnorm_outputs_batch[:, ch_i]
                
                lat_accum[ch_name].append(vals)

                if is_highlight:
                    for step_i, lat_val in enumerate(lat_range):
                         results_lat.append({
                            'sample_label': sample_label, 
                            'group': group_label, 
                            'latitude': lat_val, 
                            'orig_lat': orig_lat,
                            'orig_lon': orig_lon,
                            ch_name: vals[step_i],
                            f"{ch_name}_gt": gt_values[ch_name]
                        })

        # --- Longitude Sweep ---
        meta_batch = metadata_orig.repeat(lon_steps, 1)
        lon_norms = (lon_range - meta_mean[1]) / meta_std[1]
        meta_batch[:, 1] = torch.tensor(lon_norms, device=CONFIG.device, dtype=meta_batch.dtype)

        if metadata_input_length == 8:
            t1_batch = t1_dates.repeat(lon_steps, 1)
            t2_batch = t2_dates.repeat(lon_steps, 1)
            meta_full_batch = torch.cat([meta_batch, t1_batch, t2_batch], dim=1)
        else:
            meta_full_batch = meta_batch
            
        input_stack_batch = input_stack.repeat(lon_steps, 1, 1, 1)
        temp_series_batch = temp_series.repeat(lon_steps, 1)
        
        with torch.no_grad():
            outputs = model(input_stack_batch, temp_series_batch, meta_full_batch)
            outputs_np = outputs.cpu().numpy()
            dummy_targets = np.zeros_like(outputs_np)
            _, unnorm_outputs_batch = get_unnormalized_data(dummy_targets, outputs_np, metrics)
            
            for ch_i, ch_name in enumerate(target_channels):
                if unnorm_outputs_batch.ndim == 4:
                    vals = np.mean(unnorm_outputs_batch[:, ch_i, :, :], axis=(1, 2))
                else:
                    vals = unnorm_outputs_batch[:, ch_i]
                
                lon_accum[ch_name].append(vals)

                if is_highlight:
                    for step_i, lon_val in enumerate(lon_range):
                        results_lon.append({
                            'sample_label': sample_label, 
                            'group': group_label, 
                            'longitude': lon_val,
                            'orig_lat': orig_lat,
                            'orig_lon': orig_lon,
                            ch_name: vals[step_i],
                            f"{ch_name}_gt": gt_values[ch_name]
                        })

        # --- 2D Heatmap Sweep ---
        # Only run heatmap for highlighted samples to save time/space
        if is_highlight:
            lat_steps_2d = 20
            lon_steps_2d = 20
            lat_range_2d = np.linspace(-60, 70, lat_steps_2d)
            lon_range_2d = np.linspace(-180, 180, lon_steps_2d)

            
            lats_2d, lons_2d = np.meshgrid(lat_range_2d, lon_range_2d, indexing='ij')
            lats_flat = lats_2d.flatten()
            lons_flat = lons_2d.flatten()
            n_grid = len(lats_flat)
            
            meta_batch = metadata_orig.repeat(n_grid, 1)
            lat_norms_2d = (lats_flat - meta_mean[0]) / meta_std[0]
            lon_norms_2d = (lons_flat - meta_mean[1]) / meta_std[1]
            
            meta_batch[:, 0] = torch.tensor(lat_norms_2d, device=CONFIG.device, dtype=meta_batch.dtype)
            meta_batch[:, 1] = torch.tensor(lon_norms_2d, device=CONFIG.device, dtype=meta_batch.dtype)

            if metadata_input_length == 8:
                t1_batch = t1_dates.repeat(n_grid, 1)
                t2_batch = t2_dates.repeat(n_grid, 1)
                meta_full_batch = torch.cat([meta_batch, t1_batch, t2_batch], dim=1)
            else:
                meta_full_batch = meta_batch

            heatmap_batch_size = 50
            
            for b_start in range(0, n_grid, heatmap_batch_size):
                b_end = min(b_start + heatmap_batch_size, n_grid)
                curr_batch_size = b_end - b_start
                
                meta_mini = meta_full_batch[b_start:b_end]
                input_mini = input_stack.repeat(curr_batch_size, 1, 1, 1)
                temp_mini = temp_series.repeat(curr_batch_size, 1)
                
                with torch.no_grad():
                    outputs = model(input_mini, temp_mini, meta_mini)
                    outputs_np = outputs.cpu().numpy()
                    _, unnorm_outputs_mini = get_unnormalized_data(np.zeros_like(outputs_np), outputs_np, metrics)
                    
                    for local_i in range(curr_batch_size):
                        global_i = b_start + local_i
                        lat_val = lats_flat[global_i]
                        lon_val = lons_flat[global_i]
                        
                        row = {
                            'sample_idx': current_sample_idx,
                            'sample_label': sample_label,
                            'latitude': lat_val,
                            'longitude': lon_val,
                            'orig_lat': orig_lat,
                            'orig_lon': orig_lon
                        }
                        
                        for ch_i, ch_name in enumerate(target_channels):
                            if unnorm_outputs_mini.ndim == 4:
                                val = np.mean(unnorm_outputs_mini[local_i, ch_i, :, :])
                            else:
                                val = unnorm_outputs_mini[local_i, ch_i]
                            row[ch_name] = val
                        
                        results_heatmap.append(row)


    # --- Plotting Average Sensitivity ---
    # Calculate stats
    def plot_avg_sensitivity(accum_dict, x_range, x_label, filename_suffix):
        paths = []
        for ch_name in target_channels:
            # Stack: (num_samples, num_steps)
            data_stack = np.stack(accum_dict[ch_name])
            
            mean_curve = np.mean(data_stack, axis=0)
            std_curve = np.std(data_stack, axis=0)
            
            fig, ax = get_styled_figure_ax(figsize=(10, 6), grid=True)
            
            color = 'tab:green'
            ax.plot(x_range, mean_curve, color=color, linewidth=2, label='Average Response')
            ax.fill_between(x_range, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.2, label='±1 Std Dev')
            
            # Zoom in on the mean curve to show sensitivity, ignoring the large variance between samples
            y_range = mean_curve.max() - mean_curve.min()
            if y_range < 1e-6:
                y_pad = 0.1
            else:
                y_pad = y_range * 0.2
            
            ax.set_ylim(mean_curve.min() - y_pad, mean_curve.max() + y_pad)
            
            ax.set_xlabel(f"{x_label} (degree)")
            ax.set_ylabel(convert_label(f"Predicted {ch_name}"))
            # ax.set_title(f"Average Sensitivity to {x_label} (All Samples)")
            style_legend(ax)
            
            plt.tight_layout()
            fname = f"sensitivity_{model_name}_AVG_{filename_suffix}_{ch_name}.pdf"
            path = os.path.join(output_dir, fname)
            plt.savefig(path, dpi=150)
            print(f"Saved average plot to {path}")
            paths.append(path)
            plt.close(fig)
        return paths

    path_avg_lat = plot_avg_sensitivity(lat_accum, lat_range, "Latitude", "latitude")
    path_avg_lon = plot_avg_sensitivity(lon_accum, lon_range, "Longitude", "longitude")

    # --- Plotting 1D Individual (Existing Logic) ---
    df_lat = pd.DataFrame(results_lat)
    df_lon = pd.DataFrame(results_lon)
    
    # Helper function to plot 1D
    def plot_sensitivity(df, x_col, x_label, filename_suffix) -> List[str]:
        # fig, axes = plt.subplots(1, len(target_channels), figsize=(15, 5))
        paths = []
        
        for i, ch_name in enumerate(target_channels):
            fig, ax = get_styled_figure_ax(figsize=(17, 8), aspect='none', grid=True)
            
            unique_labels = df['sample_label'].unique()
            # colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            colors = DATASET_COLORS_5V5
            
            for j, label in enumerate(unique_labels):
                subset = df[df['sample_label'] == label]
                
                # Determine line style based on group
                group = subset['group'].iloc[0]
                linestyle = '-' if group == 'Best' else '--'
                
                # Plot predicted curve
                ax.plot(subset[x_col], subset[ch_name], linestyle=linestyle, alpha=0.9, color=colors[j], label=label, linewidth=2)
                
                # Ground Truth Logic
                orig_x = subset[f'orig_lat' if x_col == 'latitude' else 'orig_lon'].iloc[0]
                gt_val = subset[f'{ch_name}_gt'].iloc[0]
                
                ax.scatter([orig_x], [gt_val], color=colors[j], s=30, zorder=5, marker='o', edgecolors='black')

            # ax.set_title(f"Impact of {x_label} on {ch_name}")
            ax.set_xlabel(f"{x_label} (degree)")
            ax.set_ylabel(convert_label(f"Predicted Mean {ch_name}"))
            # Set axis limits based on data range for y with padding:
            data_min = df[ch_name].min()
            data_max = df[ch_name].max()
            data_range = data_max - data_min
            
            if data_range == 0:
                padding = abs(data_min) * 0.05 if data_min != 0 else 0.05
                y_min = data_min - padding
                y_max = data_max + padding
            else:
                padding = 0.05 * data_range
                y_min = data_min - padding
                y_max = data_max + padding
            
            ax.set_ylim(y_min, y_max)
            # ax.legend(fontsize='x-small', bbox_to_anchor=(1.05, 1), loc='upper left')
            style_legend(ax, ncol=1, loc="center left", bbox_to_anchor=(1, 0.5))
            ax.grid(True, linestyle=':', alpha=0.6)
        
            plt.tight_layout()
            fname_suffix_full = filename_suffix
            if global_status_str:
                fname_suffix_full = f"{filename_suffix}_{global_status_str}"
            path = os.path.join(output_dir, f"sensitivity_{model_name}_{fname_suffix_full}_{ch_name}.pdf")
            plt.savefig(path, dpi=150)
            print(f"Saved plot to {path}")
            paths.append(path)
        return paths

    path_lat = []
    if not df_lat.empty:
        path_lat = plot_sensitivity(df_lat, 'latitude', 'Latitude', 'latitude')
    else:
        print("No highlighted samples found for Latitude sensitivity plot.")

    path_lon = []
    if not df_lon.empty:
        path_lon = plot_sensitivity(df_lon, 'longitude', 'Longitude', 'longitude')
    else:
        print("No highlighted samples found for Longitude sensitivity plot.")

    # --- Plotting 2D Heatmaps ---
    df_heatmap = pd.DataFrame(results_heatmap)
    heatmap_paths = []
    
    unique_indices = []
    if not df_heatmap.empty and 'sample_idx' in df_heatmap.columns:
        unique_indices = df_heatmap['sample_idx'].unique()
    else:
        print("No data for heatmaps.")
    
    for sample_idx in unique_indices:
        subset = df_heatmap[df_heatmap['sample_idx'] == sample_idx]
        sample_label = subset['sample_label'].iloc[0]
        orig_lat = subset['orig_lat'].iloc[0]
        orig_lon = subset['orig_lon'].iloc[0]
        
        fig, axes = plt.subplots(1, len(target_channels), figsize=(8 * len(target_channels), 7))
        if len(target_channels) == 1: axes = [axes]
        # fig.suptitle(f"2D Sensitivity: {sample_label}")

        for i, ch_name in enumerate(target_channels):
            ax = axes[i]
            
            # Pivot for heatmap
            pivot_table = subset.pivot(index='latitude', columns='longitude', values=ch_name)
            
            # Plot heatmap
            # extent = [left, right, bottom, top]
            extent = [pivot_table.columns.min(), pivot_table.columns.max(), pivot_table.index.min(), pivot_table.index.max()]
            if 'ndvi' in ch_name.lower():
                color_map = sns.color_palette("crest", as_cmap=True)
            else:
                color_map = sns.color_palette("RdBu_r", as_cmap=True)

            im = ax.imshow(pivot_table, extent=extent, origin='lower', aspect='auto', cmap=color_map)
            plt.colorbar(im, ax=ax, label=convert_label(f"Predicted {ch_name}"))
            
            # Crosshairs for original location
            ax.axhline(y=orig_lat, color='red', linestyle='--', alpha=0.8)
            ax.axvline(x=orig_lon, color='red', linestyle='--', alpha=0.8)
            ax.scatter([orig_lon], [orig_lat], color='red', marker='x', s=100, label='Original Loc')

            # ax.set_title(convert_label(f"{ch_name} Map"))
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        # Sanitize label for filename
        safe_label = sample_label.replace('\n', '_').replace(' ', '_').replace('$', '').replace('\\', '').replace('{', '').replace('}', '')
        
        heatmap_fname = f"sensitivity_{model_name}_heatmap_{sample_idx}_{safe_label}"
        if global_status_str:
             heatmap_fname += f"_{global_status_str}"
        path = os.path.join(output_dir, f"{heatmap_fname}.pdf")
        plt.savefig(path, dpi=150)
        heatmap_paths.append(path)
        plt.close(fig)
        print(f"Saved heatmap to {path}")

    # --- Export Data for Comparison ---
    print("Exporting sensitivity data for comparison...")
    export_data = {
        "model_name": model_name,
        "model_type": model_type,
        "sweeps": {
            "latitude": {"x": lat_range.tolist(), "channels": {}},
            "longitude": {"x": lon_range.tolist(), "channels": {}}
        }
    }

    def process_accumulation_for_export(accum_dict, key):
        for ch_name in target_channels:
            data_stack = np.stack(accum_dict[ch_name])
            # Calculate stats
            mean_curve = np.mean(data_stack, axis=0)
            std_curve = np.std(data_stack, axis=0)
            
            export_data["sweeps"][key]["channels"][ch_name] = {
                "mean": mean_curve.tolist(),
                "std": std_curve.tolist()
            }

    process_accumulation_for_export(lat_accum, "latitude")
    process_accumulation_for_export(lon_accum, "longitude")

    # Export Heatmap Data
    heatmap_export = {}
    if results_heatmap:
        df_hm = pd.DataFrame(results_heatmap)
        unique_indices = df_hm['sample_idx'].unique()
        
        for idx in unique_indices:
            subset = df_hm[df_hm['sample_idx'] == idx]
            sample_label = subset['sample_label'].iloc[0]
            orig_lat = subset['orig_lat'].iloc[0]
            orig_lon = subset['orig_lon'].iloc[0]
            
            heatmap_export[str(idx)] = {
                "sample_label": sample_label,
                "orig_lat": float(orig_lat),
                "orig_lon": float(orig_lon),
                "channels": {}
            }
            
            for ch_name in target_channels:
                # pivot index is lat (ascending), columns is lon
                pivot = subset.pivot(index='latitude', columns='longitude', values=ch_name)
                heatmap_export[str(idx)]["channels"][ch_name] = {
                    "values": pivot.values.tolist(),
                    "lats": pivot.index.tolist(),
                    "lons": pivot.columns.tolist()
                }
    export_data["heatmaps"] = heatmap_export

    export_path = os.path.join(output_dir, f"sensitivity_data_{model_name}.json")
    with open(export_path, 'w') as f:
        json.dump(export_data, f, indent=4)
    print(f"Saved sensitivity data to {export_path}")

    if wandblog:
        log_dict = {}
        for i, path in enumerate(path_lat):
            ch_name = target_channels[i]
            log_dict[f"sensitivity/latitude/{ch_name}"] = wandb.Image(path)
        for i, path in enumerate(path_lon):
            ch_name = target_channels[i]
            log_dict[f"sensitivity/longitude/{ch_name}"] = wandb.Image(path)
            
        # Log Average Plots
        for i, path in enumerate(path_avg_lat):
            ch_name = target_channels[i]
            log_dict[f"sensitivity/latitude_avg/{ch_name}"] = wandb.Image(path)
        for i, path in enumerate(path_avg_lon):
            ch_name = target_channels[i]
            log_dict[f"sensitivity/longitude_avg/{ch_name}"] = wandb.Image(path)
            
        # log_dict = {
        #     "sensitivity/latitude": wandb.Image(path_lat),
        #     "sensitivity/longitude": wandb.Image(path_lon)
        # }
        for p in heatmap_paths:
            # Use filename as key to distinguish samples
            fname = os.path.basename(p)
            log_dict[f"sensitivity/heatmaps/{fname}"] = wandb.Image(p)
            
        wandb.log(log_dict)
        wandb.finish()

if __name__ == "__main__":
    app()
