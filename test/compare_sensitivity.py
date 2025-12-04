import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typer import Typer
from typing import List
import matplotlib.colors as mcolors

from src.utils.plot_utils import get_styled_figure_ax, style_legend, convert_label, DATASET_COLORS_5

app = Typer()

def get_model_color(idx, total):
    """Generate a distinct color from a high-contrast palette."""
    cmap = mcolors.ListedColormap(DATASET_COLORS_5)
    return cmap(idx % cmap.N)

@app.command()
def main(
    input_dir: str = "reports/tests/sensitivity",
    output_dir: str = "reports/tests/sensitivity/comparison"
):
    """
    Compare sensitivity analysis results from multiple models.
    Reads 'sensitivity_data_*.json' files from input_dir and generates combined plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Find and Load Data
    pattern = os.path.join(input_dir, "sensitivity_data_*.json")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No sensitivity data files found in {input_dir}")
        return

    print(f"Found {len(files)} model files. Loading...")
    
    models_data = []
    for fpath in files:
        with open(fpath, 'r') as f:
            try:
                data = json.load(f)
                models_data.append(data)
                print(f"  Loaded: {data.get('model_name', 'Unknown')}")
            except json.JSONDecodeError:
                print(f"  Error loading {fpath}")

    if not models_data:
        return

    # 2. Identify Channels and Sweeps
    ref_model = models_data[0]
    sweeps = ["latitude", "longitude"]
    channels = list(ref_model["sweeps"]["latitude"]["channels"].keys())

    # 3. Plotting Loop
    for sweep_key in sweeps:
        for ch_name in channels:
            print(f"Generating comparison plot for {sweep_key} - {ch_name}...")
            
            fig, ax = get_styled_figure_ax(figsize=(12, 7), grid=True, aspect='auto')
            
            # Track min/max for zooming
            global_min = float('inf')
            global_max = float('-inf')
            
            for i, model in enumerate(models_data):
                model_name = model.get("model_name", f"Model {i}")
                sweep_data = model["sweeps"][sweep_key]
                
                if ch_name not in sweep_data["channels"]:
                    print(f"  Warning: Channel {ch_name} not found in model {model_name}")
                    continue
                
                x = sweep_data["x"]
                y_mean = np.array(sweep_data["channels"][ch_name]["mean"])
                y_std = np.array(sweep_data["channels"][ch_name]["std"]) 
                
                color = get_model_color(i, len(models_data))
                
                ax.plot(x, y_mean, color=color, linewidth=2.5, label=model_name)
                # Add shading for variance
                ax.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=0.1)
                
                # Update range stats to include the full extent of mean +/- std
                global_min = min(global_min, (y_mean - y_std).min())
                global_max = max(global_max, (y_mean + y_std).max())

            # Formatting
            ax.set_xlabel(f"{sweep_key.capitalize()} (degree)")
            if "temperature" in ch_name.lower():
                ax.set_ylabel(convert_label(f"Predicted {ch_name} (°C)"))
            else:
                ax.set_ylabel(convert_label(f"Predicted {ch_name}"))
            # ax.set_title(f"Model Sensitivity Comparison: {sweep_key.capitalize()} ({convert_label(ch_name)})")
            
            # Auto-Scale Y-Axis (Zoom logic)
            y_range = global_max - global_min
            if y_range < 1e-6:
                y_pad = 0.15 
            else:
                y_pad = y_range * 0.15 # 5% padding
            
            ax.set_ylim(global_min - y_pad, global_max + y_pad)
            # ax.set_ylim(global_min, global_max)
            print(f"  Y-axis range: {global_min:.4f} to {global_max:.4f}")
            
            style_legend(ax, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
            
            # Save
            filename = f"comparison_{sweep_key}_{ch_name}.pdf"
            out_path = os.path.join(output_dir, filename)
            if os.path.exists(out_path):
                os.remove(out_path)
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            print(f"  Saved to {out_path}")
            plt.close(fig)

    # 4. Heatmap Generation (Average)
    print("Generating average heatmaps...")
    for i, model in enumerate(models_data):
        model_name = model.get("model_name", f"Model {i}")
        heatmaps_data = model.get("heatmaps", {})
        
        if not heatmaps_data:
            print(f"  No heatmap data found for model {model_name}")
            continue
            
        aggregated_grids = {}
        grid_coords = {} # Stores (lats, lons) for extent

        # Accumulate data
        for sample_idx, sample_data in heatmaps_data.items():
            channels_data = sample_data.get("channels", {})
            
            for ch_name, ch_data in channels_data.items():
                values = np.array(ch_data["values"])
                lats = np.array(ch_data["lats"])
                lons = np.array(ch_data["lons"])
                
                if ch_name not in aggregated_grids:
                    aggregated_grids[ch_name] = []
                    grid_coords[ch_name] = (lats, lons)
                
                aggregated_grids[ch_name].append(values)
        
        if not aggregated_grids:
            continue

        # Plot Average
        channel_names = list(aggregated_grids.keys())
        fig, axes = plt.subplots(1, len(channel_names), figsize=(8 * len(channel_names), 7))
        if len(channel_names) == 1: axes = [axes]
        
        for ch_i, ch_name in enumerate(channel_names):
            ax = axes[ch_i]
            
            grids = aggregated_grids[ch_name]
            if not grids: continue
            
            # Compute Mean Grid
            mean_grid = np.mean(np.stack(grids), axis=0)
            lats, lons = grid_coords[ch_name]
            
            # extent = [left, right, bottom, top]
            extent = [lons.min(), lons.max(), lats.min(), lats.max()]
            
            if 'ndvi' in ch_name.lower():
                color_map = sns.color_palette("crest", as_cmap=True)
            else:
                color_map = sns.color_palette("RdBu_r", as_cmap=True)

            im = ax.imshow(mean_grid, extent=extent, origin='lower', aspect='auto', cmap=color_map)
            plt.colorbar(im, ax=ax, label=convert_label(f"Avg Predicted {ch_name}"))
            
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            # ax.set_title(f"Average Sensitivity: {convert_label(ch_name)}")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        heatmap_fname = f"comparison_heatmap_{model_name}_AVERAGE.pdf"
        out_path = os.path.join(output_dir, heatmap_fname)
        
        plt.savefig(out_path, dpi=150)
        print(f"  Saved average heatmap to {out_path}")
        plt.close(fig)

if __name__ == "__main__":
    app()
