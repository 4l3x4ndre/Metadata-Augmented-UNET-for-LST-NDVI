import os
import json
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from urban_planner.config import CONFIG

def plot_predictions_vs_targets(input_stack, metadata, temp_series, t1_dates, t2_dates, outputs, targets, study_name, trial_id, step, batch_loss):
    """
    Plots predictions and targets for the first sample in a batch.
    """
    os.makedirs("reports/training/visualizations", exist_ok=True)

    # Use only the first sample in the batch
    input_stack = input_stack[0].cpu().numpy()
    metadata_normalized = metadata[0].cpu().numpy()
    temp_series_normalized = temp_series[0].cpu().numpy()
    outputs = outputs[0].detach().cpu().numpy()
    targets = targets[0].detach().cpu().numpy()
    t1_year = int(t1_dates[0, 0].item())
    t1_month = int(t1_dates[0, 1].item())
    t1_date_str = f"{t1_year}-{t1_month:02d}"

    t2_year = int(t2_dates[0, 0].item())
    t2_month = int(t2_dates[0, 1].item())
    t2_date_str = f"{t2_year}-{t2_month:02d}"

    # Get min/max to log in figure:
    output_ndvi_min = np.min(outputs[CONFIG.dataset.target_channels.index('after_ndvi')])
    output_ndvi_max = np.max(outputs[CONFIG.dataset.target_channels.index('after_ndvi')])
    output_temp_min = np.min(outputs[CONFIG.dataset.target_channels.index('after_temp')])
    output_temp_max = np.max(outputs[CONFIG.dataset.target_channels.index('after_temp')])
    min_max_values = [(output_ndvi_min, output_ndvi_max), (output_temp_min, output_temp_max)]

    # Load normalization metrics
    metrics_path = os.path.join(CONFIG.PROCESSED_IMAGE_DATASET, 'normalization_metrics.json')
    if not os.path.exists(metrics_path):
        logger.warning(f"Normalization metrics not found at {metrics_path}. Cannot un-normalize.")
        metrics = None
    else:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

    # --- Un-normalize data for visualization ---
    if metrics:
        # Input stack
        dw_t1_image = np.argmax(input_stack[:9], axis=0)
        
        rgb_normalized = input_stack[9:12]
        rgb_image = (rgb_normalized * np.array(metrics['rgb_std'])[:, np.newaxis, np.newaxis] + np.array(metrics['rgb_mean'])[:, np.newaxis, np.newaxis]) * 255.0
        rgb_image = np.clip(rgb_image.transpose(1, 2, 0), 0, 255).astype(np.uint8)

        ndvi_image = input_stack[12] # NDVI is not normalized in visualize_npz
        
        temp_normalized = input_stack[13]
        temp_image = temp_normalized * metrics['temp_std'] + metrics['temp_mean']
        
        dw_t2_image = np.argmax(input_stack[14:23], axis=0)

        # Target stack (ground truth)
        unnorm_targets = np.zeros_like(targets)
        target_channel_names = CONFIG.dataset.target_channels
        for i, ch in enumerate(target_channel_names):
            if 'ndvi' in ch.lower():
                unnorm_targets[i] = targets[i] # not normalized
            elif 'temp' in ch.lower():
                unnorm_targets[i] = targets[i] * metrics['temp_std'] + metrics['temp_mean']
            else:
                unnorm_targets[i] = targets[i]

        # Predictions (outputs)
        unnorm_outputs = np.zeros_like(outputs)
        for i, ch in enumerate(target_channel_names):
            if 'ndvi' in ch.lower():
                unnorm_outputs[i] = outputs[i]
            elif 'temp' in ch.lower():
                unnorm_outputs[i] = outputs[i] * metrics['temp_std'] + metrics['temp_mean']
            else:
                unnorm_outputs[i] = outputs[i]

        # Metadata
        metadata = metadata_normalized * metrics['meta_std'] + metrics['meta_mean']
        
        # Temperature series
        temp_series = temp_series_normalized * metrics['temp_series_std'] + metrics['temp_series_mean']
    else:
        # Fallback if metrics not found
        dw_t1_image = np.argmax(input_stack[:9], axis=0)
        rgb_image = input_stack[9:12].transpose(1, 2, 0)
        ndvi_image = input_stack[12]
        temp_image = input_stack[13]
        dw_t2_image = np.argmax(input_stack[14:23], axis=0)
        
        unnorm_targets = targets
        unnorm_outputs = outputs
        metadata = metadata_normalized
        temp_series = temp_series_normalized

    # --- Visualization ---
    fig = plt.figure(figsize=(20, 20))
    target_channel_names = [c.replace('after_', '') for c in CONFIG.dataset.target_channels]
    fig.suptitle(f"Trial {trial_id}, Step {step}, Targets: {', '.join(target_channel_names)}, Batch Loss: {batch_loss:.4f}", fontsize=16)

    # --- Metadata Display ---
    ax_meta = plt.subplot2grid((5, 4), (0, 0), colspan=1)
    ax_meta.text(0, 1, "Metadata", va='top', ha='left', fontsize=14, weight='bold')
    meta_keys = ['lat', 'lon', 'population', 'delta_time_years']
    if len(metadata) == len(meta_keys):
        meta_text = '\n'.join([f"{k}: {v:.4f}" for k, v in zip(meta_keys, metadata)])
    else:
        meta_text = "Metadata: " + ", ".join([f"{v:.4f}" for v in metadata])
    meta_text += f"\nT1 date: {t1_date_str}"
    meta_text += f"\nT2 date: {t2_date_str}"
    ax_meta.text(0, 0.8, meta_text, va='top', ha='left', fontsize=10)
    ax_meta.axis('off')

    # --- Input Images ---
    input_images = [dw_t1_image, rgb_image, ndvi_image, temp_image, dw_t2_image]
    titles_input = ['Input DW (t1)', 'Input RGB (t1)', 'Input NDVI (t1)', 'Input Temp (t1)', 'Input DW (t2)']
    
    for i, (img, title) in enumerate(zip(input_images, titles_input)):
        ax = plt.subplot2grid((5, 5), (1, i))
        if len(img.shape) == 2:
            if 'DW' in title:
                im = ax.imshow(img, cmap='tab10', vmin=0, vmax=8)
            else:
                im = ax.imshow(img, cmap='viridis')
            plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    # --- Target and Prediction Images ---
    num_targets = unnorm_targets.shape[0]
    for i in range(num_targets):
        # Ground Truth
        ax_gt = plt.subplot2grid((5, num_targets), (2, i))
        im = ax_gt.imshow(unnorm_targets[i], cmap='viridis')
        ax_gt.set_title(f"GT {target_channel_names[i]}", fontsize=10)
        ax_gt.axis('off')
        plt.colorbar(im, ax=ax_gt, orientation='vertical', fraction=0.05, pad=0.04)
        
        # Prediction
        min, max = min_max_values[i]
        ax_pred = plt.subplot2grid((5, num_targets), (3, i))
        im = ax_pred.imshow(unnorm_outputs[i], cmap='viridis')
        ax_pred.set_title(f"Pred {target_channel_names[i]}\n(min: {min:.2f}, max: {max:.2f})", fontsize=10)
        ax_pred.axis('off')
        plt.colorbar(im, ax=ax_pred, orientation='vertical', fraction=0.05, pad=0.04)

    # --- Temperature Series ---
    ax_temp_series = plt.subplot2grid((5, 4), (4, 0), colspan=4)
    ax_temp_series.plot(temp_series)
    ax_temp_series.set_title('Historical Temperature Series (un-normalized)')
    ax_temp_series.set_xlabel('Year index (from 1951)')
    ax_temp_series.set_ylabel('Temperature')
    ax_temp_series.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    img_filename = f"{study_name}_trial_{trial_id}_generated.png"
    img_path = os.path.join("reports/training/visualizations", img_filename)
    plt.savefig(img_path, dpi=200)
    plt.close(fig)
