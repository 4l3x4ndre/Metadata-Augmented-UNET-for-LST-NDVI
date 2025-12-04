from PIL import Image
from loguru import logger
import streamlit as st
import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from urban_planner.config import CONFIG
from src.model import UrbanPredictor
from src.dataset import FuturePredictionDataset, collate_fn
from src.utils.visualization import dw_to_rgb

# --- Configuration & Caching ---

@st.cache_resource
def load_model(checkpoint_path):
    """Loads the model from a checkpoint path."""
    if not os.path.exists(checkpoint_path):
        st.error(f"Checkpoint not found at: {checkpoint_path}")
        return None, None, None, None

    try:
        checkpoint = torch.load(checkpoint_path, map_location=CONFIG.device)
        
        hyperparams = checkpoint['hyperparameters']
        model_type = checkpoint.get('model_type', 'unet') # Default to unet if not specified
        metadata_input_length = checkpoint.get('metadata_input_length', 4)
        
        # Resolve embedding flags (Backwards compatibility)
        if 'temporal_embeddings' in hyperparams:
            temporal_embeddings = hyperparams['temporal_embeddings']
            metadata_embeddings = hyperparams['metadata_embeddings']
        else:
            # Legacy fallback
            default_emb = True
            checkpoint_study = checkpoint.get('study_name', '')
            if 'noemb' in checkpoint_study:
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

        #debug keys:
        logger.debug(f"Checkpoint path: {checkpoint_path}")
        logger.debug(f"Checkpoint keys: {list(checkpoint.keys())}")
        # debug metadatainputlength
        logger.debug(f"Metadata input length from checkpoint: {metadata_input_length}")
        logger.debug(f"Resolved embeddings - temporal: {temporal_embeddings}, metadata: {metadata_embeddings}")

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
        
        st.success(f"Loaded model from trial {checkpoint.get('trial_id', 'N/A')} from study '{checkpoint.get('study_name', 'N/A')}'")
        return model, checkpoint, temporal_embeddings, metadata_embeddings
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

@st.cache_resource
def load_dataset():
    """Loads the test dataset."""
    try:
        test_dataset = FuturePredictionDataset(split='test')
        return test_dataset
    except FileNotFoundError as e:
        st.error(f"Test dataset not found. Please ensure the data is processed. Details: {e}")
        return None

@st.cache_data
def load_normalization_metrics():
    """Loads the normalization metrics from the processed dataset folder."""
    metrics_path = os.path.join(CONFIG.PROCESSED_IMAGE_DATASET, 'normalization_metrics.json')
    if not os.path.exists(metrics_path):
        st.warning(f"Normalization metrics not found at {metrics_path}. Visualizations will be normalized.")
        return None
    with open(metrics_path, 'r') as f:
        st.success(f"Loaded normalization metrics from {metrics_path}")
        return json.load(f)

# --- Visualization Helpers ---
def plot_zoomed_views(gt_img, pred_img, title_prefix):
    """Creates a figure with 4 zoomed-in quadrants for GT and Pred."""
    h, w = gt_img.shape
    quadrants = {
        "Top-Left": (0, h // 2, 0, w // 2),
        "Top-Right": (0, h // 2, w // 2, w),
        "Bottom-Left": (h // 2, h, 0, w // 2),
        "Bottom-Right": (h // 2, h, w // 2, w),
    }
    
    fig, axes = plt.subplots(4, 2, figsize=(6, 12))
    fig.suptitle(f"Zoomed Quadrants for {title_prefix}", fontsize=14)

    for i, (name, (y1, y2, x1, x2)) in enumerate(quadrants.items()):
        # Ground Truth
        ax_gt = axes[(i%2)*2, (i//2)]
        im_gt = ax_gt.imshow(gt_img[y1:y2, x1:x2], cmap='viridis')
        ax_gt.set_title(f"GT {name}")
        ax_gt.axis('off')
        plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

        # Prediction
        ax_pred = axes[(i%2)*2+1, (i//2)]
        im_pred = ax_pred.imshow(pred_img[y1:y2, x1:x2], cmap='viridis')
        ax_pred.set_title(f"Pred {name}")
        ax_pred.axis('off')
        plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def plot_zoomed_comparison(gt_img, pred_imgs, pred_names, title_prefix):
    """Creates a figure with 4 zoomed-in quadrants for GT and multiple predictions."""
    h, w = gt_img.shape
    quadrants = {
        "Top-Left": (0, h // 2, 0, w // 2),
        "Top-Right": (0, h // 2, w // 2, w),
        "Bottom-Left": (h // 2, h, 0, w // 2),
        "Bottom-Right": (h // 2, h, w // 2, w),
    }
    
    num_preds = len(pred_imgs)
    
    fig, axes = plt.subplots(4, 1 + num_preds, figsize=(3 * (1 + num_preds), 12))
    fig.suptitle(f"Zoomed Quadrants for {title_prefix}", fontsize=14)

    for i, (name, (y1, y2, x1, x2)) in enumerate(quadrants.items()):
        # Ground Truth
        ax_gt = axes[i, 0]
        im_gt = ax_gt.imshow(gt_img[y1:y2, x1:x2], cmap='viridis')
        ax_gt.set_title(f"GT {name}")
        ax_gt.axis('off')
        plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

        # Predictions
        for j in range(num_preds):
            ax_pred = axes[i, j + 1]
            im_pred = ax_pred.imshow(pred_imgs[j][y1:y2, x1:x2], cmap='viridis')
            ax_pred.set_title(f"{pred_names[j][:10]}... {name}") # Shorten name
            ax_pred.axis('off')
            plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
            
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def plot_zoomed_views_with_error(gt_img, pred_img, title_prefix):
    """Creates a figure with 4 zoomed-in quadrants for GT, Pred, and Error."""
    h, w = gt_img.shape
    quadrants = {
        "Top-Left": (0, h // 2, 0, w // 2),
        "Top-Right": (0, h // 2, w // 2, w),
        "Bottom-Left": (h // 2, h, 0, w // 2),
        "Bottom-Right": (h // 2, h, w // 2, w),
    }
    
    fig, axes = plt.subplots(4, 3, figsize=(9, 12))
    fig.suptitle(f"Zoomed Quadrants for {title_prefix}", fontsize=14)

    for i, (name, (y1, y2, x1, x2)) in enumerate(quadrants.items()):
        gt_quad = gt_img[y1:y2, x1:x2]
        pred_quad = pred_img[y1:y2, x1:x2]
        error_quad = pred_quad - gt_quad
        err_max_abs = np.max(np.abs(error_quad)) if np.max(np.abs(error_quad)) > 0 else 1
        vmin = np.min([gt_quad.min(), pred_quad.min()])
        vmax = np.max([gt_quad.max(), pred_quad.max()])

        # Ground Truth
        ax_gt = axes[i, 0]
        im_gt = ax_gt.imshow(gt_quad, cmap='viridis', vmin=vmin, vmax=vmax)
        ax_gt.set_title(f"GT {name}")
        ax_gt.axis('off')
        plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

        # Prediction
        ax_pred = axes[i, 1]
        im_pred = ax_pred.imshow(pred_quad, cmap='viridis', vmin=vmin, vmax=vmax)
        ax_pred.set_title(f"Pred {name}")
        ax_pred.axis('off')
        plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

        # Error
        ax_err = axes[i, 2]
        im_err = ax_err.imshow(error_quad, cmap='coolwarm', vmin=-err_max_abs, vmax=err_max_abs)
        ax_err.set_title(f"Error {name}")
        ax_err.axis('off')
        plt.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


    

def get_unnormalized_data(input_stack, metadata, temp_series, targets, outputs, metrics):
    """Un-normalizes all data for visualization if metrics are available."""
    if not metrics:
        return input_stack, metadata, temp_series, targets, outputs

    # --- Inputs ---
    rgb_normalized = input_stack[9:12]
    rgb_image = (rgb_normalized * np.array(metrics['rgb_std'])[:, np.newaxis, np.newaxis] + np.array(metrics['rgb_mean'])[:, np.newaxis, np.newaxis]) * 255.0
    rgb_image = np.clip(rgb_image.transpose(1, 2, 0), 0, 255).astype(np.uint8)
    ndvi_image = input_stack[12]
    temp_normalized = input_stack[13]
    temp_image = temp_normalized * metrics['temp_std'] + metrics['temp_mean']
    
    dw_t1_image = dw_to_rgb(
        np.argmax(
            np.stack(
                [input_stack[i]*i for i in range(9)] # 0 for the first one is fine as argmax
            ), 
            axis=0
        ) 
    )
    dw_t2_image = dw_to_rgb(
        np.argmax(
            np.stack(
                [input_stack[i+14]*i for i in range(9)] # 0 for the first one is fine as argmax
            ), 
            axis=0
        ) 
    )


    unnorm_inputs = {
        'DW (t1)': dw_t1_image, 'RGB (t1)': rgb_image, 'NDVI (t1)': ndvi_image,
        'Temp (t1)': temp_image, 'DW (t2)': dw_t2_image
    }

    # --- Targets & Outputs ---
    unnorm_targets = np.zeros_like(targets)
    unnorm_outputs = np.zeros_like(outputs)
    target_channel_names = CONFIG.dataset.target_channels
    for i, ch in enumerate(target_channel_names):
        if 'temp' in ch.lower():
            unnorm_targets[i] = targets[i] * metrics['temp_std'] + metrics['temp_mean']
            unnorm_outputs[i] = outputs[i] * metrics['temp_std'] + metrics['temp_mean']
        else: 
            unnorm_targets[i] = targets[i]
            unnorm_outputs[i] = outputs[i]

    # --- Metadata & Temp Series ---
    unnorm_meta = metadata
    unnorm_meta[:4] = metadata[:4] * metrics['meta_std'] + metrics['meta_mean']
    unnorm_temp_series = temp_series * metrics['temp_series_std'] + metrics['temp_series_mean']
    
    return unnorm_inputs, unnorm_meta, unnorm_temp_series, unnorm_targets, unnorm_outputs
