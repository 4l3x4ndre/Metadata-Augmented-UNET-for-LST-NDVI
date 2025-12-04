import numpy as np
import rasterio
from rasterio.enums import Resampling
from PIL import Image
import torch
import json
import os
from scipy.interpolate import interp1d
import streamlit as st
from src.data.process_temperature import TemperatureQuery

from urban_planner.config import CONFIG

# Hardcoded metrics from previous step
METRICS = {
    "rgb_mean": [0.5045, 0.4785, 0.4885],
    "rgb_std": [0.2355, 0.1755, 0.1391],
    "temp_mean": 32.1837,
    "temp_std": 13.3625,
    "meta_mean": [19.9373, 11.3007, 1379817.47, 2.2468],
    "meta_std": [23.0396, 71.8749, 5424837.30, 1.5172],
    "temp_series_mean": 0.1135,
    "temp_series_std": 1.0049
}

@st.cache_resource
def get_temperature_query():
    try:
        return TemperatureQuery(CONFIG.PROCESSED_TEMPERATURE_DATA_DIR)
    except Exception as e:
        print(f"Could not initialize TemperatureQuery: {e}")
        return None

# Dynamic World Palette (Hex -> Class ID)
DW_PALETTE = {
    "#419bdf": 0, # Water
    "#397d49": 1, # Trees
    "#88b053": 2, # Grass
    "#7a87c6": 3, # Flooded vegetation
    "#e49635": 4, # Crops
    "#dfc35a": 5, # Shrub and scrub
    "#c4281b": 6, # Built
    "#a59b8f": 7, # Bare
    "#b39fe1": 8, # Snow and ice
}

DW_PALETTE_INV = {v: k for k, v in DW_PALETTE.items()}

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def get_palette_rgb():
    return {k: hex_to_rgb(v) for k, v in DW_PALETTE_INV.items()}

def load_and_resize(path, target_shape, resampling=Resampling.bilinear):
    with rasterio.open(path) as src:
        data = src.read(
            out_shape=(src.count, target_shape[0], target_shape[1]),
            resampling=resampling
        )
        return data

def one_hot_encode(img_array, num_classes=9):
    # img_array: (1, H, W) or (H, W) with integer class labels
    if img_array.ndim == 3:
        img_array = img_array[0]
    return np.eye(num_classes)[img_array.astype(int)].transpose(2, 0, 1)

def canvas_to_dw_map(canvas_rgba, target_shape, original_map=None):
    """
    Converts the canvas RGBA numpy array back to a DW class map (H, W).
    Uses nearest neighbor color matching.
    If original_map is provided, pixels with 0 alpha in canvas will retain original_map class.
    original_map should be (H, W) or (1, H, W).
    """
    # Resize canvas to target shape
    img = Image.fromarray(canvas_rgba.astype('uint8'))
    img = img.resize((target_shape[1], target_shape[0]), Image.NEAREST)
    arr = np.array(img) # (H, W, 4)
    
    # Extract Alpha to find where user drew
    alpha = arr[:, :, 3]
    mask_drawn = alpha > 0 # Boolean mask where user drew
    
    # Extract RGB
    arr_rgb = arr[:, :, :3]
    
    # Map colors to classes
    # Flatten image
    pixels = arr_rgb.reshape(-1, 3)
    
    # Centers
    palette_rgb = get_palette_rgb()
    centers = np.array([palette_rgb[i] for i in range(9)]) # (9, 3)
    
    from scipy.spatial.distance import cdist
    dists = cdist(pixels, centers)
    nearest_class = np.argmin(dists, axis=1) # (H*W,)
    nearest_class = nearest_class.reshape(target_shape) # (H, W)
    
    if original_map is not None:
        # Squeeze original map if needed
        if original_map.ndim == 3:
            original_map = original_map[0]
        
        output_map = np.where(mask_drawn, nearest_class, original_map)
        return output_map.astype(np.uint8)
    
    return nearest_class.astype(np.uint8)

def prepare_input(paths, canvas_data, lat, lon, population, year_t1, month_t1, year_t2, month_t2):
    """
    Prepares the input tensor for the model.
    paths: dict with 'dw', 'rgb', 'ndvi', 'temp'
    canvas_data: numpy array from streamlit canvas
    """
    
    # Reference shape from DW
    with rasterio.open(paths['dw']) as src:
        # target_shape = (src.height, src.width)
        target_shape = (CONFIG.model.img_size, CONFIG.model.img_size)
        
    # 1. Load and Resize t1 data
    dw_t1 = load_and_resize(paths['dw'], target_shape, Resampling.nearest)
    rgb_t1 = load_and_resize(paths['rgb'], target_shape, Resampling.bilinear)
    ndvi_t1 = load_and_resize(paths['ndvi'], target_shape, Resampling.bilinear)
    temp_t1 = load_and_resize(paths['temp'], target_shape, Resampling.bilinear)
    
    # 2. Process t2 (Canvas)
    # Pass dw_t1 (original map) to preserve areas not drawn on
    dw_t2_map = canvas_to_dw_map(canvas_data, target_shape, original_map=dw_t1)
    
    # 3. Normalization
    # RGB: (X / 255 - mean) / std
    rgb_t1_norm = (rgb_t1 / 255.0 - np.array(METRICS['rgb_mean'])[:, None, None]) / np.array(METRICS['rgb_std'])[:, None, None]
    
    # Temp: (X - mean) / std
    temp_t1_norm = (temp_t1 - METRICS['temp_mean']) / METRICS['temp_std']
    
    # One Hot Encode DW
    dw_t1_ohe = one_hot_encode(dw_t1)
    dw_t2_ohe = one_hot_encode(dw_t2_map)
    
    # 4. Stacking
    # Stack: [DW_t1 (9), RGB_t1 (3), NDVI_t1 (1), Temp_t1 (1), DW_t2 (9)]
    
    input_stack = np.vstack([dw_t1_ohe, rgb_t1_norm, ndvi_t1, temp_t1_norm, dw_t2_ohe])
    input_tensor = torch.from_numpy(input_stack).float().unsqueeze(0) # Add batch dim
    
    # 5. Metadata
    delta_time = (year_t2 - year_t1) + (month_t2 - month_t1) / 12.0
    meta_raw = np.array([lat, lon, population, delta_time])
    meta_norm = (meta_raw - np.array(METRICS['meta_mean'])) / np.array(METRICS['meta_std'])
    
    t1_date = np.array([year_t1, month_t1])
    t2_date = np.array([year_t2, month_t2])
    
    meta_full = np.concatenate([meta_norm, t1_date, t2_date])
    meta_tensor = torch.from_numpy(meta_full).float().unsqueeze(0)
    
    # 6. Temperature Series
    temp_query = get_temperature_query()
    if temp_query is not None:
        try:
            ts = temp_query.query(lat, lon, int(year_t1), int(month_t1))
            ts_norm = (np.array(ts) - METRICS['temp_series_mean']) / METRICS['temp_series_std']
            temp_series_tensor = torch.from_numpy(ts_norm).float().unsqueeze(0) # (1, seq_len)
        except Exception as e:
            print(f"Error querying temperature series: {e}")
            # Fallback to dummy
            temp_series_tensor = torch.zeros((1, 60)).float()
    else:
        # Dummy temp series for now (normalized 0 = mean)
        temp_series_tensor = torch.zeros((1, 60)).float() # Length?
    
    return input_tensor, meta_tensor, temp_series_tensor

def denormalize_output(ndvi_norm, temp_norm):
    temp_denorm = (temp_norm * METRICS['temp_std']) + METRICS['temp_mean']
    return ndvi_norm, temp_denorm
