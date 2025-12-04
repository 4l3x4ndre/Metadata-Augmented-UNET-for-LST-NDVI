import os
import re
from typing import Dict, List, Tuple
import numpy as np
import rasterio
from rasterio.warp import Resampling
from loguru import logger

def parse_filename(filename: str) -> Dict:
    """Parses a filename to extract metadata."""
    try:
        parts = os.path.basename(filename).split('_')
        city_name = "_".join(parts[:-8])
        city_id = int(parts[-8])
        lat = float(parts[-7])
        lon = float(parts[-6])
        offset_x = float(parts[-5])
        offset_y = float(parts[-4])
        year = int(parts[-3])
        month = int(parts[-2])
        img_type = parts[-1].split('.')[0]

        return {
            'city_name': city_name,
            'city_id': city_id,
            'lat': lat,
            'lon': lon,
            'offset_x': offset_x,
            'offset_y': offset_y,
            'year': year,
            'month': month,
            'type': img_type,
            'filepath': filename
        }
    except (IndexError, ValueError) as e:
        logger.warning(f"Could not parse filename: {filename}. Error: {e}")
        return None

def group_files_by_location_and_time(image_dir: str) -> Dict:
    """Groups files by location and then by time."""
    locations = {}
    for filename in os.listdir(image_dir):
        if not filename.endswith('.tif'):
            continue
        
        metadata = parse_filename(filename)
        if metadata is None:
            continue

        location_key = (metadata['city_id'], metadata['lat'], metadata['lon'])
        if location_key not in locations:
            locations[location_key] = {
                'lat': metadata['lat'],
                'lon': metadata['lon'],
                'city_id': metadata['city_id'],
                'city_name': metadata['city_name'],
                'timestamps': {}
            }

        timestamp_key = (metadata['year'], metadata['month'])
        if timestamp_key not in locations[location_key]['timestamps']:
            locations[location_key]['timestamps'][timestamp_key] = {}
        
        locations[location_key]['timestamps'][timestamp_key][metadata['type']] = os.path.join(image_dir, filename)

    return locations

def load_and_resize_image(image_path: str, target_shape: Tuple[int, int], resample_method=Resampling.bilinear) -> np.ndarray:
    """Loads a single band image and resizes it to the target shape."""
    with rasterio.open(image_path) as src:
        data = src.read(
            1,
            out_shape=target_shape,
            resampling=resample_method
        )
    return data

def load_and_resize_rgb(image_path: str, target_shape: Tuple[int, int], resample_method=Resampling.bilinear) -> np.ndarray:
    """Loads a 3-band RGB image and resizes it."""
    with rasterio.open(image_path) as src:
        # Read all 3 bands
        data = src.read(
            [1, 2, 3],
            out_shape=(3, target_shape[0], target_shape[1]),
            resampling=resample_method
        )
    return data


def acquire_temp_min_max(dataset_dir:str, eda_dir:str) -> Dict[str, float]:
    """
    Load json, if temp min/max exists load them,
    other with get eda values and compute min/max.
    """
    normalized_data_json = os.path.join(dataset_dir, 'normalization_metrics.json')
    import json
    if not os.path.exists(normalized_data_json):
        raise FileNotFoundError(f"{normalized_data_json} does not exist.")

    with open(normalized_data_json, 'r') as f:
        normalization_metrics = json.load(f)
    if 'target_temp_t2_min' in normalization_metrics:
        temps =  {
            'target_temp_t2_min': normalization_metrics['target_temp_t2_min'],
            'target_temp_t2_max': normalization_metrics['target_temp_t2_max'],
            'input_temp_t1_min': normalization_metrics['input_temp_t1_min'],
            'input_temp_t1_max': normalization_metrics['input_temp_t1_max']
        }
        logger.info(f"Loaded temp min/max from {normalized_data_json}: {temps}")
        return temps

    logger.info(f"Temp min/max not found in {normalized_data_json}, computing from EDA data.")
    eda_file = os.path.join(eda_dir, 'dataset_processed_metrics.csv')
    import pandas as pd 
    df = pd.read_csv(eda_file)
    normalization_metrics['target_temp_t2_min'] = df['target_temp_t2_min'].min()
    normalization_metrics['target_temp_t2_max'] = df['target_temp_t2_max'].max()
    normalization_metrics['input_temp_t1_min'] = df['input_temp_t1_min'].min()
    normalization_metrics['input_temp_t1_min'] = df['input_temp_t1_max'].max()
    with open(normalized_data_json, 'w') as f:
        json.dump(normalization_metrics, f, indent=4)
    temps = {
        'target_temp_t2_min': normalization_metrics['target_temp_t2_min'],
        'target_temp_t2_max': normalization_metrics['target_temp_t2_max'],
        'input_temp_t1_min': normalization_metrics['input_temp_t1_min'],
        'input_temp_t1_max': normalization_metrics['input_temp_t1_max']
    }
    logger.info(f"Computed and saved temp min/max to {normalized_data_json}: {temps}")
    return temps
