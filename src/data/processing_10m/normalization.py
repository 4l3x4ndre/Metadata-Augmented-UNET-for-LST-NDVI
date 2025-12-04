import numpy as np
from typing import List, Dict
from tqdm import tqdm
from loguru import logger
import json
import os

from src.data.processing_10m.utils import load_and_resize_image, load_and_resize_rgb
from src.data.process_temperature import TemperatureQuery

def calculate_normalization_metrics(train_samples: List[Dict], target_shape: tuple, config) -> Dict:
    """
    Calculates normalization metrics (mean, std) on the training set using a streaming approach.
    """
    logger.info("Calculating normalization metrics on the training set...")

    # Initializers for streaming mean and std calculation
    rgb_sum = np.zeros(3)
    rgb_sum_sq = np.zeros(3)
    
    temp_sum = 0.0
    temp_sum_sq = 0.0
    
    meta_sum = np.zeros(4)
    meta_sum_sq = np.zeros(4)
    
    temp_series_sum = 0.0
    temp_series_sum_sq = 0.0

    pixel_count = 0
    sample_count = 0
    temp_series_point_count = 0

    temp_query = TemperatureQuery(config.PROCESSED_TEMPERATURE_DATA_DIR)

    for sample in tqdm(train_samples, desc="Calculating metrics"):
        sample_count += 1
        
        # --- RGB ---
        rgb = load_and_resize_rgb(sample['files']['rgb'], target_shape) / 255.0
        rgb_sum += np.sum(rgb, axis=(1, 2))
        rgb_sum_sq += np.sum(rgb**2, axis=(1, 2))
        
        # --- Satellite Temperature ---
        temp = load_and_resize_image(sample['files']['temp'], target_shape)
        temp_sum += np.sum(temp)
        temp_sum_sq += np.sum(temp**2)
        
        pixel_count += temp.size

        # --- Metadata ---
        meta = np.array([sample['lat'], sample['lon'], sample['population'], sample['delta_time_years']])
        meta_sum += meta
        meta_sum_sq += meta**2
        
        # --- Temperature Series ---
        temp_series = temp_query.query(sample['lat'], sample['lon'], int(sample['t1_year']), int(sample['t1_month']))
        temp_series_sum += np.sum(temp_series)
        temp_series_sum_sq += np.sum(np.square(temp_series))
        temp_series_point_count += len(temp_series)

    # Calculate mean and std
    rgb_mean = rgb_sum / (pixel_count)
    rgb_std = np.sqrt(rgb_sum_sq / (pixel_count) - rgb_mean**2)

    temp_mean = temp_sum / pixel_count
    temp_std = np.sqrt(temp_sum_sq / pixel_count - temp_mean**2)

    meta_mean = meta_sum / sample_count
    meta_std = np.sqrt(meta_sum_sq / sample_count - meta_mean**2)
    
    temp_series_mean = temp_series_sum / temp_series_point_count
    temp_series_std = np.sqrt(temp_series_sum_sq / temp_series_point_count - temp_series_mean**2)

    metrics = {
        'rgb_mean': rgb_mean.tolist(),
        'rgb_std': rgb_std.tolist(),
        'temp_mean': temp_mean,
        'temp_std': temp_std,
        'meta_mean': meta_mean.tolist(),
        'meta_std': meta_std.tolist(),
        'temp_series_mean': temp_series_mean,
        'temp_series_std': temp_series_std,
    }
    
    # Save metrics to a JSON file
    metrics_path = os.path.join(config.PROCESSED_IMAGE_DATASET, 'normalization_metrics.json')
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Normalization metrics saved to {metrics_path}")

    return metrics

def one_hot_encode(img: np.ndarray, num_classes: int = 9) -> np.ndarray:
    """
    Converts a 2D semantic map to a one-hot encoded 3D array.
    """
    return np.eye(num_classes)[img.astype(int)].transpose(2, 0, 1)
