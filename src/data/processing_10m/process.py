import os
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
import rasterio
import json

import sys

# Force line buffering for stderr to see logs in real-time, in HPC environments
sys.stderr.reconfigure(line_buffering=True)

from urban_planner.config import CONFIG
from src.data.processing_10m.utils import group_files_by_location_and_time, load_and_resize_image, load_and_resize_rgb
from src.data.processing_10m.split import train_test_val_split
from src.data.processing_10m.normalization import one_hot_encode
from src.data.process_temperature import TemperatureQuery, process_temperature
from rasterio.warp import Resampling

# --- Thresholds for filtering samples with small changes ---
NDVI_CHANGE_THRESHOLD = 0.1
TEMP_CHANGE_THRESHOLD = 0.1
DW_CHANGE_THRESHOLD = 0.1  # Max proportion of pixels changed in any single class

def filter_subset(samples: list, target_shape: tuple, subset_name: str) -> list:
    """
    Filters a list of samples to remove those with negligible changes between t1 and t2.
    """
    kept_samples = []
    filtered_count = 0
    logger.info(f"Filtering {len(samples)} samples for subset: {subset_name}")

    for sample in tqdm(samples, desc=f"Filtering {subset_name}"):
        try:
            # --- Load data for FILTERING ---
            ndvi_t1 = load_and_resize_image(sample['files']['ndvi'], target_shape)
            temp_t1 = load_and_resize_image(sample['files']['temp'], target_shape)
            dw_t1 = load_and_resize_image(sample['files']['dw'], target_shape, resample_method=Resampling.nearest)
            ndvi_t2 = load_and_resize_image(sample['files']['ndvi_t2'], target_shape)
            temp_t2 = load_and_resize_image(sample['files']['temp_t2'], target_shape)
            dw_t2 = load_and_resize_image(sample['files']['dw_t2'], target_shape, resample_method=Resampling.nearest)

            # --- Perform filter check ---
            dw_t1_ohe = one_hot_encode(dw_t1)
            dw_t2_ohe = one_hot_encode(dw_t2)
            ndvi_diff = np.abs(ndvi_t2 - ndvi_t1).mean()
            temp_diff = np.abs(temp_t2 - temp_t1).mean()
            dw_channel_diffs = np.mean(np.abs(dw_t2_ohe - dw_t1_ohe), axis=(1, 2))
            dw_diff = np.max(dw_channel_diffs) if dw_channel_diffs.size > 0 else 0

            if ndvi_diff < NDVI_CHANGE_THRESHOLD and temp_diff < TEMP_CHANGE_THRESHOLD and dw_diff < DW_CHANGE_THRESHOLD:
                logger.info(f"Filtering out sample {sample['files']} due to negligible changes: NDVI diff={ndvi_diff}, Temp diff={temp_diff}, DW diff={dw_diff}")
                filtered_count += 1
                continue
            
            kept_samples.append(sample)

        except Exception as e:
            logger.error(f"Failed during filtering for sample {sample.get('city_name')}_{sample.get('city_id')}: {e}. Skipping.")
            continue

    logger.info(f"Filtered out {filtered_count} of {len(samples)} samples for {subset_name}.")
    return kept_samples

def filter_and_calculate_metrics(samples: list, target_shape: tuple):
    """
    Filters training samples and calculates normalization metrics in a single pass.
    """
    kept_samples = []
    # Running stats for normalization
    rgb_sum, rgb_sum_sq, rgb_pixel_count = np.zeros(3), np.zeros(3), 0
    temp_sum, temp_sum_sq, temp_pixel_count = 0, 0, 0
    meta_values = []
    temp_query = TemperatureQuery(CONFIG.PROCESSED_TEMPERATURE_DATA_DIR)   
    temp_series_sum = 0.0
    temp_series_sum_sq = 0.0
    temp_series_point_count = 0

    logger.info(f"Filtering and calculating metrics for {len(samples)} training samples...")
    for sample in tqdm(samples, desc="Filtering & Calculating Metrics for Train"):
        try:
            # --- Load data for FILTERING ---
            ndvi_t1 = load_and_resize_image(sample['files']['ndvi'], target_shape)
            temp_t1_filter = load_and_resize_image(sample['files']['temp'], target_shape)
            dw_t1 = load_and_resize_image(sample['files']['dw'], target_shape, resample_method=Resampling.nearest)
            ndvi_t2 = load_and_resize_image(sample['files']['ndvi_t2'], target_shape)
            temp_t2 = load_and_resize_image(sample['files']['temp_t2'], target_shape)
            dw_t2 = load_and_resize_image(sample['files']['dw_t2'], target_shape, resample_method=Resampling.nearest)

            # --- Perform filter check ---
            dw_t1_ohe = one_hot_encode(dw_t1)
            dw_t2_ohe = one_hot_encode(dw_t2)
            ndvi_diff = np.abs(ndvi_t2 - ndvi_t1).mean()
            temp_diff = np.abs(temp_t2 - temp_t1_filter).mean()
            dw_channel_diffs = np.mean(np.abs(dw_t2_ohe - dw_t1_ohe), axis=(1, 2))
            dw_diff = np.max(dw_channel_diffs) if dw_channel_diffs.size > 0 else 0

            if ndvi_diff < NDVI_CHANGE_THRESHOLD and temp_diff < TEMP_CHANGE_THRESHOLD and dw_diff < DW_CHANGE_THRESHOLD:
                logger.info(f"Filtering out sample {sample['files']} due to negligible changes: NDVI diff={ndvi_diff}, Temp diff={temp_diff}, DW diff={dw_diff}")
                print(f"Filtering out sample {sample['files']} due to negligible changes: NDVI diff={ndvi_diff}, Temp diff={temp_diff}, DW diff={dw_diff}")
                continue

            # --- If kept, load data for NORMALIZATION and update stats ---
            kept_samples.append(sample)

            rgb_t1 = load_and_resize_rgb(sample['files']['rgb'], target_shape) / 255.0
            rgb_sum += np.sum(rgb_t1, axis=(1, 2))
            rgb_sum_sq += np.sum(rgb_t1**2, axis=(1, 2))
            rgb_pixel_count += rgb_t1.shape[1] * rgb_t1.shape[2]

            temp_t1_norm = temp_t1_filter # Already loaded for filtering
            temp_sum += np.sum(temp_t1_norm)
            temp_sum_sq += np.sum(temp_t1_norm**2)
            temp_pixel_count += temp_t1_norm.size

            meta_values.append([sample['lat'], sample['lon'], sample['population'], sample['delta_time_years']])

            # --- Temperature Series ---
            temp_series = temp_query.query(sample['lat'], sample['lon'], int(sample['t1_year']), int(sample['t1_month']))
            temp_series_sum += np.sum(temp_series)
            temp_series_sum_sq += np.sum(np.square(temp_series))
            temp_series_point_count += len(temp_series)

        except Exception as e:
            logger.error(f"Failed during metric calculation for sample {sample.get('city_name')}_{sample.get('city_id')}: {e}. Skipping.")

    # --- Finalize metrics calculation ---
    logger.info("Finalizing normalization metrics...")
    rgb_mean = rgb_sum / rgb_pixel_count
    rgb_std = np.sqrt(rgb_sum_sq / rgb_pixel_count - rgb_mean**2)
    temp_mean = temp_sum / temp_pixel_count
    temp_std = np.sqrt(temp_sum_sq / temp_pixel_count - temp_mean**2)
    meta_mean = np.mean(meta_values, axis=0)
    meta_std = np.std(meta_values, axis=0)
    temp_series_mean = temp_series_sum / temp_series_point_count
    temp_series_std = np.sqrt(temp_series_sum_sq / temp_series_point_count - temp_series_mean**2)
    #
    #
    metrics = {
        'rgb_mean': rgb_mean.tolist(), 'rgb_std': rgb_std.tolist(),
        'temp_mean': temp_mean, 'temp_std': temp_std,
        'meta_mean': meta_mean.tolist(), 'meta_std': meta_std.tolist(),
        'temp_series_mean': temp_series_mean, 'temp_series_std': temp_series_std
    }

    logger.info(f"Kept {len(kept_samples)} out of {len(samples)} training samples.")
    return metrics, kept_samples

def process_and_save_subset(samples: list, metrics: dict, temp_query: TemperatureQuery, output_dir: str, target_shape: tuple):
    """
    Processes a pre-filtered subset of samples, normalizes, and saves them as .npz files.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Processing and saving {len(samples)} samples for subset: {os.path.basename(output_dir)}")

    for sample in tqdm(samples, desc=f"Processing & Saving {os.path.basename(output_dir)}"):
        output_filename = f"{sample['city_name']}_{sample['city_id']}_{sample['lat']:.4f}_{sample['lon']:.4f}_{sample['t1_year']}_{sample['t1_month']:02d}_to_{sample['t2_year']}_{sample['t2_month']:02d}.npz"
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_path):
            continue

        try:
            # --- Load all data for processing ---
            dw_t1 = load_and_resize_image(sample['files']['dw'], target_shape, resample_method=Resampling.nearest)
            rgb_t1 = load_and_resize_rgb(sample['files']['rgb'], target_shape)
            ndvi_t1 = load_and_resize_image(sample['files']['ndvi'], target_shape)
            temp_t1 = load_and_resize_image(sample['files']['temp'], target_shape)
            dw_t2 = load_and_resize_image(sample['files']['dw_t2'], target_shape, resample_method=Resampling.nearest)
            ndvi_t2 = load_and_resize_image(sample['files']['ndvi_t2'], target_shape)
            temp_t2 = load_and_resize_image(sample['files']['temp_t2'], target_shape)

            # --- Normalization ---
            rgb_t1 = (rgb_t1 / 255.0 - np.array(metrics['rgb_mean'])[:, np.newaxis, np.newaxis]) / np.array(metrics['rgb_std'])[:, np.newaxis, np.newaxis]
            dw_t1 = one_hot_encode(dw_t1)
            dw_t2 = one_hot_encode(dw_t2)
            temp_t1 = (temp_t1 - metrics['temp_mean']) / metrics['temp_std']
            temp_t2 = (temp_t2 - metrics['temp_mean']) / metrics['temp_std']
            
            # --- Stack and Save ---
            input_stack = np.vstack([dw_t1, rgb_t1, ndvi_t1[np.newaxis, :, :], temp_t1[np.newaxis, :, :], dw_t2])
            target_stack = np.stack([ndvi_t2, temp_t2], axis=0)
            meta = (np.array([sample['lat'], sample['lon'], sample['population'], sample['delta_time_years']]) - metrics['meta_mean']) / metrics['meta_std']
            temp_series = (np.array(temp_query.query(sample['lat'], sample['lon'], int(sample['t1_year']), int(sample['t1_month']))) - metrics['temp_series_mean']) / metrics['temp_series_std']

            np.savez_compressed(output_path, input=input_stack.astype(np.float32), target=target_stack.astype(np.float32), metadata=meta.astype(np.float32), temperature_serie=temp_series.astype(np.float32))

        except Exception as e:
            logger.error(f"Failed to process and save sample {sample['city_name']}_{sample['city_id']}: {e}")

def process_future_data():
    """
    Processes the downloaded satellite images to create a dataset for forecasting.
    """
    os.makedirs(CONFIG.IMAGE_DATASET, exist_ok=True)
    os.makedirs(CONFIG.PROCESSED_IMAGE_DATASET, exist_ok=True)
    
    logger.info("Processing historical temperature data...")
    process_temperature(CONFIG.RAW_TEMPERATURE_DATA_DIR_CRU, CONFIG.PROCESSED_TEMPERATURE_DATA_DIR)
    temp_query = TemperatureQuery(CONFIG.PROCESSED_TEMPERATURE_DATA_DIR)

    cities_df = pd.read_csv(os.path.join(CONFIG.PROCESSED_DATA_DIR, 'cities', 'worldcities_processed.csv'))
    city_population_map = cities_df.set_index('id')['population'].to_dict()

    logger.info("Generating sample list...")
    locations = group_files_by_location_and_time(CONFIG.IMAGE_DATASET)
    all_samples = []
    target_shape = None
    for location_key, location_data in locations.items():
        city_id, lat, lon = location_key
        sorted_timestamps = sorted(location_data['timestamps'].keys())
        for i in range(len(sorted_timestamps)):
            for j in range(i + 1, len(sorted_timestamps)):
                t1, t2 = sorted_timestamps[i], sorted_timestamps[j]
                if target_shape is None:
                    with rasterio.open(location_data['timestamps'][t1]['ndvi']) as ref_img:
                        target_shape = (ref_img.height, ref_img.width)
                required_files_t1 = ['dw', 'rgb', 'ndvi', 'temp']
                required_files_t2 = ['ndvi', 'temp', 'dw']
                if not (all(k in location_data['timestamps'][t1] for k in required_files_t1) and all(k in location_data['timestamps'][t2] for k in required_files_t2)):
                    continue
                all_samples.append({
                    'city_id': city_id, 'lat': lat, 'lon': lon, 'city_name': location_data['city_name'],
                    'population': city_population_map.get(city_id, 0), 't1_year': t1[0], 't1_month': t1[1],
                    't2_year': t2[0], 't2_month': t2[1], 'delta_time_years': (t2[0] - t1[0]) + (t2[1] - t1[1]) / 12.0,
                    'files': {**location_data['timestamps'][t1], 'ndvi_t2': location_data['timestamps'][t2]['ndvi'], 'temp_t2': location_data['timestamps'][t2]['temp'], 'dw_t2': location_data['timestamps'][t2]['dw']}
                })

    logger.info("Splitting data into train, validation, and test sets...")
    train_samples, val_samples, test_samples = train_test_val_split(all_samples)

    metrics_path = os.path.join(CONFIG.PROCESSED_IMAGE_DATASET, 'normalization_metrics.json')
    if os.path.exists(metrics_path):
        logger.info(f"Loading existing normalization metrics from {metrics_path}")
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        train_samples_filtered = filter_subset(train_samples, target_shape, "train")
    else:
        metrics, train_samples_filtered = filter_and_calculate_metrics(train_samples, target_shape)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Saved new normalization metrics to {metrics_path}")

    val_samples_filtered = filter_subset(val_samples, target_shape, "validation")
    test_samples_filtered = filter_subset(test_samples, target_shape, "test")

    process_and_save_subset(train_samples_filtered, metrics, temp_query, os.path.join(CONFIG.PROCESSED_IMAGE_DATASET, 'train'), target_shape)
    process_and_save_subset(val_samples_filtered, metrics, temp_query, os.path.join(CONFIG.PROCESSED_IMAGE_DATASET, 'val'), target_shape)
    process_and_save_subset(test_samples_filtered, metrics, temp_query, os.path.join(CONFIG.PROCESSED_IMAGE_DATASET, 'test'), target_shape)

    logger.success("Finished processing all data.")

if __name__ == '__main__':
    process_future_data()

