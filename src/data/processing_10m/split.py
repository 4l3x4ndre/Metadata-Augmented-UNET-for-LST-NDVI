import numpy as np
from typing import List, Dict, Tuple
import random
from loguru import logger

def train_test_val_split(samples: List[Dict], holdout_ratio: float = 0.01) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Splits samples into train, validation, and test sets based on year and a holdout set of cities.

    Args:
        samples (List[Dict]): A list of all samples, where each sample is a dictionary of metadata.
        holdout_ratio (float): The ratio of cities to hold out for the test set.

    Returns:
        Tuple[List[Dict], List[Dict], List[Dict]]: train, val, test samples.
    """
    # Get unique city IDs
    all_cities = list(set(s['city_id'] for s in samples))
    random.shuffle(all_cities)

    # Identify holdout cities
    holdout_city_count = int(len(all_cities) * holdout_ratio)
    holdout_cities = set(all_cities[:holdout_city_count])
    logger.info(f"Holding out {len(holdout_cities)} cities for the test set.")

    train_samples, val_samples, test_samples = [], [], []

    for sample in samples:
        # Samples from holdout cities go directly to the test set
        if sample['city_id'] in holdout_cities:
            test_samples.append(sample)
            continue

        # Split the rest based on the target year
        if sample['t2_year'] == 2025:
            test_samples.append(sample)
        elif sample['t2_year'] == 2024:
            val_samples.append(sample)
        elif sample['t2_year'] <= 2023:
            train_samples.append(sample)
            
    logger.info(f"Train samples: {len(train_samples)}")
    logger.info(f"Validation samples: {len(val_samples)}")
    logger.info(f"Test samples: {len(test_samples)}")

    return train_samples, val_samples, test_samples
