"""
Data set and loaders for training.
"""
import os
import numpy as np
import pandas as pd
import torch
from typing import Tuple
from loguru import logger
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random

from urban_planner.config import CONFIG



class FuturePredictionDataset(Dataset):
    def __init__(self, split: str, transform=None):
        """
        Dataset for future prediction based on processed .npz files from process_10m.

        Args:
            split (str): One of 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.processed_dir = CONFIG.PROCESSED_IMAGE_DATASET
        self.split = split
        self.transform = transform
        self.data_dir = os.path.join(self.processed_dir, self.split)

        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Directory for split '{self.split}' not found at: {self.data_dir}")

        self.file_list = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        self.file_list.sort()
        
        logger.info(f"Found {len(self.file_list)} samples in {self.data_dir}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        
        # Parse dates from filename
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        t1_year = int(parts[-5])
        t1_month = int(parts[-4])
        t2_year = int(parts[-2])
        t2_month = int(parts[-1].split('.')[0])

        data = np.load(filepath)
        
        input_data = data['input']
        target_data = data['target']
        metadata = data['metadata']
        temp_series = data['temperature_serie']

        if self.transform:
            input_data, target_data = self.transform(input_data, target_data)

        input_tensor = torch.from_numpy(input_data).float()
        target_tensor = torch.from_numpy(target_data).float()
        metadata_tensor = torch.from_numpy(metadata).float()
        temp_series_tensor = torch.from_numpy(temp_series).float()
        
        t1_date_tensor = torch.tensor([t1_year, t1_month]).float()
        t2_date_tensor = torch.tensor([t2_year, t2_month]).float()

        return input_tensor, metadata_tensor, temp_series_tensor, t1_date_tensor, t2_date_tensor, target_tensor

    def get_metadata_from_idx(self, idx: int) -> dict:
        """Extracts metadata (city, lat, lon) from the filename at a given index."""
        filepath = self.file_list[idx]
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        city = " ".join(parts[:-8])
        lat = float(parts[-7])
        lon = float(parts[-6])
        return {"city": city, "lat": lat, "lon": lon}
        

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch): 
    device = CONFIG.device
    # Filter out None entries
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])

    inputs, metadatas, temp_series, t1_dates, t2_dates, targets = zip(*batch)
    
    # Get original lengths of temperature series
    temp_series_lengths = torch.tensor([len(ts) for ts in temp_series])

    inputs = torch.stack(inputs).float().to(device)
    metadatas = torch.stack(metadatas).float().to(device)
    targets = torch.stack(targets).float().to(device)
    t1_dates = torch.stack(t1_dates).float().to(device)
    t2_dates = torch.stack(t2_dates).float().to(device)
    
    # Pad temperature series
    temp_series_padded = pad_sequence(temp_series, batch_first=True, padding_value=0.0).float().to(device)

    return inputs, metadatas, temp_series_padded, temp_series_lengths, t1_dates, t2_dates, targets

def create_dataloader(
        split:str,
        batch_size:int,
        shuffle:bool,
        dataset_type:str,
        transform=None,
        num_workers:int=0
    ):
    assert dataset_type == 'future', "Only 'future' dataset_type is supported in create_dataloader."

    dataset = FuturePredictionDataset(
        split=split,
        transform=transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return dataloader


class RandomFlip:
    def __init__(self):
        random.seed(CONFIG.seed)
    def __call__(self, x, y):
        if random.random() < 0.5:
            x = np.flip(x, axis=2).copy()  # horizontal flip
            y = np.flip(y, axis=2).copy()
        return x, y
