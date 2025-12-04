from pathlib import Path

import os
from dotenv import load_dotenv
from loguru import logger
import sys

# Force line buffering for stderr to see logs in real-time, in HPC environments
sys.stderr.reconfigure(line_buffering=True)

from hydra import compose, initialize
from omegaconf import OmegaConf

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERMEDIATE_DATA_DIR = DATA_DIR / "intermediate"

DATASET_NAME = 'image_dataset_10m_perm'
DATASET_NAME_PROCESSED = f"{DATASET_NAME}_thresholded"
IMAGE_DATASET = RAW_DATA_DIR / DATASET_NAME
PROCESSED_IMAGE_DATASET = PROCESSED_DATA_DIR / DATASET_NAME_PROCESSED

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
    logger.add("log.log", colorize=False, mode='a')
except ModuleNotFoundError:
    pass


def load_config():
    with initialize(config_path="../conf", version_base=None):
        cfg = compose(config_name="config")
        OmegaConf.set_struct(cfg, False)
    cfg.PROJ_ROOT = PROJ_ROOT
    cfg.DATA_DIR = DATA_DIR
    cfg.PROCESSED_DATA_DIR = PROCESSED_DATA_DIR
    cfg.RAW_DATA_DIR = RAW_DATA_DIR
    cfg.IMAGE_DATASET = IMAGE_DATASET
    cfg.VISUALIZATION_DIR = PROJ_ROOT / "reports" / "visualizations"

    cfg.PROCESSED_IMAGE_DATASET = PROCESSED_IMAGE_DATASET
    cfg.INTERMEDIATE_DATA_DIR = INTERMEDIATE_DATA_DIR
    cfg.RAW_TEMPERATURE_DATA_DIR_CRU = RAW_DATA_DIR / "temperatures" / "yearly_CRU"
    cfg.RAW_TEMPERATURE_DATA_DIR_REGION = RAW_DATA_DIR / "temperatures" / "per_region"
    cfg.PROCESSED_TEMPERATURE_DATA_DIR = PROCESSED_DATA_DIR / "temperatures"

    cfg.GLOBAL_TEMPERATURE_INDEX_FILE = cfg.PROCESSED_TEMPERATURE_DATA_DIR

    cfg.SPLIT_INDICES_FILE = cfg.PROCESSED_DATA_DIR / "split_indices.npz"
    cfg.DATASET_METRICS_FILE = "repoorts/eda_thresholded/dataset_processed_metrics.csv"

    cfg.MODELS_DIR = PROJ_ROOT / "models"
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)

    cfg.DATASET_NAME = DATASET_NAME
    cfg.DATASET_NAME_PROCESSED = DATASET_NAME_PROCESSED

    return cfg


CONFIG = load_config()
