from loguru import logger
import urban_planner.config

from src.data.retrieve_temperature import download_temperature
from src.data.process_temperature import process_temperature

def main():
    logger.info("Starting dataset processing...")
    download_temperature()
    process_temperature(
        urban_planner.config.CONFIG.RAW_TEMPERATURE_DATA_DIR_CRU,
        urban_planner.config.CONFIG.PROCESSED_TEMPERATURE_DATA_DIR
    )
    logger.info("Dataset processing completed.")


if __name__ == "__main__":
    main()
