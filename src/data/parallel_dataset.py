import os
import multiprocessing
import numpy as np
from loguru import logger
from glob import glob
import pandas as pd

import src.data.gee_functions_future as gee
from urban_planner.config import CONFIG

def main():
    NUM_PROCESSES = 8
    logger.info(f"Starting parallel processing with {NUM_PROCESSES} workers.")

    # 1. Authenticate ONCE in the main process
    gee.authenticate()

    # 2. Load and prepare the city data
    coords_df = gee.load_cities(force=True)
    
    # Ensure the output directory exists
    output_dir = gee.CONFIG.IMAGE_DATASET
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory is set to: {output_dir}")

    # # Filter out any already processed cities.
    # # Only when there is any many image files as expected per city.
    # processed_files = glob(os.path.join(output_dir, "*.tif"))
    # processed_city_ids = {}
    # for file in processed_files:
    #     basename = os.path.basename(file)
    #     filename_start = "_".join(basename.split('_')[:-2]).split('_')[-1]
    #     if filename_start in processed_city_ids:
    #         processed_city_ids[filename_start] += 1
    #     else:
    #         processed_city_ids[filename_start] = 1
    # cities_to_process = []
    # nb_cities_already_processed = 0
    # for _, row in coords_df.iterrows():
    #     city_id = str(row['id'])
    #     if city_id in processed_city_ids and processed_city_ids[city_id] == CONFIG.dataset.nb_images_per_cities:
    #         logger.info(f"Skipping already processed city ID: {city_id}")
    #         nb_cities_already_processed += 1
    #     else:
    #         cities_to_process.append(row)
    # coords_df = pd.DataFrame(cities_to_process, columns=coords_df.columns)
    # logger.info(f"Total cities to process: {len(coords_df)} (skipped {nb_cities_already_processed} already processed)")
    # print(f"Total cities to process: {len(coords_df)} (skipped {nb_cities_already_processed} already processed)")

    # 3. Split the DataFrame into chunks for each process
    # This ensures no process works on the same data
    df_chunks = np.array_split(coords_df, NUM_PROCESSES)
    
    # Prepare arguments for each worker. We pass a chunk ID for better logging.
    tasks = [(i, chunk, output_dir) for i, chunk in enumerate(df_chunks)]

    # 4. Create and run the multiprocessing pool
    logger.info("Distributing tasks to worker pool...")
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        # map will send one item from 'tasks' to each call of 'process_city_chunk'
        pool.map(gee.process_city_chunk, tasks)

    logger.success("All processing chunks have been completed!")


if __name__ == "__main__":
    main()
