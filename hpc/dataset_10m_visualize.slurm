#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --time=0-10:00:00     # 0 days, 6 hours 00 minute limit
#SBATCH --nodes=1             #  compute nodes
#SBATCH --cpus-per-task=1     #  CPU cores
#SBATCH --mem=10G              # Gigabytes system memory (not gpu mem)
#SBATCH --job-name="VIZ_DB"
#SBATCH --output=logs/tmp_placeholder.log  # Temporary placeholder

TIMESTAMP=$(date +"%Y%m%d_%H%M") # Create a unique timestamp
LOGFILE="logs/dataset_10m_viz_${TIMESTAMP}_${SLURM_JOB_ID}.log" 
exec > "$LOGFILE" 2>&1 

echo "Starting job with ID: $SLURM_JOB_ID"
echo "Logging to: $LOGFILE"
echo "Running with arguments: $@"

module load Python/3.12.3-GCCcore-13.3.0

poetry run python -m src.utils.visualize_npz extract data/processed/image_dataset_10m_perm_thresholded/ reports/eda_thresholded/dataset_processed_metrics.csv

echo "Job ended on $(date)"
echo "Elapsed time: $(( SECONDS / 3600 ))h $(( (SECONDS % 3600) / 60 ))m $(( SECONDS % 60 ))s"
