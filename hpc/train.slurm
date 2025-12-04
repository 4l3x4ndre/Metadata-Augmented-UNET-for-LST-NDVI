#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --time=0-50:00:00     # 0 days, hours minute limit
#SBATCH --nodes=1             #  compute nodes
#SBATCH --cpus-per-task=1     #  CPU cores
#SBATCH --mem=5G              # Gigabytes system memory (not gpu mem)
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name="TRAIN"
#SBATCH --output=logs/tmp_placeholder.log  # Temporary placeholder

TIMESTAMP=$(date +"%Y%m%d_%H%M%S") # Create a unique timestamp
LOGFILE="logs/train_${TIMESTAMP}_${SLURM_JOB_ID}.log"
exec > "$LOGFILE" 2>&1 # Redirect all future output to the unique log file

echo "Starting job with ID: $SLURM_JOB_ID"
echo "Logging to: $LOGFILE"
echo "Running with arguments: $@"


module load Python/3.12.3-GCCcore-13.3.0

poetry run python -m src.train --device gpu "$@" --jobid "$SLURM_JOB_ID"


echo "Job ended on $(date)"
echo "Elapsed time: $(( SECONDS / 3600 ))h $(( (SECONDS % 3600) / 60 ))m $(( SECONDS % 60 ))s"



