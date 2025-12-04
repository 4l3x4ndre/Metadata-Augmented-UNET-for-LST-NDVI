#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --time=0-50:00:00     # 0 days, hours minute limit
#SBATCH --nodes=1             #  compute nodes
#SBATCH --cpus-per-task=1     #  CPU cores
#SBATCH --mem=5G              # Gigabytes system memory (not gpu mem)
#SBATCH --gres=gpu:v100:1
#SBATCH --constraint=gpu32g
#SBATCH --job-name="TEST"
#SBATCH --output=logs/tmp_placeholder.log  

TIMESTAMP=$(date +"%Y%m%d_%H%M")
LOGFILE="logs/test_${TIMESTAMP}_${SLURM_JOB_ID}.log"
exec > "$LOGFILE" 2>&1 

echo "Starting job with ID: $SLURM_JOB_ID"
echo "Logging to: $LOGFILE"
echo "Running with arguments: $@"

module load Python/3.12.3-GCCcore-13.3.0

# poetry run python -m src.test --device gpu "$@"
poetry run python -m test.evaluate --device gpu "$@" --jobid "$SLURM_JOB_ID"

echo "Job ended on $(date)"
echo "Elapsed time: $(( SECONDS / 3600 ))h $(( (SECONDS % 3600) / 60 ))m $(( SECONDS % 60 ))s"

