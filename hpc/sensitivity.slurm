#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --time=0-02:00:00     # Short time limit for analysis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --gres=gpu:v100:1          # Generic GPU request
#SBATCH --job-name="SENSITIVITY"
#SBATCH --output=logs/sensitivity_%j.log

TIMESTAMP=$(date +"%Y%m%d_%H%M")
LOGFILE="logs/sensitivity_${TIMESTAMP}_${SLURM_JOB_ID}.log"
exec > "$LOGFILE" 2>&1 

echo "Starting sensitivity analysis job with ID: $SLURM_JOB_ID"
echo "Logging to: $LOGFILE"

module load Python/3.12.3-GCCcore-13.3.0

# Default to checking the latest best model if not provided (example usage)
# poetry run python -m test.metadata_sensitivity --device gpu "$@"

poetry run python -m test.metadata_sensitivity --device gpu --jobid $SLURM_JOB_ID "$@"

echo "Job ended on $(date)"
