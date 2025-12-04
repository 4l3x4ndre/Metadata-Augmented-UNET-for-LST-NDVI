#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --time=0-02:00:00     # 2 hours limit
#SBATCH --nodes=1             # compute nodes
#SBATCH --cpus-per-task=4     # CPU cores (matching num_workers in script)
#SBATCH --mem=32G             
#SBATCH --job-name="GT_SENS"
#SBATCH --output=logs/tmp_placeholder.log  # Temporary placeholder

TIMESTAMP=$(date +"%Y%m%d_%H%M")
LOGFILE="logs/ground_truth_sensitivity_${TIMESTAMP}_${SLURM_JOB_ID}.log"
exec > "$LOGFILE" 2>&1

echo "Starting job with ID: $SLURM_JOB_ID"
echo "Logging to: $LOGFILE"

module load Python/3.12.3-GCCcore-13.3.0

# Run the generation script
poetry run python test/generate_ground_truth_sensitivity.py

echo "Job ended on $(date)"
echo "Elapsed time: $(( SECONDS / 3600 ))h $(( (SECONDS % 3600) / 60 ))m $(( SECONDS % 60 ))s"
