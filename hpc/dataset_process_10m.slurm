#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --time=0-50:00:00     # 0 days, 6 hours 00 minute limit
#SBATCH --nodes=1             #  compute nodes
#SBATCH --cpus-per-task=1     #  CPU cores
#SBATCH --mem=7G              # Gigabytes system memory (not gpu mem)
#SBATCH --job-name="PROC_DB"
#SBATCH --output=logs/tmp_placeholder.log  # Temporary placeholder

TIMESTAMP=$(date +"%Y%m%d_%H%M") # Create a unique timestamp
LOGFILE="logs/dataset_process_10m_${TIMESTAMP}.log"
exec > "$LOGFILE" 2>&1 

echo "Starting job with ID: $SLURM_JOB_ID"
echo "Logging to: $LOGFILE"

module load Python/3.12.3-GCCcore-13.3.0

poetry run python -m src.data.processing_10m.process

echo "Job ended on $(date)"
echo "Elapsed time: $(( SECONDS / 3600 ))h $(( (SECONDS % 3600) / 60 ))m $(( SECONDS % 60 ))s"
