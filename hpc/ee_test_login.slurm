#!/bin/sh
#SBATCH --partition=short
#SBATCH --time=0-00:20:00     # days, hours minute limit
#SBATCH --nodes=1             # compute nodes
#SBATCH --mem=5G              # Gigabytes system memory (not gpu mem)
#SBATCH --job-name="EE_login"
#SBATCH --output=logs/ee_test_login.log    # Log file

echo "Starting job with ID: $SLURM_JOB_ID"

module load Python/3.12.3-GCCcore-13.3.0

poetry run python -m src.utils.ee_test_login
