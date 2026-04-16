#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH --job-name=tornado_aug
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

# Run from the directory where sbatch was submitted
cd "$SLURM_SUBMIT_DIR"

# Activate conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tornado_aug

# Paths
export INPUT_DIR="/homes/j244s673/documents/wsu/phd/tornado_data/tornado"
export OUTPUT_DIR="/homes/j244s673/documents/wsu/phd/tornado_data/augmented_tornadoes"

# Make sure logs dir exists
mkdir -p logs

# Run script from the submit directory
python augment_tornadoes.py