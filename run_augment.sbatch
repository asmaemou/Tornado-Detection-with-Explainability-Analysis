#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH --job-name=tornado_aug
#SBATCH --output=/homes/j244s673/Tornado-Detection-with-Explainability-Analysis/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/Tornado-Detection-with-Explainability-Analysis/logs/%x_%j.err
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

# Paths for your current train split
export INPUT_DIR="/homes/j244s673/documents/wsu/phd/tornado/train"
export OUTPUT_DIR="/homes/j244s673/documents/wsu/phd/tornado/train_augmented"

# Make sure output dir exists
mkdir -p "$OUTPUT_DIR"

# Run script from the submit directory
python augment_tornadoes.py