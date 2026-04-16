#!/bin/bash
#SBATCH --job-name=tornado_aug
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH --output=/homes/j244s673/documents/wsu/phd/tornado_data/tornado_aug_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/tornado_data/tornado_aug_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

# Activate conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tornado_aug

# Paths
export INPUT_DIR="/homes/j244s673/documents/wsu/phd/tornado_data/tornado"
export OUTPUT_DIR="/homes/j244s673/documents/wsu/phd/tornado_data/augmented_tornadoes"

# Go to working directory
cd /homes/j244s673/documents/wsu/phd/tornado_data

# Run script
python augment_tornadoes.py