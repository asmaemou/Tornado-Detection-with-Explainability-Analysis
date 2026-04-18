#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH --job-name=pretrained_nb
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Uncomment one of these if your cluster supports GPU requests
# #SBATCH --gres=gpu:1
# #SBATCH --gpus=1

set -euo pipefail

# Run from the directory where sbatch was submitted
cd "$SLURM_SUBMIT_DIR"

# Activate conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tornado_aug

# Notebook names
NOTEBOOK="pretrained_models_new.ipynb"
OUTPUT_NOTEBOOK="pretrained_models_new_executed.ipynb"

# Execute notebook
jupyter nbconvert \
  --to notebook \
  --execute "$NOTEBOOK" \
  --output "$OUTPUT_NOTEBOOK" \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=python3