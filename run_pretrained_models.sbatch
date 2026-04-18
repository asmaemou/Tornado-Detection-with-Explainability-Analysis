#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH --job-name=pretrained_py
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Uncomment one if your cluster needs a GPU request
# #SBATCH --gres=gpu:1
# #SBATCH --gpus=1

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tornado_aug

which python
python --version

python pretrained_models_new.py