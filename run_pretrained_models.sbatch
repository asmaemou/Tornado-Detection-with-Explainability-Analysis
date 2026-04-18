#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH --job-name=pretrained_grid
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# Use the correct GPU directive for your cluster
#SBATCH --gres=gpu:1
# or, if your cluster uses this form instead:
# SBATCH --gpus=1

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tornado_aug

which python
python --version
nvidia-smi || true

python -u pretrained_models_new.py