#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH --job-name=fix_ann
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tornado_aug

which python
python --version

python fix_annotations_hpc.py