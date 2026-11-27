#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --threads-per-core=1
#SBATCH --mem=256000

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100.80gb"

# 3 experiments, maximum 3 running at the same time
#SBATCH --array=0-2%3

#SBATCH --job-name=himol

# Separate log files for every array task
#SBATCH --output=himol_%A_%a.out
#SBATCH --error=himol_%A_%a.err


# ============================================================
# Shell settings
# ============================================================
set -x
set -euo pipefail
# ============================================================
# Define seeds
# ============================================================

SEEDS=(0 42 12345)

if [[ -z "${SEEDS[$SLURM_ARRAY_TASK_ID]}" ]]; then
    echo "ERROR: Invalid array task ID: $SLURM_ARRAY_TASK_ID"
    exit 1
fi

SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# set temp directory
export TMPDIR=/home/staff/a/akhedekar/atmp_dir/
export SSL_CERT_FILE=$(python -m certifi)

# Load Python environment
source /home/staff/a/akhedekar/python-chebai-graph/.venv/bin/activate

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access a CUDA GPU; stopping before training.")

print(f"PyTorch detected {torch.cuda.device_count()} CUDA GPU(s).")
PY


# Set current working directory
CHEBAI_DIR="/home/staff/a/akhedekar/HiMol/finetune"
cd $CHEBAI_DIR

RUN_NAME="himol_chebi_final_${SEED}_default_params"
# Tried with other dropout values too 
python finetune.py \
    --dataset=chebi \
    --input_model_file=/home/staff/a/akhedekar/HiMol/.saved_model/pretrain.pth \
    --split_file=/home/staff/a/akhedekar/python-chebai/data/chebi_v252/ChEBI50/processed/splits.csv \
    --num_workers=10 \
    --wandb_run_name="$RUN_NAME" \
    --epochs=200 \
    --runseed="$SEED"
    # ----- default parameters give the best macro-f1 performance than 
    # the corresponding parameters for ChEBI and others
    # --dropout_ratio=0 \
    # --batch_size=64 \
    # --lr_feat=0.002 \
    # --lr_pred=0.002 \