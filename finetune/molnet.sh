#!/usr/bin/env bash

set -Eeuo pipefail
trap 'status=$?; echo "ERROR: command failed with status ${status} at line ${BASH_LINENO[0]}: ${BASH_COMMAND}" >&2; exit "$status"' ERR

HIMOL_DIR="/home/aditya/HiMol"
FINETUNE_DIR="$HIMOL_DIR/finetune"
LOG_DIR="$FINETUNE_DIR/logs"
CHECKPOINT_DIR="$FINETUNE_DIR/model_checkpoints"
RESULTS_FILE="$LOG_DIR/himol_molnet_results.tsv"
RUN_LOG="$LOG_DIR/himol_molnet_$(date +%Y%m%d_%H%M%S).txt"

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"
exec > >(tee -a "$RUN_LOG") 2>&1
set -x

cd "$FINETUNE_DIR"
printf 'dataset\tseed\tmetric\tvalue\tcheckpoint\n' > "$RESULTS_FILE"

DATASETS=(bbbp bace sider clintox tox21 toxcast)
SEEDS=(0 42 12345)

for SEED in "${SEEDS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        case "$DATASET" in
            bbbp)
                LR_FEAT="0.0005"
                DROPOUT="0.5"
                METRICS=(test_auc test_f1)
                ;;
            bace)
                LR_FEAT="0.001"
                DROPOUT="0.5"
                METRICS=(test_auc test_f1)
                ;;
            sider)
                LR_FEAT="0.0005"
                DROPOUT="0.5"
                METRICS=(test_auc test_micro_f1 test_macro_f1)
                ;;
            clintox)
                LR_FEAT="0.001"
                DROPOUT="0.5"
                METRICS=(test_auc test_micro_f1 test_macro_f1)
                ;;
            tox21)
                LR_FEAT="0.0005"
                DROPOUT="0.7"
                METRICS=(test_auc test_micro_f1 test_macro_f1)
                ;;
            toxcast)
                LR_FEAT="0.001"
                DROPOUT="0.7"
                METRICS=(test_auc test_micro_f1 test_macro_f1)
                ;;
            *)
                echo "Unsupported dataset: $DATASET" >&2
                exit 1
                ;;
        esac

        RUN_NAME="${DATASET}_run${SEED}"
        RUN_OUTPUT="$LOG_DIR/${RUN_NAME}.txt"
        CHECKPOINT="$CHECKPOINT_DIR/${DATASET}_${SEED}.pth"

        echo "============================================================"
        echo "Starting HiMol run: $RUN_NAME"
        echo "Dataset: $DATASET"
        echo "Seed:    $SEED"
        echo "============================================================"

        python finetune.py \
            --dataset="$DATASET" \
            --input_model_file=/home/aditya/HiMol/.saved_model/pretrain.pth \
            --lr_feat="$LR_FEAT" \
            --lr_pred=0.001 \
            --dropout_ratio="$DROPOUT" \
            --runseed="$SEED" \
            --wandb_run_name="$RUN_NAME" \
            --wandb_mode=online \
            2>&1 | tee "$RUN_OUTPUT"

        if [[ ! -f "$CHECKPOINT_DIR/${DATASET}.pth" ]]; then
            echo "Expected checkpoint was not created: $CHECKPOINT_DIR/${DATASET}.pth" >&2
            exit 1
        fi
        cp "$CHECKPOINT_DIR/${DATASET}.pth" "$CHECKPOINT"

        python - "$RUN_OUTPUT" "$DATASET" "$SEED" "$CHECKPOINT" "${METRICS[@]}" >> "$RESULTS_FILE" <<'PY'
import re
import sys

log_path, dataset, seed, checkpoint, *metrics = sys.argv[1:]
text = open(log_path, encoding="utf-8").read()

patterns = {
    "test_auc": r"test_auc:\s*([0-9.eE+-]+)",
    "test_f1": r"test_f1:\s*([0-9.eE+-]+)",
    "test_micro_f1": r"test_micro_f1:\s*([0-9.eE+-]+)",
    "test_macro_f1": r"test_macro_f1:\s*([0-9.eE+-]+)",
}
for metric in metrics:
    matches = re.findall(patterns[metric], text)
    if not matches:
        raise SystemExit(f"Could not find {metric} in {log_path}")
    print(f"{dataset}\t{seed}\t{metric}\t{matches[-1]}\t{checkpoint}")
PY

        echo "Finished HiMol run: $RUN_NAME"
    done
done

python - "$RESULTS_FILE" <<'PY'
import csv
import math
import statistics
import sys
from collections import defaultdict

results = defaultdict(list)
with open(sys.argv[1], newline="", encoding="utf-8") as results_file:
    for row in csv.DictReader(results_file, delimiter="\t"):
        results[(row["dataset"], row["metric"])].append(float(row["value"]))

print("\nHiMol test performance")
print("dataset\tmetric\tmean\tsample_sd\tn")
for (dataset, metric), values in sorted(results.items()):
    mean = statistics.mean(values)
    sample_sd = statistics.stdev(values) if len(values) > 1 else math.nan
    print(f"{dataset}\t{metric}\t{mean:.6f}\t{sample_sd:.6f}\t{len(values)}")
PY

echo "Run log: $RUN_LOG"
echo "Per-seed results: $RESULTS_FILE"
