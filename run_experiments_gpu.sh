#!/usr/bin/env bash
# GPU experiment launcher: runs all 32 experiments (8 losses × 2 datasets × 2 models)
# on the RTX 4090 (24GB). ResNet runs 16 jobs in parallel (16GB), ViT runs 8 at a time.
#
# Usage: nohup bash run_experiments_gpu.sh [epochs] [seed] > runs/experiments.log 2>&1 &
# Defaults: epochs=100, seed=0

set -e
cd /home/proj/lossfunction

PYTHON=/root/.conda/envs/torch/bin/python
EPOCHS=${1:-100}
SEED=${2:-0}
LOSS_LIST="ce focal poly lace_multi f_multi mba_ce mba_f mba_ps"
DATASET_LIST="cifar10 cifar100"
LOG_DIR=runs/logs
mkdir -p "$LOG_DIR"

# Clear pycache to ensure latest code
find src -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

echo "[$(date '+%H:%M:%S')] Starting GPU experiments: epochs=$EPOCHS seed=$SEED"
echo "  Python: $($PYTHON --version 2>&1)"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "  Losses: $LOSS_LIST"
echo "  Datasets: $DATASET_LIST"

run_jobs() {
    # Args: model_key dataset1 [dataset2] — launches 8 or 16 jobs in parallel
    local model_key=$1; shift
    local datasets=("$@")
    local pids=()
    echo "[$(date '+%H:%M:%S')] Launching ${#datasets[@]} dataset(s) × 8 losses = $((${#datasets[@]} * 8)) jobs for model=$model_key"
    for dataset in "${datasets[@]}"; do
        for loss in $LOSS_LIST; do
            local outdir="runs/${dataset}_${model_key}_${loss}_seed${SEED}"
            local logfile="$LOG_DIR/${dataset}_${model_key}_${loss}.log"
            rm -rf "$outdir"
            echo "  Launching: $dataset / $model_key / $loss"
            MBA_NUM_THREADS=4 OMP_NUM_THREADS=4 \
                $PYTHON src/train.py \
                    --config "src/configs/${dataset}_${model_key}.yaml" \
                    --override loss="$loss" seed="$SEED" epochs="$EPOCHS" \
                        output_dir="$outdir" num_workers=4 log_interval=200 \
                        save_checkpoint=false \
                    > "$logfile" 2>&1 &
            pids+=($!)
        done
    done
    echo "  Waiting for ${#pids[@]} jobs..."
    local fail=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "  [FAIL] pid $pid"
            fail=$((fail+1))
        fi
    done
    echo "[$(date '+%H:%M:%S')] Batch $model_key done. Failures: $fail"
    # Print quick results
    for dataset in "${datasets[@]}"; do
        for loss in $LOSS_LIST; do
            local h="runs/${dataset}_${model_key}_${loss}_seed${SEED}/history.json"
            if [ -f "$h" ]; then
                $PYTHON -c "import json; h=json.load(open('$h')); print(f'  ${dataset}_${model_key}_${loss}: top1={h[\"test_top1\"][-1]*100:.2f}% ece={h[\"test_ece\"][-1]:.4f} epochs={len(h[\"epoch\"])}')" 2>/dev/null || echo "  ${dataset}_${model_key}_${loss}: parse error"
            else
                echo "  ${dataset}_${model_key}_${loss}: FAILED"
            fi
        done
    done
}

# Batch 1: ResNet-56, both datasets in parallel (16 jobs, ~16GB GPU memory)
echo ""
echo "=========================================="
echo "=== Batch 1: ResNet-56 (16 jobs parallel) ==="
echo "=========================================="
run_jobs resnet56 cifar10 cifar100

# Batch 2: ViT, one dataset at a time (8 jobs, ~8-12GB GPU memory)
echo ""
echo "=========================================="
echo "=== Batch 2: ViT cifar10 (8 jobs parallel) ==="
echo "=========================================="
run_jobs vit cifar10

echo ""
echo "=========================================="
echo "=== Batch 3: ViT cifar100 (8 jobs parallel) ==="
echo "=========================================="
run_jobs vit cifar100

echo ""
echo "[$(date '+%H:%M:%S')] All GPU experiments complete."
echo "Results in runs/*_seed${SEED}/"
