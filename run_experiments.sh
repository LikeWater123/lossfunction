#!/usr/bin/env bash
# Launch all 32 experiment combinations in 2 batches (16 jobs each, 8 threads per job = 128 threads total).
# Each batch runs all 8 losses × 2 datasets in parallel for one model.
# Usage: ./run_experiments.sh [epochs] [seed]
# Defaults: epochs=30, seed=0

set -e
cd /home/proj/lossfunction

EPOCHS=${1:-30}
SEED=${2:-0}
LOSS_LIST="ce focal poly lace_multi f_multi mba_ce mba_f mba_ps"
DATASET_LIST="cifar10 cifar100"
LOG_DIR=runs/logs
mkdir -p "$LOG_DIR"

echo "[$(date +%H:%M:%S)] Starting experiments: epochs=$EPOCHS seed=$SEED"
echo "  Losses: $LOSS_LIST"
echo "  Datasets: $DATASET_LIST"

run_batch() {
    local model=$1
    local config=$2
    local pids=()
    echo
    echo "[$(date +%H:%M:%S)] === Batch: model=$model (config=$config) ==="
    for dataset in $DATASET_LIST; do
        for loss in $LOSS_LIST; do
            local outdir="runs/${dataset}_${model}_${loss}_seed${SEED}"
            local logfile="$LOG_DIR/${dataset}_${model}_${loss}.log"
            echo "  Launching: $dataset / $model / $loss -> $logfile"
            MBA_NUM_THREADS=8 OMP_NUM_THREADS=8 \
                python3 src/train.py \
                    --config "src/configs/${dataset}_${model}.yaml" \
                    --override loss="$loss" seed="$SEED" epochs="$EPOCHS" \
                        output_dir="$outdir" num_workers=2 log_interval=200 \
                        save_checkpoint=false \
                > "$logfile" 2>&1 &
            pids+=($!)
        done
    done
    echo "  Waiting for ${#pids[@]} jobs to finish..."
    local fail=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "  [FAIL] pid $pid"
            fail=$((fail+1))
        fi
    done
    echo "[$(date +%H:%M:%S)] Batch $model done. Failures: $fail"
}

run_batch resnet56 cifar10_resnet56.yaml
run_batch vit cifar10_vit.yaml

echo
echo "[$(date +%H:%M:%S)] All experiments complete."
echo "Results in runs/*_seed${SEED}/"
