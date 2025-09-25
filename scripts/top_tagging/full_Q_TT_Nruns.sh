#!/bin/bash

# Top Tagging Lorentz-CGGNN experiment (single config)
# Runs the same configuration multiple times for statistical significance
# Uses batch_size=32 and max_steps=50000; other params kept constant

# Check if number of runs is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <N> [dataroot]"
    echo "  N: Number of runs per experiment"
    echo "  dataroot: Path to data root (default: datasets)"
    echo ""
    echo "Example: $0 5 datasets"
    echo "         $0 10 /path/to/data"
    exit 1
fi

NUM_RUNS="$1"
DATAROOT="${2:-datasets}"

echo "Starting Top Tagging Lorentz-CGGNN experiment ($NUM_RUNS runs)..."
echo "Using dataroot: $DATAROOT"
echo ""

# Fixed configuration
DATASET="top_tagging"
MODEL="lorentz_cggnn"
BATCH_SIZE=32
MAX_STEPS=100000 #331126
NUM_WORKERS=16
SCHEDULER="cosine"
SUBSPACE_TYPE="Q"

echo "Configuration:"
echo "  dataset=$DATASET, model=$MODEL"
echo "  batch_size=$BATCH_SIZE, max_steps=$MAX_STEPS"
echo "  num_workers=$NUM_WORKERS, scheduler=$SCHEDULER, subspace_type=$SUBSPACE_TYPE"
echo ""

# Choose python interpreter: prefer `python`, fallback to `python3`
if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
else
    echo "Error: neither 'python' nor 'python3' found in PATH" >&2
    exit 1
fi

i=1
while [ $i -le $NUM_RUNS ]; do
    echo "  Run $i/$NUM_RUNS"
    $PYTHON_BIN train.py \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --dataroot "$DATAROOT" \
        --batch_size "$BATCH_SIZE" \
        --max_steps "$MAX_STEPS" \
        --num_workers "$NUM_WORKERS" \
        --scheduler "$SCHEDULER" \
        --subspace_type "$SUBSPACE_TYPE" \
        --seed $((42 + i)) \
	    --tt_num_train -1 \
	    --tt_num_val -1 \
	    --tt_num_test -1 \
        --resume \
        --val_n_interval 20 \
        --run_name full_long
    i=$((i + 1))
done

echo "Completed $NUM_RUNS runs"
echo "----------------------------------------"

echo "All Top Tagging Lorentz-CGGNN runs completed!"
echo "Results are saved in the lightning_logs directory"
echo "You can view the results using: tensorboard --logdir lightning_logs"
