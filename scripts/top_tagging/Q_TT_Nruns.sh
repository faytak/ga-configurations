#!/bin/bash

# Top Tagging Lorentz-CGGNN experiments with varying parameters
# Matches the multi-experiment style of A_oN_nbody_Nruns.sh
# Usage: $0 <N> [dataroot]

# Check if number of runs is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <N> [dataroot]"
    echo "  N: Number of runs per experiment"
    echo "  dataroot: Path to data root (default: datasets)"
    echo ""
    echo "Example: $0 3"
    echo "         $0 5 /absolute/path/to/datasets"
    exit 1
fi

NUM_RUNS="$1"
DATAROOT="${2:-datasets}"

echo "Starting Top Tagging Lorentz-CGGNN experiments with varying parameters ($NUM_RUNS runs each)..."
echo "Using dataroot: $DATAROOT"
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

# Fixed configuration
DATASET="top_tagging"
MODEL="lorentz_cggnn"
NUM_WORKERS=16
SCHEDULER="cosine"
SUBSPACE_TYPE="Q"

# Function to run an experiment multiple times
run_experiment() {
    local exp_name="$1"
    local tt_num_train="$2"
    local max_steps="$3"
    local batch_size="$4"
    local num_runs="$5"

    echo "Starting $exp_name ($num_runs runs)..."

    i=1
    while [ $i -le $num_runs ]; do
        echo "  Run $i/$num_runs of $exp_name"
        $PYTHON_BIN train.py \
            --dataset "$DATASET" \
            --model "$MODEL" \
            --dataroot "$DATAROOT" \
            --batch_size "$batch_size" \
            --max_steps "$max_steps" \
            --num_workers "$NUM_WORKERS" \
            --scheduler "$SCHEDULER" \
            --subspace_type "$SUBSPACE_TYPE" \
            --tt_num_train "$tt_num_train" \
            --tt_num_val 1024 \
            --tt_num_test 1024 \
            --val_n_interval 30 \
            --seed $((42 + i))
        i=$((i + 1))
    done

    echo "Completed $exp_name ($num_runs runs)"
    echo "----------------------------------------"
}

# Parameter table (toggle experiments by commenting/uncommenting lines)
# Format: run_experiment "Name" <tt_num_train> <max_steps> <batch_size> "$NUM_RUNS"

# Example small subsets
# run_experiment "TT_Q: 128 train, 1000 steps, bs32" 128 1000 32 "$NUM_RUNS"
run_experiment "TT_Q: 256 train, 1000 steps, bs32" 256 1000 32 10
# run_experiment "TT_Q: 512 train, 1000 steps, bs32" 512 1000 32 "$NUM_RUNS"

# Large / long-ish
# run_experiment "TT_Q: 1024 train, 3000 steps, bs32" 1024 3000 32 "$NUM_RUNS"

echo "All Top Tagging Lorentz-CGGNN experiments completed!"
echo "Results are saved in the lightning_logs directory"
echo "You can view the results using: tensorboard --logdir lightning_logs"