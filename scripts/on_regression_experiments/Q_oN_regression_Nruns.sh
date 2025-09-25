#!/bin/bash

# ON Regression GLGMLP experiments with varying parameters
# Each experiment varies num_samples, max_steps, and batch_size according to the table
# while keeping all other parameters the same
# This script matches the interface of Q_convex_hull_Nruns.sh: Usage: $0 <N> [dataroot]

# Check if number of runs is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <N> [dataroot]"
    echo "  N: Number of runs per experiment"
    echo "  dataroot: Path to data root (unused for on_regression; kept for compatibility)"
    echo ""
    echo "Example: $0 5"
    echo "         $0 10 datasets"
    exit 1
fi

NUM_RUNS="$1"
DATAROOT="${2:-datasets}"

echo "Starting ON Regression GLGMLP experiments with varying parameters ($NUM_RUNS runs each)..."
echo "Using dataroot: $DATAROOT (unused for on_regression)"
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

# Function to run an experiment multiple times
run_experiment() {
    local exp_name="$1"
    local num_samples="$2"
    local max_steps="$3"
    local batch_size="$4"
    local num_runs="$5"
    
    echo "Starting $exp_name ($num_runs runs)..."
    
    i=1
    while [ $i -le $num_runs ]; do
        echo "  Run $i/$num_runs of $exp_name"
        $PYTHON_BIN train.py \
            --dataset on_regression \
            --model on_glg \
            --n 8 \
            --on_output_qtgp 8 \
            --on_hidden_mlp_1 380 \
            --on_hidden_mlp_2 380 \
            --on_use_mlp \
            --num_samples "$num_samples" \
            --batch_size "$batch_size" \
            --max_steps "$max_steps" \
            --num_workers 8 \
            --prefetch_factor 4 \
            --subspace_type "Q" \
            --seed $((42 + i))
        i=$((i + 1))
    done
    
    echo "Completed $exp_name ($num_runs runs)"
    echo "----------------------------------------"
}

# Run each experiment N times based on the parameter table used previously
# run_experiment "Experiment 1: 50 samples, 1024 steps, batch_size 16" 50 1024 16 "$NUM_RUNS"
# run_experiment "Experiment 2: 500 samples, 1024 steps, batch_size 32" 500 1024 32 "$NUM_RUNS"
run_experiment "Experiment 3: 5000 samples, 1024 steps, batch_size 32" 5000 1024 32 "$NUM_RUNS"
# run_experiment "Experiment 4: 30000 samples, 3001 steps, batch_size 15" 30000 3001 15 "$NUM_RUNS"

echo "All ON Regression GLGMLP experiments completed!"
echo "Results are saved in the lightning_logs directory"
echo "You can view the results using: tensorboard --logdir lightning_logs"