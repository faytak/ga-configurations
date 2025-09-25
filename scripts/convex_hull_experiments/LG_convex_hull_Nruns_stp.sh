#!/bin/bash

# Convex Hull GLGMLP experiments with varying parameters
# Each experiment varies num_samples, max_steps, and batch_size according to the table
# while keeping all other parameters the same
# Each experiment is run N times for statistical significance

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

echo "Starting Convex Hull GLGMLP experiments with varying parameters ($NUM_RUNS runs each)..."
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
            --dataset convex_hull \
            --model convex_hull_glgmlp \
            --dataroot "$DATAROOT" \
            --n 6 \
            --num_samples "$num_samples" \
            --max_steps "$max_steps" \
            --batch_size "$batch_size" \
            --hidden_features 32 \
            --num_layers 4 \
            --subspace_type "LG" \
            --num_workers 4 \
            --prefetch_factor 2 \
            --seed $((42 + i)) \
            --run_name "stItt"
        i=$((i + 1))
    done
    
    echo "Completed $exp_name ($num_runs runs)"
    echo "----------------------------------------"
}

# Run each experiment N times based on the parameter table:
# num_samples  max_steps  batch_size
# 256          512        128
# 1024         1024       128  
# 4096         2048       128
# 16384        1024       256

# $PYTHON_BIN data/generate_hull.py

# run_experiment "Experiment 1: 256 samples, 512 steps, batch_size 128" 256 512 128 "$NUM_RUNS"
# run_experiment "Experiment 2: 1024 samples, 1024 steps, batch_size 128" 1024 1024 128 "$NUM_RUNS"
# run_experiment "Experiment 3: 4096 samples, 2048 steps, batch_size 128" 4096 2048 128 "$NUM_RUNS"
# run_experiment "Experiment 4: 16384 samples, 1024 steps, batch_size 256" 16384 1024 256 "$NUM_RUNS"

# run_experiment "Experiment 3: 4096 samples, 2048 steps, batch_size 128" 4096 16 128 "$NUM_RUNS"
run_experiment "Experiment 3: 4096 samples, 2048 steps, batch_size 128" 4096 32 128 "$NUM_RUNS"
# run_experiment "Experiment 3: 4096 samples, 2048 steps, batch_size 128" 4096 64 128 "$NUM_RUNS"
run_experiment "Experiment 3: 4096 samples, 2048 steps, batch_size 128" 4096 128 128 "$NUM_RUNS"
# run_experiment "Experiment 3: 4096 samples, 2048 steps, batch_size 128" 4096 256 128 "$NUM_RUNS"
run_experiment "Experiment 3: 4096 samples, 2048 steps, batch_size 128" 4096 512 128 "$NUM_RUNS"
# run_experiment "Experiment 3: 4096 samples, 2048 steps, batch_size 128" 4096 1024 128 "$NUM_RUNS"
run_experiment "Experiment 3: 4096 samples, 2048 steps, batch_size 128" 4096 2048 128 "$NUM_RUNS"

echo "All Convex Hull GLGMLP experiments completed!"
echo "Results are saved in the lightning_logs directory"
echo "You can view the results using: tensorboard --logdir lightning_logs"