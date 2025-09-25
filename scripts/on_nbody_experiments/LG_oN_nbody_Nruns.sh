#!/bin/bash

# ON N-Body GLGENN-GNN experiments with varying parameters
# Matches the interface of Q_oN_regression_Nruns.sh: Usage: $0 <N> [dataroot]

# Check if number of runs is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <N> [dataroot]"
    echo "  N: Number of runs per experiment"
    echo "  dataroot: Path to data root (required for nbody; default: datasets)"
    echo ""
    echo "Example: $0 3"
    echo "         $0 5 /absolute/path/to/datasets"
    exit 1
fi

NUM_RUNS="$1"
DATAROOT="${2:-datasets}"

echo "Starting ON N-Body GLGENN-GNN experiments with varying parameters ($NUM_RUNS runs each)..."
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

# Model hyperparameters (aligned with experiments/nbody.csv and train.py defaults)
NB_HIDDEN=${NB_HIDDEN:-32}
NB_LAYERS=${NB_LAYERS:-3}

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
            --dataset nbody \
            --model nbody_glgenn_gnn \
            --dataroot "$DATAROOT" \
            --num_samples "$num_samples" \
            --batch_size "$batch_size" \
            --max_steps "$max_steps" \
            --num_workers 8 \
            --prefetch_factor 4 \
            --subspace_type "LG" \
            --nb_hidden_features "$NB_HIDDEN" \
            --nb_n_layers "$NB_LAYERS" \
            --seed $((42 + i))
        i=$((i + 1))
    done

    echo "Completed $exp_name ($num_runs runs)"
    echo "----------------------------------------"
}

# Parameter table derived from experiments/nbody.csv
# Feel free to toggle experiments by commenting/uncommenting lines below.
# run_experiment "Experiment 1: 50 samples, 1024 steps, batch_size 16" 50 1024 16 "$NUM_RUNS"
# run_experiment "Experiment 2: 500 samples, 4096 steps, batch_size 64" 500 4096 64 8
run_experiment "Experiment 3: 5000 samples, 4096 steps, batch_size 128" 5000 4096 128 7

echo "All ON N-Body GLGENN-GNN experiments completed!"
echo "Results are saved in the lightning_logs directory"
echo "You can view the results using: tensorboard --logdir lightning_logs"


