#!/bin/bash

# Script to run analysis on ON Q5 experiments
# This script will try both TensorBoard and CSV analysis methods

echo "Analyzing experiment results..."
echo "==============================="

# Check if lightning_logs directory exists
if [ ! -d "lightning_logs" ]; then
    echo "Error: lightning_logs directory not found!"
    echo "Please run this script from the project root directory."
    exit 1
fi

# Choose python interpreter: prefer `python`, fallback to `python3`
if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
else
    echo "Error: neither 'python' nor 'python3' found in PATH" >&2
    exit 1
fi

OUTPUT_PATH=${1:-src/experiments_summary.csv}
SOURCE=${2:-auto} # auto | tb | csv
INCLUDE_REGEX=${3:-}
DATASET_FILTER=${4:-}
MODEL_FILTER=${5:-}

CMD="$PYTHON_BIN scripts/analyze_experiments.py --log_dir lightning_logs --output ${OUTPUT_PATH} --source ${SOURCE}"

if [ -n "$INCLUDE_REGEX" ]; then
  CMD="$CMD --include_regex $INCLUDE_REGEX"
fi
if [ -n "$DATASET_FILTER" ]; then
  CMD="$CMD --dataset $DATASET_FILTER"
fi
if [ -n "$MODEL_FILTER" ]; then
  CMD="$CMD --model $MODEL_FILTER"
fi

echo "Running: $CMD"
eval $CMD
STATUS=$?

if [ $STATUS -ne 0 ]; then
  echo "Analysis failed. Please verify logs exist under lightning_logs/."
  exit $STATUS
fi

echo ""
echo "Analysis complete! Check the output CSV file for detailed results."
