#!/usr/bin/env bash
set -euo pipefail

# Run training inside Singularity with GPU(s) and PyTorch Lightning DDP.
# Usage examples:
#   scripts/run_singularity.sh --dataset top_tagging --model lorentz_cggnn --dataroot /path/to/datasets --batch_size 8
#   scripts/run_singularity.sh --dataset on_regression --model on_glg --n 5 --num_samples 30000 --batch_size 32

IMAGE=${IMAGE:-glgenn.sif}
WORKDIR=${WORKDIR:-/workspace}

if [[ ! -f "$IMAGE" ]]; then
  echo "Container image $IMAGE not found. Build with scripts/build_singularity.sh" >&2
  exit 1
fi

# Bind current repo into the container at /workspace, and optionally datasets dir if provided.
BIND_OPTS="--bind $(pwd):$WORKDIR"

# If a datasets directory exists locally, bind it to /workspace/datasets
if [[ -d datasets ]]; then
  BIND_OPTS="$BIND_OPTS --bind $(pwd)/datasets:$WORKDIR/datasets"
fi

set -x
singularity exec --nv $BIND_OPTS "$IMAGE" \
  python3 $WORKDIR/train.py "$@"

