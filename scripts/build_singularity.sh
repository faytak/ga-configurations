#!/usr/bin/env bash
set -euo pipefail

# Build GLGENN Singularity image using Sylabs Remote Builder
# Requires: `module load singularity/3.9.0` on HSE HPC and `singularity remote login` (token)

IMAGE_NAME="glgenn.sif"
DEF_FILE="Singularity.def"

if ! command -v singularity >/dev/null 2>&1; then
  echo "Singularity not found. On HPC, run: module load singularity/3.9.0" >&2
  exit 1
fi

if [[ ! -f "$DEF_FILE" ]]; then
  echo "Missing $DEF_FILE at repo root." >&2
  exit 1
fi

echo "Building $IMAGE_NAME from $DEF_FILE using remote builder..."
singularity build --remote "$IMAGE_NAME" "$DEF_FILE"

echo "Built $IMAGE_NAME"

