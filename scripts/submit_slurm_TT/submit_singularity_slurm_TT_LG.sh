#!/usr/bin/env bash
#SBATCH --job-name=glgenn
#SBATCH --time=06:00:00                 # Максимальное время выполнения
#SBATCH --nodes=1                    # Требуемое кол-во узлов
#SBATCH --gpus=4                     # Требуемое кол-во GPU
#SBATCH --cpus-per-task=8            # Требуемое кол-во CPU 
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module load singularity/3.9.0 || true

IMAGE=${IMAGE:-$SLURM_SUBMIT_DIR/glgenn.sif}
WORKDIR=${WORKDIR:-/workspace}

mkdir -p logs

if [[ ! -f "$IMAGE" ]]; then
  echo "Missing container image: $IMAGE" >&2
  echo "Build it first via scripts/build_singularity.sh" >&2
  exit 1
fi

cd "$SLURM_SUBMIT_DIR"

# Bind repo and datasets if present
BIND_OPTS="--bind $SLURM_SUBMIT_DIR:$WORKDIR"
if [[ -d $SLURM_SUBMIT_DIR/datasets ]]; then
  BIND_OPTS="$BIND_OPTS --bind $SLURM_SUBMIT_DIR/datasets:$WORKDIR/datasets"
fi

echo "GPUs requested: $SLURM_GPUS on node $(hostname)"

# Run convex hull experiment launcher inside the container.
# Args: N_RUNS [DATAROOT]
#   N_RUNS   - number of runs per experiment (default: 5)
#   DATAROOT - path to datasets (default: $WORKDIR/datasets)
N_RUNS=${1:-1}
DATA_ROOT=${2:-$WORKDIR/datasets}

# Ensure a stable tmp inside the bound workspace to avoid cleanup races on /tmp
HOST_JOB_TMP_DIR="$SLURM_SUBMIT_DIR/tmp/${SLURM_JOB_ID:-$$}"
CONTAINER_JOB_TMP_DIR="$WORKDIR/tmp/${SLURM_JOB_ID:-$$}"
mkdir -p "$HOST_JOB_TMP_DIR"
export SINGULARITYENV_TMPDIR="$CONTAINER_JOB_TMP_DIR"
export SINGULARITYENV_TMP="$CONTAINER_JOB_TMP_DIR"
export SINGULARITYENV_TEMP="$CONTAINER_JOB_TMP_DIR"
# bind host tmp dir into the container's tmp location
BIND_OPTS="$BIND_OPTS --bind $HOST_JOB_TMP_DIR:$CONTAINER_JOB_TMP_DIR"

set -x
# singularity exec --nv $BIND_OPTS "$IMAGE" \
#   bash $WORKDIR/scripts/convex_hull_experiments/A_convex_hull_Nruns.sh "$N_RUNS" "$DATA_ROOT"

# singularity exec --nv $BIND_OPTS "$IMAGE" \
#   bash scripts/run_analysis.sh src/convex_hull.csv auto '' convex_hull

singularity exec --nv $BIND_OPTS "$IMAGE" \
  bash $WORKDIR/scripts/top_tagging/LG_TT_Nruns.sh "$N_RUNS" "$DATA_ROOT"


