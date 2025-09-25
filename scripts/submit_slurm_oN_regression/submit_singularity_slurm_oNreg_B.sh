#!/usr/bin/env bash
#SBATCH --job-name=glgenn
#SBATCH --time=04:00:00                 # Максимальное время выполнения
#SBATCH --ntasks-per-node=2          # Количество MPI процессов на узел
#SBATCH --nodes=1                    # Требуемое кол-во узлов
#SBATCH --gpus=1                     # Требуемое кол-во GPU
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

set -x
singularity exec --nv $BIND_OPTS "$IMAGE" \
  bash $WORKDIR/scripts/on_regression_experiments/B_oN_regression_Nruns.sh "$N_RUNS" "$DATA_ROOT"

# singularity exec --nv $BIND_OPTS "$IMAGE" \
#   bash scripts/run_analysis.sh src/convex_hull.csv auto '' convex_hull

# singularity exec --nv $BIND_OPTS "$IMAGE" \
#   bash $WORKDIR/scripts/top_tagging/Q_TT_Nruns.sh "$N_RUNS" "$DATA_ROOT"