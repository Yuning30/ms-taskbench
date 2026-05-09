#!/bin/bash
#SBATCH --job-name=pickfeas
#SBATCH --partition=unlimited
#SBATCH --array=0-49
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/common/users/shared/pracsys/ms-taskbench-data/datasets/pick_2dgrid_mplib_500k/slurm_logs/task_%a.out
#SBATCH --error=/common/users/shared/pracsys/ms-taskbench-data/datasets/pick_2dgrid_mplib_500k/slurm_logs/task_%a.err
#
# Pick-feasibility 500K data collection.
# Submit:
#   mkdir -p /common/users/shared/pracsys/ms-taskbench-data/datasets/pick_2dgrid_mplib_500k/slurm_logs
#   sbatch scripts/slurm_collect_array.sh
#
# 50 array tasks × 10,000 samples = 500,000 samples total.
# Outputs: /common/users/shared/pracsys/ms-taskbench-data/datasets/pick_2dgrid_mplib_500k/task_<id:04d>/

set -euo pipefail

REPO_ROOT="/common/home/st1122/Projects/ms-taskbench"
PROD_OUT="/common/users/shared/pracsys/ms-taskbench-data/datasets/pick_2dgrid_mplib_500k"

TASK_ID=$SLURM_ARRAY_TASK_ID
SEED=$((TASK_ID * 1000000))
TASK_OUT="$PROD_OUT/task_$(printf '%04d' "$TASK_ID")"

echo "[$(date -Iseconds)] node=$(hostname) task_id=$TASK_ID seed=$SEED out=$TASK_OUT"
echo "  ulimit -u: $(ulimit -u), cpus-per-task: ${SLURM_CPUS_PER_TASK:-?}"

cd "$REPO_ROOT"

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python -m taskbench.data.collect \
    --n 10000 \
    --grid-rows 3 --grid-cols 3 \
    --seed "$SEED" \
    --task-id "$TASK_ID" \
    --shard-size 500 \
    --out "$TASK_OUT" \
    --resume

echo "[$(date -Iseconds)] task_id=$TASK_ID complete"
