#!/bin/bash
# Launch a pool of taskbench.data.collect processes in parallel.
#
# Usage: run_collect_pool.sh N_PROCS SAMPLES_PER_PROC OUT_DIR [BASE_SEED] [GRID_ROWS] [GRID_COLS] [SHARD_SIZE] [STAGGER_S]
#
# Each task i writes shards under OUT_DIR/task_<i:04d>/ and logs to OUT_DIR/task_<i:04d>.log.
# Seeds are deterministic: BASE_SEED + i * 1000000 (large stride to keep streams disjoint).
# Sets OMP_NUM_THREADS=1 to avoid BLAS thread oversubscription with many parallel processes.
# STAGGER_S sleeps between background launches to avoid simultaneous fork/thread spikes
# (each sapien init spawns ~24 threads; on systems with low ulimit -u, peaks can fail).

set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 N_PROCS SAMPLES_PER_PROC OUT_DIR [BASE_SEED] [GRID_ROWS] [GRID_COLS] [SHARD_SIZE] [STAGGER_S]" >&2
  exit 2
fi

N=$1
M=$2
OUT=$3
BASE_SEED=${4:-0}
GR=${5:-3}
GC=${6:-3}
SHARD_SIZE=${7:-100}
STAGGER_S=${8:-0}

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$OUT"

START_TIME=$(date +%s)
echo "[$(date -Iseconds)] Launching $N processes × $M samples = $((N*M)) total. grid=${GR}x${GC} out=$OUT"

PIDS=()
for i in $(seq 0 $((N-1))); do
  TASK_NAME=$(printf 'task_%04d' "$i")
  TASK_DIR="$OUT/$TASK_NAME"
  LOG="$OUT/${TASK_NAME}.log"
  SEED=$((BASE_SEED + i * 1000000))
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    nohup uv run --project "$REPO_ROOT" python -m taskbench.data.collect \
      --n "$M" \
      --grid-rows "$GR" --grid-cols "$GC" \
      --seed "$SEED" \
      --task-id "$i" \
      --shard-size "$SHARD_SIZE" \
      --out "$TASK_DIR" \
      --resume \
      > "$LOG" 2>&1 &
  PIDS+=($!)
  if [ "$STAGGER_S" != "0" ] && [ "$i" -lt $((N-1)) ]; then
    sleep "$STAGGER_S"
  fi
done

echo "Launched PIDs: ${PIDS[*]}"

# Wait for all
RC=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    RC=1
    echo "PID $pid exited with non-zero status"
  fi
done

ELAPSED=$(($(date +%s) - START_TIME))
echo "[$(date -Iseconds)] All $N processes complete in ${ELAPSED}s. Exit code: $RC"
exit $RC
