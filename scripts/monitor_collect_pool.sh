#!/bin/bash
# Quick progress report for a collect pool. Counts shards/rows per task.
#
# Usage: monitor_collect_pool.sh OUT_DIR
#
# OUT_DIR is the parent that contains task_0000/, task_0001/, etc. Pair with
# `watch -n 30 scripts/monitor_collect_pool.sh <dir>` for live tracking.

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 OUT_DIR" >&2
  exit 2
fi

OUT=$1
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

uv run --project "$REPO_ROOT" python - "$OUT" <<'PY'
import sys
from pathlib import Path

import pyarrow.parquet as pq

out = Path(sys.argv[1])
task_dirs = sorted(out.glob("task_*"))
total_rows = 0
total_shards = 0
running = 0
for td in task_dirs:
    if not td.is_dir():
        continue
    shards = sorted(td.glob("shard_*.parquet"))
    rows = 0
    for s in shards:
        try:
            rows += pq.read_metadata(s).num_rows
        except Exception:
            pass
    total_rows += rows
    total_shards += len(shards)
    log = out / f"{td.name}.log"
    last = ""
    if log.exists():
        try:
            with open(log, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 4096))
                tail = f.read().decode(errors="replace").splitlines()
            last = tail[-1] if tail else ""
        except Exception:
            pass
    if last and "Wrote" in last and "done" in last:
        running += 1
    print(f"  {td.name}: {len(shards):>3} shards, {rows:>5} rows | {last[-90:]}")

print()
print(f"Tasks: {len(task_dirs)}  active(recent log): {running}")
print(f"Total shards: {total_shards}  Total rows: {total_rows}")
PY
