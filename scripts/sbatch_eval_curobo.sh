#!/bin/bash -l
# Template for cuRobo controller eval on SLURM.
# Required env vars:
#   VERSION   - controller tag (e.g., c7_slip_k24)
#   TOPK      - TASKBENCH_CUROBO_TOPK value (0 = use all candidates)
set -euo pipefail

source /common/home/st1122/Projects/ms-taskbench/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TASKBENCH_MOTION_BACKEND=curobo
export TASKBENCH_CUROBO_TOPK="${TOPK:-0}"
export TASKBENCH_CUROBO_INVERT="${INVERT:-0}"
export TASKBENCH_CUROBO_GRIPPER_STEPS="${GSTEPS:-6}"
export TASKBENCH_CUROBO_FINGER_COLL="${FINGER_COLL:-0}"
export TASKBENCH_CUROBO_SETTLE_STEPS="${SETTLE:-0}"
export TASKBENCH_CUROBO_TWO_STAGE_CLOSE="${TWO_STAGE:-0}"
export TASKBENCH_PANDA_WRIST_STIFFNESS_MULT="${WRIST_K:-1.0}"
export TASKBENCH_PANDA_WRIST_DAMPING_MULT="${WRIST_D:-1.0}"

cd /common/home/st1122/Projects/ms-taskbench

python scripts/eval_controller.py \
    --scenes outputs/controller_eval/baseline_scenes.parquet \
    --out outputs/controller_eval/${VERSION}.parquet \
    --version "${VERSION}"
