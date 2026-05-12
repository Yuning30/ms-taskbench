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
if [ -n "${POS_TOL:-}" ]; then export TASKBENCH_CUROBO_POS_TOL="${POS_TOL}"; fi
if [ -n "${ORI_TOL:-}" ]; then export TASKBENCH_CUROBO_ORI_TOL="${ORI_TOL}"; fi
if [ -n "${CUBE_MU:-}" ]; then export TASKBENCH_CUBE_FRICTION="${CUBE_MU}"; fi
if [ -n "${GRIP_K:-}" ]; then export TASKBENCH_PANDA_GRIPPER_STIFFNESS_MULT="${GRIP_K}"; fi
if [ -n "${GRIP_D:-}" ]; then export TASKBENCH_PANDA_GRIPPER_DAMPING_MULT="${GRIP_D}"; fi

cd /common/home/st1122/Projects/ms-taskbench

python scripts/eval_controller.py \
    --scenes outputs/controller_eval/baseline_scenes.parquet \
    --out outputs/controller_eval/${VERSION}.parquet \
    --version "${VERSION}"
