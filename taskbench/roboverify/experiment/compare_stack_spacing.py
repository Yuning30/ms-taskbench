"""Collect Stack demonstrations with a controlled initial spacing sweep.

All spacings use the production bounded sampler and retain the original workspace,
table height and seed-dependent cube rotations. Retries use a local RNG rather
than advancing to another requested seed. Spacing is max(abs(dx), abs(dy)), as
in Scattered, not Euclidean distance or an edge-to-edge gap.
"""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from taskbench.roboverify.backend import BLOCK_LENGTH, StackBackend
from taskbench.roboverify.cfg.collection import record_execution, validate_trace
from taskbench.roboverify.cfg.program_source import load_program
from taskbench.roboverify.cfg.recordings import load_traces, save_traces
from taskbench.roboverify.stack_reset import XY_HIGH, XY_LOW
from taskbench.roboverify.util.on import using_block_length
from taskbench.roboverify.verification_lib.highlevel_verification_lib import (
    HighLevelContext,
)

PROGRAM = "taskbench.roboverify.examples.stack:build_program"


def failure_kind(trace):
    if trace.metadata["status"] == "valid":
        return "accepted"
    reason = trace.metadata.get("reason", "")
    if "LayoutSamplingError" in reason:
        return "sampling"
    if "pick:" in reason:
        return "pick"
    if "move_relative:" in reason:
        return "move"
    if "release:" in reason:
        return "release"
    if "guard witness" in reason:
        return "loop"
    return trace.metadata["status"]


def collect_one(blocks, separation, seed, output, *, save_video=False):
    details = {}

    def factory():
        env = StackBackend(blocks, separation=separation)
        original_close = env.close

        def close():
            try:
                details["layout_sampling"] = env.layout_sampling
                details["final_checks"] = {
                    k: bool(v.item()) for k, v in env.raw.evaluate().items()
                }
            finally:
                original_close()

        env.close = close
        return env

    definition = load_program(PROGRAM, HighLevelContext(), blocks)
    trace = record_execution(
        definition,
        seed=seed,
        num_blocks=blocks,
        env_factory=factory,
        max_loop_iterations=blocks - 1,
        timeout_seconds=60,
        video_path=output / "videos" / f"seed_{seed:04d}.mp4" if save_video else None,
    )
    trace.metadata["spacing_experiment"] = details
    valid = bool(validate_trace(trace))
    trace_path = None
    if trace.states:
        trace_path = output / "traces" / f"seed_{seed:04d}.npz"
        save_traces(trace_path, [trace])
    row = dict(
        seed=seed,
        valid=valid,
        status=trace.metadata["status"],
        reason=trace.metadata["reason"],
        failure_kind=failure_kind(trace),
        actions=len(trace.actions[0]),
        seconds=trace.metadata["elapsed_seconds"],
        trace=str(trace_path.relative_to(output)) if trace_path else None,
        **details,
    )
    if trace.states:
        observation = np.asarray(trace.states[0])
        positions = np.array(
            [observation[10 + 12 * i : 13 + 12 * i] for i in range(blocks)]
        )
        row["initial_positions"] = positions.tolist()
        row["layout_hash"] = hashlib.sha256(positions.tobytes()).hexdigest()
        row["minimum_spacing_m"] = min(
            float(np.max(np.abs(a[:2] - b[:2])))
            for i, a in enumerate(positions)
            for b in positions[i + 1 :]
        )
    return row


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-blocks", type=int, choices=(3, 4), required=True)
    parser.add_argument("--separation-cm", type=float, required=True)
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()
    if args.num_seeds < 1 or not 0 <= args.seed_start < 2**32 - args.num_seeds + 1:
        parser.error("Require a positive count and seeds in [0, 2**32)")
    separation = args.separation_cm / 100
    if not np.isfinite(separation) or separation < 2 * BLOCK_LENGTH:
        parser.error("Separation must be finite and at least 8 cm")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    (output / "traces").mkdir()
    if args.save_video:
        (output / "videos").mkdir()
    project = Path(__file__).resolve().parents[3]
    files = [
        Path(__file__).resolve(),
        project / "taskbench/envs/stack_n_cube.py",
        project / "taskbench/roboverify/backend.py",
        project / "taskbench/roboverify/stack_reset.py",
        project / "taskbench/roboverify/api/control.py",
        project / "taskbench/roboverify/cfg/collection.py",
        project / "taskbench/roboverify/examples/stack.py",
    ]
    config = dict(
        blocks=args.num_blocks,
        separation_m=separation,
        seeds=list(range(args.seed_start, args.seed_start + args.num_seeds)),
        workspace_xy_m=[XY_LOW.tolist(), XY_HIGH.tolist()],
        gripper_exclusion_m=0,
        settling_steps=50,
        max_loop_iterations=args.num_blocks - 1,
        timeout_seconds=60,
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project, text=True
        ).strip(),
        source_sha256={
            str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
    )
    write_json(output / "config.json", config)
    report = dict(config=config, status="collecting", rows=[])
    write_json(output / "collection.json", report)
    with using_block_length(BLOCK_LENGTH):
        for seed in config["seeds"]:
            row = collect_one(
                args.num_blocks, separation, seed, output, save_video=args.save_video
            )
            report["rows"].append(row)
            report["counts"] = dict(Counter(r["failure_kind"] for r in report["rows"]))
            write_json(output / "collection.json", report)
            if len(report["rows"]) % 10 == 0 or not row["valid"]:
                print(
                    f"{args.num_blocks} blocks {args.separation_cm:g} cm "
                    f"{len(report['rows'])}/{args.num_seeds}: {report['counts']}",
                    flush=True,
                )
    report["status"] = "complete"
    report["unique_layouts"] = len(
        {r["layout_hash"] for r in report["rows"] if "layout_hash" in r}
    )
    if all(r["valid"] for r in report["rows"]):
        save_traces(
            output / "demonstrations.npz",
            [load_traces(output / r["trace"])[0] for r in report["rows"]],
        )
    write_json(output / "collection.json", report)
    print(
        json.dumps(
            dict(counts=report["counts"], unique_layouts=report["unique_layouts"])
        )
    )


if __name__ == "__main__":
    main()
