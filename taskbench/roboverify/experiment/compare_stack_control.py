"""Paired Panda measurements of the old and updated RoboVerify controllers.

Adapted from RoboVerify's compare_stack_control/compare_stack_paths experiments.
Historical behavior is patched only inside this serial diagnostic. The production
controller has no legacy mode. Full demo traces stay in memory; outputs are metrics
and sampled paths, not demonstration archives.
"""

import argparse
import json
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from taskbench.roboverify.api import control
from taskbench.roboverify.api.control import PrimitiveController
from taskbench.roboverify.backend import BLOCK_LENGTH, StackBackend
from taskbench.roboverify.cfg import collection, program_source
from taskbench.roboverify.cfg.collection import record_execution, validate_trace
from taskbench.roboverify.cfg.program_source import load_program
from taskbench.roboverify.experiment.run_logger import RunLogger
from taskbench.roboverify.util.on import using_block_length
from taskbench.roboverify.verification_lib.highlevel_verification_lib import (
    HighLevelContext,
)

PHASES = ("approach", "descend", "lift", "transfer", "lower", "retreat")
MODES = ("previous", "updated")


@contextmanager
def previous_controls():
    """Reproduce f748f42: component clipping, 10 mm Pick, Z-only Release.

    Pick's tolerance is set on the diagnostic program by make_definition().
    Patching both command generation and the backend matters: patching only one
    would still let the other normalize the historical command uniformly.
    """

    def raw_action(observation, target_position, *, gain=20.0, close_gripper=False):
        return np.r_[
            gain * (np.asarray(target_position) - np.asarray(observation)[:3]),
            -0.2 if close_gripper else 0.0,
        ]

    def clipped_action(action):
        action = np.asarray(action, dtype=float)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("Expected four finite action coordinates")
        return np.clip(action, -1, 1)

    def release(controller, box_id, offset):
        target = controller.observation[:3].copy()
        target[2] = controller.box_position(box_id)[2] + offset
        if not controller.gripper(opened=True):
            return False
        return controller.move(
            target, close_gripper=False, vertical_only=True, phase="retreat"
        )

    with (
        patch.object(control, "get_move_action", raw_action),
        patch.object(collection, "get_move_action", raw_action),
        patch("taskbench.roboverify.backend.bound_delta_action", clipped_action),
        patch.object(PrimitiveController, "release", release),
        patch.object(program_source, "CARTESIAN_ACTION_MODE", "component-clip-v1"),
    ):
        yield


def make_definition(num_blocks, mode):
    definition = load_program(
        "taskbench.roboverify.examples.stack:build_program",
        HighLevelContext(),
        num_blocks,
    )
    if mode == "previous":
        pick = definition.program.instructions[1].body[0]
        pick.control = replace(pick.control, position_tolerance=0.01)
    return definition


def segment_distance(points, start, target):
    """Distance to the finite intended segment; overshoot is not ignored."""
    points, start, target = map(np.asarray, (points, start, target))
    vector = target - start
    length2 = vector @ vector
    if length2 == 0:
        return np.linalg.norm(points - start, axis=1)
    progress = np.clip((points - start) @ vector / length2, 0, 1)
    return np.linalg.norm(points - start - progress[:, None] * vector, axis=1)


@contextmanager
def measure(rows):
    """Observe control boundaries without changing actions or stepping physics."""
    original_move, original_step = PrimitiveController.move, PrimitiveController._step
    original_pick, original_relative = (
        PrimitiveController.pick,
        PrimitiveController.move_relative,
    )
    active, placement, relative_number = None, 0, 0

    def pick(controller, box_id):
        nonlocal placement, relative_number
        placement += 1
        relative_number = 0
        return original_pick(controller, box_id)

    def relative(controller, references, offsets):
        nonlocal relative_number
        controller._measured_phase = ("lift", "transfer", "lower")[relative_number % 3]
        relative_number += 1
        return original_relative(controller, references, offsets)

    def move(controller, target, **kwargs):
        nonlocal active
        phase = kwargs.get("phase", getattr(controller, "_measured_phase", "move"))
        tcp_start = controller.observation[:3].copy()
        target = np.asarray(target).copy()
        box_id = kwargs.get("payload_id")
        start = (
            tcp_start.copy()
            if box_id is None
            else controller.box_position(box_id).copy()
        )
        tcp_target = target + tcp_start - start
        if phase == "retreat":
            # Reference the vertical segment through XY captured before opening.
            start[:2] = target[:2]
            tcp_start[:2] = target[:2]
            tcp_target = target.copy()
        row = dict(
            phase=phase,
            placement=placement,
            box_id=box_id,
            start=start.tolist(),
            target=target.tolist(),
            tcp_target=tcp_target.tolist(),
            positions=[controller.observation[:3].tolist()],
            payload=[],
            converged=False,
        )
        if box_id is not None:
            row["payload"].append(controller.box_position(box_id).tolist())
        active = row
        try:
            row["converged"] = bool(original_move(controller, target, **kwargs))
            return row["converged"]
        finally:
            active = None
            tcp = np.asarray(row["positions"])
            points = tcp if box_id is None else np.asarray(row["payload"])
            row.update(
                steps=len(tcp) - 1,
                endpoint_mm=float(np.linalg.norm(points[-1] - target) * 1000),
                path_mm=float(segment_distance(points, start, target).max() * 1000),
                tcp_path_mm=float(
                    segment_distance(tcp, tcp_start, tcp_target).max() * 1000
                ),
            )
            if phase == "retreat":
                row["xy_max_mm"] = float(
                    np.linalg.norm(tcp[:, :2] - target[:2], axis=1).max() * 1000
                )
                row["xy_end_mm"] = float(
                    np.linalg.norm(tcp[-1, :2] - target[:2]) * 1000
                )
            if box_id is not None:
                offsets = points - tcp
                row["attachment_change_mm"] = float(
                    np.linalg.norm(offsets - offsets[0], axis=1).max() * 1000
                )
            rows.append(row)

    def step(controller, action):
        result = original_step(controller, action)
        if active is not None:
            active["positions"].append(controller.observation[:3].tolist())
            if active["box_id"] is not None:
                active["payload"].append(
                    controller.box_position(active["box_id"]).tolist()
                )
        return result

    with (
        patch.object(PrimitiveController, "pick", pick),
        patch.object(PrimitiveController, "move_relative", relative),
        patch.object(PrimitiveController, "move", move),
        patch.object(PrimitiveController, "_step", step),
    ):
        yield


def check_snapshot(actual, expected):
    # PhysX normalizes float32 poses on restore (observed roundoff: 1.5e-8).
    # Check every exposed field, record the discrepancy, and keep this tighter
    # than the production action-replay tolerance (1e-5).
    if (
        actual.bindings != expected.bindings
        or actual.arrays.keys() != expected.arrays.keys()
    ):
        raise AssertionError("Paired initial snapshot structure or bindings differ")
    maximum = 0.0
    for key, value, restored in [("gt_state", expected.gt_state, actual.gt_state)] + [
        (key, value, actual.arrays[key]) for key, value in expected.arrays.items()
    ]:
        if np.issubdtype(np.asarray(value).dtype, np.floating):
            np.testing.assert_allclose(restored, value, atol=1e-7, rtol=0, err_msg=key)
        else:
            np.testing.assert_array_equal(restored, value, err_msg=key)
        maximum = max(maximum, float(np.max(np.abs(restored - value), initial=0)))
    return maximum


def run_trial(
    num_blocks, seed, mode, initial_snapshot=None, *, instrument=True, timeout=60
):
    phases, details = [], {}

    def factory():
        env = StackBackend(num_blocks)
        details["sampling_seconds"] = float(env.raw.control_timestep)
        original_close = env.close

        def close():
            try:
                details["physical_checks"] = {
                    k: bool(v.item()) for k, v in env.raw.evaluate().items()
                }
            finally:
                original_close()

        env.close = close
        return env

    with previous_controls() if mode == "previous" else nullcontext():
        definition = make_definition(num_blocks, mode)
        with measure(phases) if instrument else nullcontext():
            trace = record_execution(
                definition,
                seed=seed,
                num_blocks=num_blocks,
                env_factory=factory,
                initial_snapshot=initial_snapshot,
                max_loop_iterations=num_blocks - 1,
                timeout_seconds=timeout,
            )
    valid = validate_trace(trace)
    if initial_snapshot is not None and trace.snapshots:
        details["initial_snapshot_max_error"] = check_snapshot(
            trace.snapshots[0], initial_snapshot
        )
    controls = [e["control"] for e in trace.events if "control" in e]
    positions = (
        np.array(
            [trace.states[-1][10 + 12 * i : 13 + 12 * i] for i in range(num_blocks)]
        )
        if trace.states
        else None
    )
    return trace, dict(
        num_blocks=num_blocks,
        seed=seed,
        mode=mode,
        valid=bool(valid),
        status=trace.metadata["status"],
        reason=trace.metadata["reason"],
        actions=len(trace.actions[0]),
        phases=phases,
        instruction_steps=[c["steps"] for c in controls],
        # Failed instructions raise before instruction_end: retain the reason and
        # measured partial phases, rather than relying only on completed events.
        failed_phases=[p["phase"] for p in phases if not p["converged"]],
        final_tower_xy_mm=(
            None
            if positions is None
            else float(
                np.linalg.norm(positions[1:, :2] - positions[0, :2], axis=1).max()
                * 1000
            )
        ),
        **details,
    )


def distribution(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return dict(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        p95=float(np.percentile(values, 95)),
        maximum=float(max(values)),
    )


def summarize(records):
    summary = {}
    for blocks in sorted({r["num_blocks"] for r in records}):
        group = {}
        for mode in MODES:
            runs = [
                r for r in records if r["num_blocks"] == blocks and r["mode"] == mode
            ]
            phases = [p for r in runs for p in r["phases"]]
            group[mode] = dict(
                executions=len(runs),
                valid=sum(r["valid"] for r in runs),
                failures=[
                    {
                        k: r[k]
                        for k in ("seed", "status", "reason", "physical_checks")
                        if k in r
                    }
                    for r in runs
                    if not r["valid"]
                ],
                actions=distribution([r["actions"] for r in runs]),
                instruction_steps=distribution(
                    [s for r in runs for s in r["instruction_steps"]]
                ),
                final_tower_xy_mm=distribution([r["final_tower_xy_mm"] for r in runs]),
                phases={
                    phase: {
                        metric: distribution(
                            [
                                p[metric]
                                for p in phases
                                if p["phase"] == phase and metric in p
                            ]
                        )
                        for metric in (
                            "path_mm",
                            "tcp_path_mm",
                            "endpoint_mm",
                            "steps",
                            "xy_max_mm",
                            "xy_end_mm",
                            "attachment_change_mm",
                        )
                    }
                    for phase in PHASES
                },
            )
        previous = {
            r["seed"]: r
            for r in records
            if r["num_blocks"] == blocks and r["mode"] == "previous"
        }
        group["regression_seeds"] = [
            r["seed"]
            for r in records
            if r["num_blocks"] == blocks
            and r["mode"] == "updated"
            and not r["valid"]
            and previous[r["seed"]]["valid"]
        ]
        summary[str(blocks)] = group
    return summary


def write_report(logger, records, *, plots=False):
    summary = summarize(records)
    logger.write_artifact(
        "executions.json", json.dumps(records, indent=2, allow_nan=False)
    )
    logger.write_artifact(
        "summary.json", json.dumps(summary, indent=2, allow_nan=False)
    )
    lines = [
        "# Panda Stack controller comparison",
        "",
        "Previous: component clipping, 10 mm Pick, Z-only Release. Updated: uniform XYZ scaling, 2 mm Pick, fixed-XY/full-3D Release.",
        "",
        "Pairs restore the same serialized settled state. All seeds and failures are retained. Metrics sample control boundaries; they are not continuous-time guarantees. Opening/closing are excluded from motion phases; Release retains pre-opening XY.",
        "",
        "For carrying moves, path/endpoint errors track the cube against its requested destination. TCP errors use the corresponding translated segment. Intentional phase turns are excluded.",
        "",
    ]

    def fmt(value):
        return (
            "—"
            if value is None
            else f"{value['median']:.3f} / {value['p95']:.3f} / {value['maximum']:.3f}"
        )

    for blocks, group in summary.items():
        lines += [
            f"## {blocks} cubes",
            "",
            "| Mode | Valid programs | Actions: median / P95 / max |",
            "| --- | --- | --- |",
        ]
        for mode in MODES:
            s = group[mode]
            lines.append(
                f"| {mode} | {s['valid']}/{s['executions']} | {fmt(s['actions'])} |"
            )
        lines += [
            "",
            f"Newly failing baseline-successful seeds: {group['regression_seeds']}",
            "",
            "Path deviation (mm), median / P95 / max:",
            "",
            "| Phase | Previous | Updated |",
            "| --- | --- | --- |",
        ]
        for phase in PHASES:
            lines.append(
                f"| {phase} | {fmt(group['previous']['phases'][phase]['path_mm'])} | {fmt(group['updated']['phases'][phase]['path_mm'])} |"
            )
        lines += [
            "",
            "Failure details and endpoint/payload statistics are in summary.json; sampled paths and physical success checks are in executions.json.",
            "",
        ]
    logger.write_artifact("report.md", "\n".join(lines))
    if plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            len(summary),
            2,
            figsize=(11, 4 * len(summary)),
            squeeze=False,
            layout="constrained",
        )
        for row, blocks in enumerate(summary):
            for col, metric in enumerate(("path_mm", "xy_end_mm")):
                ax = axes[row, col]
                for mode in MODES:
                    values = sorted(
                        p[metric]
                        for r in records
                        if r["num_blocks"] == int(blocks) and r["mode"] == mode
                        for p in r["phases"]
                        if metric in p
                    )
                    if values:
                        ax.step(
                            values,
                            np.arange(1, len(values) + 1) / len(values) * 100,
                            label=mode,
                        )
                ax.set(
                    title=f"{blocks} cubes: {'path deviation (all phases)' if col == 0 else 'Release final XY error'}",
                    xlabel="mm",
                    ylabel="Samples at or below (%)",
                )
                ax.grid(alpha=0.2)
                ax.legend()
        fig.savefig(logger.artifact_dir() / "comparison.png", dpi=160)
        fig.savefig(logger.artifact_dir() / "comparison.svg")
        plt.close(fig)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-blocks", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--trajectory-timeout-seconds", type=float, default=60)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--run-name", default="paired-panda")
    args = parser.parse_args(argv)
    if (
        args.num_seeds < 1
        or args.seed_start < 0
        or args.seed_start + args.num_seeds > 2**32
    ):
        parser.error("Require a positive seed count and seeds in [0, 2**32)")
    if any(n < 2 or n > 6 for n in args.num_blocks) or len(set(args.num_blocks)) != len(
        args.num_blocks
    ):
        parser.error("Require distinct block counts from 2 through 6")
    if (
        not np.isfinite(args.trajectory_timeout_seconds)
        or args.trajectory_timeout_seconds <= 0
    ):
        parser.error("Require a positive finite timeout")
    records = []
    total = len(args.num_blocks) * args.num_seeds * 2
    with (
        using_block_length(BLOCK_LENGTH),
        RunLogger(
            args.output_dir, "stack-control", vars(args), slug=args.run_name
        ) as logger,
    ):
        logger.progress_line(f"Controller comparison: {logger.run_dir}")
        for blocks in args.num_blocks:
            for seed in range(args.seed_start, args.seed_start + args.num_seeds):
                snapshot = None
                for mode in MODES:
                    trace, record = run_trial(
                        blocks,
                        seed,
                        mode,
                        snapshot,
                        timeout=args.trajectory_timeout_seconds,
                    )
                    records.append(record)
                    if not trace.snapshots:
                        write_report(logger, records)
                        raise RuntimeError(
                            f"No settled state for {blocks} cubes, seed {seed}: {record['reason']}"
                        )
                    snapshot = trace.snapshots[0] if snapshot is None else snapshot
                    logger.log_metrics(
                        len(records),
                        blocks=blocks,
                        seed=seed,
                        mode=mode,
                        valid=record["valid"],
                        actions=record["actions"],
                    )
                    logger.set_progress(
                        len(records) - 1, total, blocks=blocks, seed=seed, mode=mode
                    )
                if (seed - args.seed_start + 1) % 10 == 0:
                    write_report(logger, records)
                    logger.progress_line(f"Completed {len(records)}/{total} executions")
        summary = write_report(logger, records, plots=True)
        regressions = {
            n: s["regression_seeds"]
            for n, s in summary.items()
            if s["regression_seeds"]
        }
        logger.finish(
            "regressions" if regressions else "measured", regression_seeds=regressions
        )
        logger.progress_line(f"Report: {logger.run_dir}/artifacts/report.md")
    return 2 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
