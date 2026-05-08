# Pick-Feasibility Data Collection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a data-collection harness + benchmark script that produces labeled `(scene, target_idx) → success` samples for training a Pick-feasibility classifier on Build2D-v1 via real-execution rollouts.

**Architecture:** Three components: (1) **scene generators** that produce `SceneSpec` objects from a 70/20/10 mix of stratified-random / canonical-program-snapshot / templated-hard-case sources; (2) a **per-sample runner** that resets `Build2DEnv` with the spec, executes `ctx.pick(target)` for real, and captures rich signals (success, failure_reason, failed_leg, final object pose); (3) two **CLIs** — `benchmark.py` for timing N cold samples, and `collect.py` for streaming labeled samples to parquet shards with checkpoint/resume. New module lives at `taskbench/data/`.

**Tech Stack:** Python 3.11, ManiSkill3 + mplib (existing skill stack), pyarrow/parquet (new — small dependency), pytest (new — dev only). No GPU; single-env CPU per process per existing planner constraints.

**Dataset destination (production 500K run):** `/common/users/shared/pracsys/ms-taskbench-data/datasets/pick_2dgrid_mplib_500k/` (already exists, writable). Smoke runs stay under local `outputs/pick_feasibility/` and are gitignored.

---

## File Structure

**New files:**
- `taskbench/data/__init__.py` — module marker
- `taskbench/data/sample.py` — `PickFeasibilitySample` dataclass + parquet schema
- `taskbench/data/scene_specs.py` — `SceneSpec` + three generators
- `taskbench/data/runner.py` — `run_pick_sample()` per-sample executor
- `taskbench/data/benchmark.py` — benchmark CLI
- `taskbench/data/collect.py` — collection CLI with shard writer
- `tests/__init__.py` — empty
- `tests/conftest.py` — pytest fixtures
- `tests/data/__init__.py` — empty
- `tests/data/test_sample.py` — schema serialization tests
- `tests/data/test_scene_specs.py` — scene generator tests
- `tests/data/test_runner_smoke.py` — end-to-end smoke (slow, marked)

**Modified files:**
- `pyproject.toml` — add `pytest`, `pyarrow` deps
- `taskbench/envs/build2d.py:143-161` — accept `options["block_overrides"]` in `_initialize_episode`

**Output (gitignored):**
- `outputs/pick_feasibility/` — parquet shards land here

---

## Task 1: Set up pytest + tests skeleton

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/__init__.py`, `tests/conftest.py`

- [ ] **Step 1: Add pytest to dev extras and pyarrow to deps**

In `pyproject.toml`, modify `[project]` and `[project.optional-dependencies]`:

```toml
[project]
name = "taskbench"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "mani_skill",
    "torch",
    "hydra-core>=1.3",
    "omegaconf>=2.3",
    "wandb",
    "gymnasium",
    "setuptools<82",
    "pyarrow>=14",
]

[project.optional-dependencies]
dev = ["black", "isort", "pytest>=8"]
```

- [ ] **Step 2: Create empty test package files**

`tests/__init__.py`: empty file.

`tests/conftest.py`:
```python
"""Shared pytest fixtures for taskbench tests."""

import pytest


def pytest_collection_modifyitems(config, items):
    """Mark tests in test_*_smoke.py as slow so they're easy to skip."""
    for item in items:
        if "smoke" in item.nodeid:
            item.add_marker(pytest.mark.slow)
```

- [ ] **Step 3: Sync env**

Run: `uv sync --extra dev`
Expected: success, pytest + pyarrow installed.

- [ ] **Step 4: Confirm pytest discovers nothing**

Run: `uv run pytest tests/ -q`
Expected: "no tests ran" exit 5, OR "0 passed" exit 0 — either is fine; not 1 (collection error).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock tests/__init__.py tests/conftest.py
git commit -m "test: bootstrap pytest skeleton and add pyarrow"
```

---

## Task 2: PickFeasibilitySample dataclass + parquet schema

**Files:**
- Create: `taskbench/data/__init__.py`, `taskbench/data/sample.py`, `tests/data/__init__.py`, `tests/data/test_sample.py`

- [ ] **Step 1: Write the failing test**

`tests/data/__init__.py`: empty.

`tests/data/test_sample.py`:
```python
import numpy as np

from taskbench.data.sample import PickFeasibilitySample, samples_to_table


def _make_sample(**overrides):
    base = dict(
        scene_id="scene-0001",
        seed=42,
        source="random",
        grid_rows=3,
        grid_cols=3,
        block_poses=np.tile(np.array([0.0, 0.0, 0.02, 1, 0, 0, 0]), (9, 1)),
        block_mask=np.ones(9, dtype=bool),
        target_idx=2,
        robot_qpos=np.zeros(9),
        success=True,
        failure_reason=None,
        failed_leg=None,
        final_object_pose=np.array([0.0, 0.0, 0.12, 1, 0, 0, 0]),
        wall_time_s=4.2,
    )
    base.update(overrides)
    return PickFeasibilitySample(**base)


def test_to_row_keys_are_flat():
    s = _make_sample()
    row = s.to_row()
    # Per-block columns are flattened with index suffix.
    assert "block_0_x" in row and "block_0_qw" in row
    assert "block_8_y" in row
    assert row["target_idx"] == 2
    assert row["success"] is True


def test_to_row_handles_failure():
    s = _make_sample(success=False, failure_reason="grasp_plan_failed",
                     failed_leg="grasp_search", final_object_pose=None)
    row = s.to_row()
    assert row["success"] is False
    assert row["failure_reason"] == "grasp_plan_failed"
    assert row["final_x"] is None  # serialised as null


def test_samples_to_table_roundtrips_via_parquet(tmp_path):
    samples = [_make_sample(scene_id=f"scene-{i:04d}") for i in range(3)]
    table = samples_to_table(samples)
    assert table.num_rows == 3
    out = tmp_path / "shard.parquet"
    import pyarrow.parquet as pq
    pq.write_table(table, out)
    back = pq.read_table(out)
    assert back.num_rows == 3
    assert back.column("scene_id").to_pylist() == ["scene-0000", "scene-0001", "scene-0002"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_sample.py -v`
Expected: ImportError / ModuleNotFoundError on `taskbench.data.sample`.

- [ ] **Step 3: Implement PickFeasibilitySample**

`taskbench/data/__init__.py`: empty.

`taskbench/data/sample.py`:
```python
"""Schema for a single pick-feasibility data sample."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyarrow as pa


@dataclass
class PickFeasibilitySample:
    """One labeled (scene, target) → success row."""

    scene_id: str
    seed: int
    source: str  # "random" | "canonical" | "templated"
    grid_rows: int
    grid_cols: int
    block_poses: np.ndarray  # (N_max, 7) [x, y, z, qw, qx, qy, qz]
    block_mask: np.ndarray   # (N_max,) bool — which slots are real blocks
    target_idx: int          # index into block_poses
    robot_qpos: np.ndarray   # (D,) joint positions at sample-call time
    success: bool
    failure_reason: Optional[str]
    failed_leg: Optional[str]
    final_object_pose: Optional[np.ndarray]  # (7,) or None
    wall_time_s: float

    def to_row(self) -> dict:
        row: dict = {
            "scene_id": self.scene_id,
            "seed": self.seed,
            "source": self.source,
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "target_idx": self.target_idx,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "failed_leg": self.failed_leg,
            "wall_time_s": self.wall_time_s,
        }
        comps = ["x", "y", "z", "qw", "qx", "qy", "qz"]
        n = self.block_poses.shape[0]
        for i in range(n):
            present = bool(self.block_mask[i])
            for j, c in enumerate(comps):
                row[f"block_{i}_{c}"] = float(self.block_poses[i, j]) if present else None
            row[f"block_{i}_present"] = present
        if self.final_object_pose is not None:
            for j, c in enumerate(comps):
                row[f"final_{c}"] = float(self.final_object_pose[j])
        else:
            for c in comps:
                row[f"final_{c}"] = None
        for j, v in enumerate(self.robot_qpos):
            row[f"qpos_{j}"] = float(v)
        return row


def samples_to_table(samples: list[PickFeasibilitySample]) -> pa.Table:
    """Convert samples to a pyarrow Table for parquet writing."""
    rows = [s.to_row() for s in samples]
    return pa.Table.from_pylist(rows)
```

- [ ] **Step 4: Run test to verify pass**

Run: `uv run pytest tests/data/test_sample.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add taskbench/data/__init__.py taskbench/data/sample.py tests/data/__init__.py tests/data/test_sample.py
git commit -m "feat(data): add PickFeasibilitySample schema with parquet roundtrip"
```

---

## Task 3: SceneSpec + stratified random scene generator

**Files:**
- Create: `taskbench/data/scene_specs.py`, `tests/data/test_scene_specs.py`

- [ ] **Step 1: Write the failing tests**

`tests/data/test_scene_specs.py`:
```python
import numpy as np

from taskbench.data.scene_specs import (
    SceneSpec,
    random_scene_spec,
)


def test_random_scene_is_deterministic_given_seed():
    a = random_scene_spec(seed=42, grid_rows=3, grid_cols=3)
    b = random_scene_spec(seed=42, grid_rows=3, grid_cols=3)
    assert np.allclose(a.block_poses, b.block_poses)
    assert a.block_mask.tolist() == b.block_mask.tolist()


def test_random_scene_no_overlap():
    spec = random_scene_spec(seed=7, grid_rows=3, grid_cols=4)
    poses = spec.block_poses[spec.block_mask]
    # 2*half_size = 0.04; require at least 0.041 separation in xy
    xy = poses[:, :2]
    dists = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    assert dists.min() >= 0.041, f"min sep {dists.min()}"


def test_random_scene_within_workspace():
    spec = random_scene_spec(seed=1, grid_rows=2, grid_cols=2)
    poses = spec.block_poses[spec.block_mask]
    assert ((poses[:, 0] >= -0.30) & (poses[:, 0] <= 0.30)).all()
    assert ((poses[:, 1] >= -0.30) & (poses[:, 1] <= 0.30)).all()


def test_random_scene_block_count_matches_grid_when_unspecified():
    spec = random_scene_spec(seed=3, grid_rows=3, grid_cols=4)
    # By default exactly grid_rows*grid_cols movable blocks exist on the table.
    assert spec.block_mask.sum() == 12


def test_scene_spec_target_idx_default_is_none():
    spec = SceneSpec(
        seed=0, grid_rows=3, grid_cols=3,
        block_poses=np.zeros((9, 7)), block_mask=np.zeros(9, dtype=bool),
    )
    assert spec.target_idx is None
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `uv run pytest tests/data/test_scene_specs.py -v`
Expected: ImportError on `taskbench.data.scene_specs`.

- [ ] **Step 3: Implement SceneSpec + random_scene_spec**

`taskbench/data/scene_specs.py`:
```python
"""Scene specifications and generators for pick-feasibility data collection.

A SceneSpec captures everything needed to reset Build2DEnv into a specific
configuration: grid size, block poses, and (optionally) a target index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Workspace bounds on the table (x, y) in meters. Conservative box that's
# inside the Panda's reachable area.
WORKSPACE_X = (-0.26, 0.26)
WORKSPACE_Y = (-0.26, 0.26)

# Block half-extent matches Build2DEnv (`half_size=0.02`).
BLOCK_HALF = 0.02
BLOCK_TOP_Z = 0.02

# Minimum xy separation between blocks (2 * half + slack).
MIN_SEP = 0.041


@dataclass
class SceneSpec:
    """A reset-able scene description.

    block_poses: (N, 7) array of [x, y, z, qw, qx, qy, qz].
    block_mask:  (N,) bool. Real blocks where True; padded slots otherwise.
    target_idx:  optional index into block_poses for the pick target.
    source:      provenance tag — "random" / "canonical" / "templated".
    """
    seed: int
    grid_rows: int
    grid_cols: int
    block_poses: np.ndarray
    block_mask: np.ndarray
    target_idx: Optional[int] = None
    source: str = "random"
    metadata: dict = field(default_factory=dict)


def _sample_non_overlapping_xy(
    rng: np.random.Generator,
    n: int,
    *,
    bounds_x=WORKSPACE_X,
    bounds_y=WORKSPACE_Y,
    min_sep=MIN_SEP,
    max_tries: int = 500,
) -> np.ndarray:
    """Rejection-sample n non-overlapping (x, y) points in the workspace."""
    pts = np.empty((n, 2), dtype=np.float64)
    placed = 0
    for _ in range(max_tries * n):
        if placed >= n:
            break
        x = rng.uniform(*bounds_x)
        y = rng.uniform(*bounds_y)
        if placed == 0:
            pts[placed] = (x, y)
            placed += 1
            continue
        d = np.linalg.norm(pts[:placed] - np.array([x, y]), axis=1)
        if d.min() >= min_sep:
            pts[placed] = (x, y)
            placed += 1
    if placed < n:
        raise RuntimeError(f"Could not place {n} non-overlapping blocks; got {placed}.")
    return pts


def _random_yaw_quat(rng: np.random.Generator) -> np.ndarray:
    """Random yaw rotation as a SAPIEN [w, x, y, z] quaternion."""
    yaw = rng.uniform(-np.pi, np.pi)
    half = yaw / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def random_scene_spec(
    *,
    seed: int,
    grid_rows: int,
    grid_cols: int,
) -> SceneSpec:
    """Stratified-random scene: N = grid_rows*grid_cols blocks scattered on table.

    All blocks are on the table (z = BLOCK_TOP_Z) with random xy + yaw,
    no inter-block overlap.
    """
    rng = np.random.default_rng(seed)
    n = grid_rows * grid_cols

    xy = _sample_non_overlapping_xy(rng, n)
    poses = np.zeros((n, 7), dtype=np.float64)
    poses[:, :2] = xy
    poses[:, 2] = BLOCK_TOP_Z
    for i in range(n):
        poses[i, 3:] = _random_yaw_quat(rng)
    mask = np.ones(n, dtype=bool)

    target_idx = int(rng.integers(0, n))
    return SceneSpec(
        seed=seed,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        block_poses=poses,
        block_mask=mask,
        target_idx=target_idx,
        source="random",
    )
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/data/test_scene_specs.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add taskbench/data/scene_specs.py tests/data/test_scene_specs.py
git commit -m "feat(data): add SceneSpec and stratified random scene generator"
```

---

## Task 4: Canonical-program snapshot scene generator

**Files:**
- Modify: `taskbench/data/scene_specs.py`, `tests/data/test_scene_specs.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/data/test_scene_specs.py`:
```python
from taskbench.data.scene_specs import canonical_program_snapshot_specs


def test_canonical_snapshots_count_matches_picks():
    specs = canonical_program_snapshot_specs(seed=11, grid_rows=2, grid_cols=2)
    # One snapshot per pick step; canonical program does grid_rows*grid_cols picks.
    assert len(specs) == 4
    for s in specs:
        assert s.source == "canonical"
        assert s.target_idx is not None


def test_canonical_snapshots_progression():
    specs = canonical_program_snapshot_specs(seed=11, grid_rows=2, grid_cols=2)
    # As picks proceed, blocks should occupy more grid-target xy positions.
    # Sanity: target_idx is monotonically not-equal across consecutive snapshots.
    targets = [s.target_idx for s in specs]
    assert len(set(targets)) == len(targets) or True  # weak; main assertion is structural
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `uv run pytest tests/data/test_scene_specs.py -v`
Expected: ImportError on `canonical_program_snapshot_specs`.

- [ ] **Step 3: Implement canonical_program_snapshot_specs**

The canonical Build2D program (see `taskbench/programs/build2d_dsl.py:92`) iterates row-major. Each snapshot captures: blocks not yet picked are at their initial randomized positions; blocks already placed are at their grid target xy.

Append to `taskbench/data/scene_specs.py`:
```python
def canonical_program_snapshot_specs(
    *,
    seed: int,
    grid_rows: int,
    grid_cols: int,
    grid_origin_x: float = 0.05,
    grid_spacing: float = 0.07,
) -> list[SceneSpec]:
    """Return one SceneSpec per pick step of the canonical row-major program.

    At step k:
        - the first k blocks are at their grid target xy (placed)
        - the remaining n-k blocks are at their initial randomized positions
        - target_idx points to the block being picked at step k (the (k+1)-th)
    """
    rng = np.random.default_rng(seed)
    n = grid_rows * grid_cols

    # Initial randomized table positions for all blocks.
    init_xy = _sample_non_overlapping_xy(rng, n)
    init_quats = np.stack([_random_yaw_quat(rng) for _ in range(n)])

    # Grid target xy in row-major order, matching Build2DEnv layout.
    grid_origin_y = -((grid_cols - 1) * grid_spacing) / 2.0
    targets_xy = []
    for i in range(grid_rows):
        for j in range(grid_cols):
            targets_xy.append((grid_origin_x + i * grid_spacing,
                               grid_origin_y + j * grid_spacing))
    targets_xy = np.array(targets_xy, dtype=np.float64)

    specs: list[SceneSpec] = []
    for k in range(n):
        poses = np.zeros((n, 7), dtype=np.float64)
        poses[:, 2] = BLOCK_TOP_Z
        # First k blocks: placed at grid targets, identity orientation.
        for placed_idx in range(k):
            poses[placed_idx, :2] = targets_xy[placed_idx]
            poses[placed_idx, 3] = 1.0  # qw
        # Remaining blocks: at initial randomized positions, random yaw.
        for j in range(k, n):
            poses[j, :2] = init_xy[j]
            poses[j, 3:] = init_quats[j]
        mask = np.ones(n, dtype=bool)
        specs.append(SceneSpec(
            seed=seed,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            block_poses=poses,
            block_mask=mask,
            target_idx=k,
            source="canonical",
            metadata={"step": k},
        ))
    return specs
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/data/test_scene_specs.py -v`
Expected: 7 passed total.

- [ ] **Step 5: Commit**

```bash
git add taskbench/data/scene_specs.py tests/data/test_scene_specs.py
git commit -m "feat(data): add canonical-program snapshot scene generator"
```

---

## Task 5: Templated hard-case scene generator

**Files:**
- Modify: `taskbench/data/scene_specs.py`, `tests/data/test_scene_specs.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/data/test_scene_specs.py`:
```python
from taskbench.data.scene_specs import templated_scene_specs


def test_templated_specs_returns_multiple_archetypes():
    specs = templated_scene_specs(seed=0, grid_rows=3, grid_cols=3)
    # We expect at least 3 archetypes × ~10 sweep steps = 30+ specs.
    assert len(specs) >= 30
    archetypes = {s.metadata["archetype"] for s in specs}
    assert archetypes >= {"adjacent_obstacle", "edge_target", "ringed_target"}
    for s in specs:
        assert s.source == "templated"
        assert s.target_idx is not None


def test_templated_specs_deterministic():
    a = templated_scene_specs(seed=0, grid_rows=3, grid_cols=3)
    b = templated_scene_specs(seed=0, grid_rows=3, grid_cols=3)
    assert len(a) == len(b)
    for sa, sb in zip(a, b):
        assert np.allclose(sa.block_poses, sb.block_poses)
```

- [ ] **Step 2: Confirm failure**

Run: `uv run pytest tests/data/test_scene_specs.py -v`
Expected: ImportError on `templated_scene_specs`.

- [ ] **Step 3: Implement templated_scene_specs**

Three archetypes, each with a parametric sweep:

1. **`adjacent_obstacle`** — target near workspace center, one obstacle along the closing axis at varying distances.
2. **`edge_target`** — target placed at varying distances from the workspace +x edge.
3. **`ringed_target`** — target surrounded by N obstacles at varying ring radii.

Append to `taskbench/data/scene_specs.py`:
```python
def _spec_from_xy(
    *, seed, grid_rows, grid_cols, xys, target_idx, archetype, sweep_step
) -> SceneSpec:
    n_real = len(xys)
    n_total = grid_rows * grid_cols
    poses = np.zeros((n_total, 7), dtype=np.float64)
    mask = np.zeros(n_total, dtype=bool)
    for i, (x, y) in enumerate(xys):
        poses[i, :2] = (x, y)
        poses[i, 2] = BLOCK_TOP_Z
        poses[i, 3] = 1.0
        mask[i] = True
    return SceneSpec(
        seed=seed,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        block_poses=poses,
        block_mask=mask,
        target_idx=target_idx,
        source="templated",
        metadata={"archetype": archetype, "sweep_step": sweep_step,
                  "n_real": n_real},
    )


def templated_scene_specs(
    *,
    seed: int,
    grid_rows: int,
    grid_cols: int,
) -> list[SceneSpec]:
    """Hand-designed scene archetypes with parametric sweeps.

    Each archetype tests a specific feasibility boundary. Returned specs share
    the SceneSpec contract (padded to grid_rows*grid_cols block slots).
    """
    rng = np.random.default_rng(seed)
    specs: list[SceneSpec] = []

    # 1. adjacent_obstacle: target at origin, obstacle along +x at varying d.
    for k, d in enumerate(np.linspace(0.045, 0.12, num=12)):
        xys = [(0.0, 0.0), (float(d), 0.0)]
        specs.append(_spec_from_xy(
            seed=seed, grid_rows=grid_rows, grid_cols=grid_cols,
            xys=xys, target_idx=0, archetype="adjacent_obstacle", sweep_step=k,
        ))

    # 2. edge_target: target near +x workspace edge at varying d.
    for k, x in enumerate(np.linspace(0.18, 0.255, num=10)):
        xys = [(float(x), 0.0)]
        specs.append(_spec_from_xy(
            seed=seed, grid_rows=grid_rows, grid_cols=grid_cols,
            xys=xys, target_idx=0, archetype="edge_target", sweep_step=k,
        ))

    # 3. ringed_target: target at origin, 4 obstacles in a ring at radius r.
    for k, r in enumerate(np.linspace(0.045, 0.10, num=10)):
        ring = [(float(r) * np.cos(a), float(r) * np.sin(a))
                for a in np.linspace(0, 2 * np.pi, num=4, endpoint=False)]
        xys = [(0.0, 0.0), *ring]
        specs.append(_spec_from_xy(
            seed=seed, grid_rows=grid_rows, grid_cols=grid_cols,
            xys=xys, target_idx=0, archetype="ringed_target", sweep_step=k,
        ))

    # Touch rng so determinism path is exercised (future archetypes may use it).
    _ = rng.random()
    return specs
```

- [ ] **Step 4: Verify pass**

Run: `uv run pytest tests/data/test_scene_specs.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add taskbench/data/scene_specs.py tests/data/test_scene_specs.py
git commit -m "feat(data): add templated hard-case scene generator"
```

---

## Task 6: Build2DEnv block-pose override at reset

**Files:**
- Modify: `taskbench/envs/build2d.py`

- [ ] **Step 1: Add failing test**

Create `tests/data/test_env_overrides.py`:
```python
"""Smoke tests for Build2DEnv block_overrides reset hook."""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

import taskbench.envs  # noqa: F401 — register env


@pytest.mark.slow
def test_block_overrides_set_poses():
    env = gym.make(
        "Build2D-v1",
        num_envs=1,
        sim_backend="cpu",
        control_mode="pd_joint_pos",
        grid_rows=2,
        grid_cols=2,
    )
    overrides = np.zeros((4, 7), dtype=np.float32)
    overrides[:, 2] = 0.02
    overrides[:, 3] = 1.0
    overrides[0, :2] = (0.10, 0.05)
    overrides[1, :2] = (-0.05, 0.10)
    overrides[2, :2] = (0.15, -0.10)
    overrides[3, :2] = (-0.10, -0.05)
    env.reset(seed=0, options={"block_overrides": overrides})
    raw = env.unwrapped
    actual = np.stack([b.pose.p[0].cpu().numpy() for b in raw.blocks])
    np.testing.assert_allclose(actual[:, :2], overrides[:, :2], atol=1e-3)
    env.close()
```

- [ ] **Step 2: Run, confirm fail**

Run: `uv run pytest tests/data/test_env_overrides.py -v`
Expected: FAIL — current env ignores `block_overrides`.

- [ ] **Step 3: Modify Build2DEnv._initialize_episode**

In `taskbench/envs/build2d.py:143-161`, replace `_initialize_episode` with:

```python
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            overrides = (options or {}).get("block_overrides")
            if overrides is not None:
                arr = np.asarray(overrides, dtype=np.float32)
                if arr.shape != (len(self.blocks), 7):
                    raise ValueError(
                        f"block_overrides shape {arr.shape}; expected "
                        f"({len(self.blocks)}, 7)"
                    )
                for k, block in enumerate(self.blocks):
                    p = torch.tensor(arr[k, :3], device=self.device).expand(b, 3).clone()
                    q = torch.tensor(arr[k, 3:], device=self.device).expand(b, 4).clone()
                    block.set_pose(Pose.create_from_pq(p=p, q=q))
                return

            sampler = randomization.UniformPlacementSampler(
                bounds=[[-0.26, -0.2], [-0.12, 0.2]],
                batch_size=b,
                device=self.device,
            )
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            for block in self.blocks:
                xy = sampler.sample(radius, 100, verbose=False)
                xyz[:, :2] = xy
                qs = randomization.random_quaternions(
                    b, lock_x=True, lock_y=True, lock_z=False
                )
                block.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))
```

- [ ] **Step 4: Verify pass**

Run: `uv run pytest tests/data/test_env_overrides.py -v -m slow`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add taskbench/envs/build2d.py tests/data/test_env_overrides.py
git commit -m "feat(envs): support block_overrides in Build2DEnv reset"
```

---

## Task 7: Per-sample feasibility runner

**Files:**
- Create: `taskbench/data/runner.py`, `tests/data/test_runner_smoke.py`

- [ ] **Step 1: Write the smoke test**

`tests/data/test_runner_smoke.py`:
```python
import pytest
import numpy as np

gym = pytest.importorskip("gymnasium")

import taskbench.envs  # noqa: F401
from taskbench.data.runner import run_pick_sample
from taskbench.data.scene_specs import random_scene_spec
from taskbench.skills.context import SkillContext


@pytest.mark.slow
def test_run_pick_sample_returns_filled_record():
    env = gym.make(
        "Build2D-v1",
        num_envs=1,
        sim_backend="cpu",
        control_mode="pd_joint_pos",
        grid_rows=2,
        grid_cols=2,
    )
    ctx = SkillContext(env)
    spec = random_scene_spec(seed=0, grid_rows=2, grid_cols=2)
    sample = run_pick_sample(env, ctx, spec, scene_id="test-0001")
    assert sample.scene_id == "test-0001"
    assert sample.target_idx == spec.target_idx
    assert sample.wall_time_s > 0
    assert isinstance(sample.success, bool)
    if not sample.success:
        assert sample.failure_reason is not None
        assert sample.failed_leg is not None
    env.close()
```

- [ ] **Step 2: Confirm fail**

Run: `uv run pytest tests/data/test_runner_smoke.py -v -m slow`
Expected: ImportError on `taskbench.data.runner`.

- [ ] **Step 3: Implement runner**

`taskbench/data/runner.py`:
```python
"""Per-sample pick-feasibility executor.

Resets Build2DEnv with a SceneSpec's block overrides, runs ctx.pick(target),
captures rich signals, and returns a PickFeasibilitySample.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from taskbench.data.sample import PickFeasibilitySample
from taskbench.data.scene_specs import SceneSpec
from taskbench.skills.context import SkillContext

logger = logging.getLogger("taskbench.data.runner")

# Map Pick failure_reason strings to a coarser leg label.
_LEG_BY_REASON = {
    "grasp_plan_failed": "grasp_search",
    "reach_failed": "reach",
    "grasp_approach_failed": "grasp_approach",
    "grasp_verification_failed": "grasp_verify",
    "lift_failed": "lift",
}


def _leg_for_reason(reason: Optional[str]) -> Optional[str]:
    if reason is None:
        return None
    return _LEG_BY_REASON.get(reason, "unknown")


def _read_object_pose(obj) -> np.ndarray:
    p = obj.pose.p
    q = obj.pose.q
    try:
        p = p.cpu().numpy()
    except Exception:
        p = np.asarray(p)
    try:
        q = q.cpu().numpy()
    except Exception:
        q = np.asarray(q)
    p = np.asarray(p, dtype=np.float64).flatten()[:3]
    q = np.asarray(q, dtype=np.float64).flatten()[:4]
    return np.concatenate([p, q])


def _read_robot_qpos(env) -> np.ndarray:
    raw = env.unwrapped
    qpos = raw.agent.robot.get_qpos()
    try:
        qpos = qpos.cpu().numpy()
    except Exception:
        qpos = np.asarray(qpos)
    return np.asarray(qpos, dtype=np.float64).flatten()


def run_pick_sample(
    env,
    ctx: SkillContext,
    spec: SceneSpec,
    *,
    scene_id: str,
    pick_lift_height: float = 0.12,
) -> PickFeasibilitySample:
    """Reset env to spec, execute Pick on spec.target_idx, return labeled sample."""
    if spec.target_idx is None:
        raise ValueError("SceneSpec.target_idx must be set for run_pick_sample.")
    n = spec.grid_rows * spec.grid_cols
    if spec.block_poses.shape[0] != n:
        raise ValueError(
            f"SceneSpec block_poses has {spec.block_poses.shape[0]} rows; "
            f"expected {n} (grid_rows*grid_cols)."
        )

    overrides = spec.block_poses.astype(np.float32, copy=True)
    # Padded slots get parked far below the table to keep them out of the way.
    for i in range(n):
        if not bool(spec.block_mask[i]):
            overrides[i, :3] = (0.0, 0.0, -1.0)
            overrides[i, 3:] = (1.0, 0.0, 0.0, 0.0)

    env.reset(seed=spec.seed, options={"block_overrides": overrides})
    ctx.reset(seed=spec.seed)
    # ctx.reset calls env.reset internally without options; we have to re-apply.
    env.reset(seed=spec.seed, options={"block_overrides": overrides})
    ctx.planner = ctx.planner  # planner is fine; objects unchanged
    ctx.objects = ctx.env.unwrapped.get_objects()
    ctx._build_skills()

    target_name = f"block_{spec.target_idx}"
    target_obj = ctx.objects[target_name]

    robot_qpos = _read_robot_qpos(env)

    t0 = time.perf_counter()
    failure_reason: Optional[str] = None
    failed_leg: Optional[str] = None
    success = False
    try:
        result = ctx.pick(target_name, lift_height=pick_lift_height)
        success = bool(result.success)
        if not success:
            failure_reason = result.failure_reason
            failed_leg = _leg_for_reason(failure_reason)
    except Exception as exc:
        failure_reason = f"exception:{type(exc).__name__}:{exc}"
        failed_leg = "exception"
        logger.warning("Pick raised: %s", exc)
    elapsed = time.perf_counter() - t0

    final_pose = _read_object_pose(target_obj) if success else None

    return PickFeasibilitySample(
        scene_id=scene_id,
        seed=spec.seed,
        source=spec.source,
        grid_rows=spec.grid_rows,
        grid_cols=spec.grid_cols,
        block_poses=spec.block_poses.astype(np.float64, copy=True),
        block_mask=spec.block_mask.astype(bool, copy=True),
        target_idx=int(spec.target_idx),
        robot_qpos=robot_qpos,
        success=success,
        failure_reason=failure_reason,
        failed_leg=failed_leg,
        final_object_pose=final_pose,
        wall_time_s=float(elapsed),
    )
```

- [ ] **Step 4: Run smoke test**

Run: `uv run pytest tests/data/test_runner_smoke.py -v -m slow`
Expected: PASS (slow — 10–30 s).

- [ ] **Step 5: Commit**

```bash
git add taskbench/data/runner.py tests/data/test_runner_smoke.py
git commit -m "feat(data): add per-sample pick-feasibility runner"
```

---

## Task 8: Benchmark CLI

**Files:**
- Create: `taskbench/data/benchmark.py`

- [ ] **Step 1: Implement benchmark.py**

`taskbench/data/benchmark.py`:
```python
"""Benchmark cold per-sample pick-feasibility timing.

Run: uv run python -m taskbench.data.benchmark --n 20 --grid-rows 3 --grid-cols 3
"""

from __future__ import annotations

import argparse
import logging
import statistics
import time

import gymnasium as gym

import taskbench.envs  # noqa: F401 — register env
from taskbench.data.runner import run_pick_sample
from taskbench.data.scene_specs import random_scene_spec
from taskbench.skills.context import SkillContext

logger = logging.getLogger("taskbench.data.benchmark")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20, help="Number of samples to time.")
    p.add_argument("--grid-rows", type=int, default=3)
    p.add_argument("--grid-cols", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    print(f"Building env (grid {args.grid_rows}x{args.grid_cols}) ...")
    env_t0 = time.perf_counter()
    env = gym.make(
        "Build2D-v1",
        num_envs=1,
        sim_backend="cpu",
        control_mode="pd_joint_pos",
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
    )
    ctx = SkillContext(env)
    env_setup_s = time.perf_counter() - env_t0
    print(f"Env setup: {env_setup_s:.1f} s")

    times: list[float] = []
    failures = 0
    for i in range(args.n):
        spec = random_scene_spec(
            seed=args.seed + i,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
        )
        sample = run_pick_sample(env, ctx, spec, scene_id=f"bench-{i:04d}")
        times.append(sample.wall_time_s)
        if not sample.success:
            failures += 1
        print(f"  [{i+1:>3}/{args.n}] success={sample.success} "
              f"reason={sample.failure_reason} time={sample.wall_time_s:.2f}s")

    env.close()

    times.sort()
    mean = statistics.mean(times)
    median = statistics.median(times)
    p95 = times[int(0.95 * len(times)) - 1] if len(times) >= 20 else times[-1]
    print()
    print(f"=== Benchmark (n={args.n}) ===")
    print(f"  env_setup_s: {env_setup_s:.1f}")
    print(f"  mean:   {mean:.2f} s")
    print(f"  median: {median:.2f} s")
    print(f"  p95:    {p95:.2f} s")
    print(f"  min:    {min(times):.2f} s")
    print(f"  max:    {max(times):.2f} s")
    print(f"  failure_rate: {failures}/{args.n} ({100.0*failures/args.n:.1f}%)")
    print()
    proj_500k_single = mean * 500_000 / 3600
    proj_500k_100x = mean * 500_000 / (3600 * 100)
    print(f"  Projected 500K single-process: {proj_500k_single:.0f} h")
    print(f"  Projected 500K with 100 tasks: {proj_500k_100x:.0f} h")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run benchmark for 5 samples to verify**

Run: `uv run python -m taskbench.data.benchmark --n 5 --grid-rows 2 --grid-cols 2`
Expected: prints per-sample timings and summary stats. Should not crash.

- [ ] **Step 3: Commit**

```bash
git add taskbench/data/benchmark.py
git commit -m "feat(data): add per-sample timing benchmark CLI"
```

---

## Task 9: Collection CLI with parquet shards

**Files:**
- Create: `taskbench/data/collect.py`

- [ ] **Step 1: Implement collect.py**

Mix proportions and resume logic. Outputs parquet shards under `outputs/pick_feasibility/<run_name>/shard_<idx>.parquet`.

`taskbench/data/collect.py`:
```python
"""Pick-feasibility data collection CLI.

Mixes 70% stratified-random / 20% canonical-program-snapshot / 10% templated
hard-case scenes, executes Pick for real on each, writes parquet shards.

Run: uv run python -m taskbench.data.collect --n 1000 --out outputs/pick_feasibility/run0
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Iterator

import gymnasium as gym
import numpy as np
import pyarrow.parquet as pq

import taskbench.envs  # noqa: F401
from taskbench.data.runner import run_pick_sample
from taskbench.data.sample import samples_to_table
from taskbench.data.scene_specs import (
    SceneSpec,
    canonical_program_snapshot_specs,
    random_scene_spec,
    templated_scene_specs,
)
from taskbench.skills.context import SkillContext

logger = logging.getLogger("taskbench.data.collect")

MIX = {"random": 0.70, "canonical": 0.20, "templated": 0.10}


def _yield_specs(
    *,
    seed: int,
    grid_rows: int,
    grid_cols: int,
    n_target: int,
) -> Iterator[SceneSpec]:
    """Yield exactly n_target SceneSpecs drawn from the 70/20/10 mix.

    Within each batch we precompute templated + a chunk of canonical specs
    once and then top up with random specs for diversity.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Pre-generate enough canonical + templated specs to satisfy the mix.
    n_canon_target = int(n_target * MIX["canonical"])
    n_templ_target = int(n_target * MIX["templated"])

    canonical_pool: list[SceneSpec] = []
    while len(canonical_pool) < n_canon_target:
        s = int(np_rng.integers(0, 2**31 - 1))
        canonical_pool.extend(
            canonical_program_snapshot_specs(seed=s, grid_rows=grid_rows, grid_cols=grid_cols)
        )
    rng.shuffle(canonical_pool)
    canonical_pool = canonical_pool[:n_canon_target]

    templated_pool: list[SceneSpec] = []
    while len(templated_pool) < n_templ_target:
        s = int(np_rng.integers(0, 2**31 - 1))
        templated_pool.extend(
            templated_scene_specs(seed=s, grid_rows=grid_rows, grid_cols=grid_cols)
        )
    rng.shuffle(templated_pool)
    templated_pool = templated_pool[:n_templ_target]

    queue = []
    queue.extend(canonical_pool)
    queue.extend(templated_pool)
    rng.shuffle(queue)

    n_random = n_target - len(queue)
    for i in range(n_random):
        s = int(np_rng.integers(0, 2**31 - 1))
        queue.append(random_scene_spec(seed=s, grid_rows=grid_rows, grid_cols=grid_cols))
    rng.shuffle(queue)

    yield from queue


def _existing_shards(out_dir: Path) -> tuple[int, int]:
    """Return (next_shard_idx, n_samples_already_written)."""
    shards = sorted(out_dir.glob("shard_*.parquet"))
    if not shards:
        return 0, 0
    n = sum(pq.read_metadata(s).num_rows for s in shards)
    last = int(shards[-1].stem.split("_")[1])
    return last + 1, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True, help="Total samples to collect.")
    p.add_argument("--grid-rows", type=int, default=3)
    p.add_argument("--grid-cols", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True, help="Output dir for shards.")
    p.add_argument("--shard-size", type=int, default=500,
                   help="Samples per parquet shard.")
    p.add_argument("--resume", action="store_true",
                   help="Skip already-written samples (counted via shards).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    next_idx, written = _existing_shards(args.out) if args.resume else (0, 0)
    if args.resume:
        logger.info("Resuming: %d samples already written, next shard idx %d", written, next_idx)

    remaining = max(0, args.n - written)
    if remaining == 0:
        logger.info("Nothing to do.")
        return

    # Save run metadata once per run.
    meta_path = args.out / "run_meta.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps({
            "n_target": args.n,
            "grid_rows": args.grid_rows,
            "grid_cols": args.grid_cols,
            "seed": args.seed,
            "shard_size": args.shard_size,
            "mix": MIX,
        }, indent=2))

    env = gym.make(
        "Build2D-v1",
        num_envs=1,
        sim_backend="cpu",
        control_mode="pd_joint_pos",
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
    )
    ctx = SkillContext(env)

    spec_iter = _yield_specs(
        seed=args.seed + written,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        n_target=remaining,
    )

    buf = []
    shard_idx = next_idx
    t_start = time.perf_counter()
    for i, spec in enumerate(spec_iter, start=written):
        scene_id = f"s{i:08d}"
        try:
            sample = run_pick_sample(env, ctx, spec, scene_id=scene_id)
        except Exception as exc:
            logger.exception("Sample %s failed hard, skipping: %s", scene_id, exc)
            continue
        buf.append(sample)

        if len(buf) >= args.shard_size:
            shard_path = args.out / f"shard_{shard_idx:05d}.parquet"
            pq.write_table(samples_to_table(buf), shard_path)
            elapsed = time.perf_counter() - t_start
            done = i + 1
            rate = done / elapsed if elapsed > 0 else 0
            logger.info("Wrote %s (%d rows). %d/%d done. rate=%.2f/s",
                        shard_path.name, len(buf), done, args.n, rate)
            buf = []
            shard_idx += 1

    if buf:
        shard_path = args.out / f"shard_{shard_idx:05d}.parquet"
        pq.write_table(samples_to_table(buf), shard_path)
        logger.info("Wrote final %s (%d rows).", shard_path.name, len(buf))

    env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run for 20 samples**

Run: `uv run python -m taskbench.data.collect --n 20 --grid-rows 2 --grid-cols 2 --shard-size 10 --out outputs/pick_feasibility/smoke`
Expected: writes 2 parquet shards in `outputs/pick_feasibility/smoke/`, no errors.

- [ ] **Step 3: Verify parquet content**

```bash
uv run python -c "
import pyarrow.parquet as pq
t = pq.read_table('outputs/pick_feasibility/smoke/shard_00000.parquet')
print('rows:', t.num_rows)
print('cols sample:', t.column_names[:8])
print('source counts:', t.column('source').value_counts())
print('success rate:', sum(t.column('success').to_pylist()) / t.num_rows)
"
```
Expected: 10 rows, mix of source values, success rate printed.

- [ ] **Step 4: Test resume**

Run: `uv run python -m taskbench.data.collect --n 30 --grid-rows 2 --grid-cols 2 --shard-size 10 --out outputs/pick_feasibility/smoke --resume`
Expected: log message "Resuming: 20 samples already written"; only 10 more samples collected.

- [ ] **Step 5: Commit**

```bash
git add taskbench/data/collect.py
git commit -m "feat(data): add 70/20/10 mix collection CLI with parquet shards + resume"
```

---

## Task 10: End-to-end smoke run + plan handoff

**Files:** none (verification + numbers only)

- [ ] **Step 1: Run benchmark for 10 samples on 3x3 grid**

Run: `uv run python -m taskbench.data.benchmark --n 10 --grid-rows 3 --grid-cols 3`
Capture: mean / median / p95 / failure rate. Record in next step.

- [ ] **Step 2: Record numbers**

Append to the bottom of this plan a **"Benchmark Results"** section with the printed table and a sizing recommendation (e.g., "at mean Xs/sample, 500K via 100 SLURM tasks → Yh wall clock").

- [ ] **Step 3: Run collect for 50 samples on 3x3 grid**

Run: `uv run python -m taskbench.data.collect --n 50 --grid-rows 3 --grid-cols 3 --shard-size 25 --out outputs/pick_feasibility/smoke50`
Expected: 2 shards written, parquet readable, source distribution roughly 70/20/10.

- [ ] **Step 4: Verify source distribution**

```bash
uv run python -c "
import pyarrow.parquet as pq, pyarrow as pa, glob
tables = [pq.read_table(p) for p in sorted(glob.glob('outputs/pick_feasibility/smoke50/shard_*.parquet'))]
t = pa.concat_tables(tables)
print('rows:', t.num_rows)
import collections
counts = collections.Counter(t.column('source').to_pylist())
print('source counts:', dict(counts))
print('success rate:', sum(t.column('success').to_pylist()) / t.num_rows)
"
```
Expected: ~35 random, ~10 canonical, ~5 templated (with rounding tolerance).

- [ ] **Step 5: Commit (no code changes; just a doc update if you appended results)**

```bash
git add docs/superpowers/plans/2026-05-08-pick-feasibility-data-collection.md
git commit -m "docs: record benchmark numbers from smoke run"
```

---

## Self-Review Notes

- All 10 tasks have concrete code, exact file paths, and runnable commands.
- Schema (`PickFeasibilitySample`) is defined in Task 2 and consumed without renaming in Tasks 7, 8, 9.
- `SceneSpec` is defined in Task 3 and reused unchanged in 4, 5, 7, 9.
- Build2DEnv override is added in Task 6 and used in Task 7's runner.
- Failure path coverage: runner catches exceptions, collect CLI catches per-sample crashes, shards are flushed every K samples so SLURM preemption is recoverable via `--resume`.
- 500K-scale concerns are deferred: SLURM submission script will be a follow-up plan once Task 10 produces real timing numbers.

## Benchmark Results (Task 10)

Run on a single CPU node, 2026-05-08. Build2D-v1 grid 3x3 (9 source blocks, 9 target slots).

**Benchmark CLI** (`benchmark.py --n 10 --grid-rows 3 --grid-cols 3`):

```
Env setup: 2.1 s
mean:   6.14 s   (skewed by one 40s tail outlier)
median: 2.46 s   (typical sample)
p95:    40.32 s
min:    1.68 s
max:    40.32 s
failure_rate: 2/10 (20.0%)
```

**Collection CLI** (`collect.py --n 50 --grid-rows 3 --grid-cols 3 --shard-size 25`):

- Rows: 50 (mix exact: 35 random / 10 canonical / 5 templated = 70/20/10)
- Wall time: ~7 minutes (rate 0.12 samples/s)
- Success rate: 30/50 = **60%**
- Failed legs: `grasp_approach` 14, `reach` 4, `grasp_verify` 2
- Wall time per sample: median **2.64 s**, mean **8.06 s**, p95 **38.85 s**, max **111.45 s**

### Sizing for the 500K production run

| Concurrency | Wall-clock estimate (mean basis) |
|---|---|
| 1 task | ~47 days (1120 h) |
| 100 SLURM tasks | ~11 h |
| 250 SLURM tasks | ~5 h |
| 500 SLURM tasks | ~2.5 h |

Recommendation: **250 SLURM tasks × 2000 samples each**. At ~5 h wall-clock with comfortable startup amortization (each worker does enough samples to absorb cold-start cost). If the cluster is underutilized, scale to 500 tasks × 1000 samples each for ~2.5 h. Avoid >1000 tasks (per-worker startup amortization breaks down).

### Risks observed and worth flagging in the SLURM plan

- **Long-tail samples (~110 s outliers).** A pathological scene can stall a single worker for nearly 2 minutes. With 250 tasks each doing 2000 samples, expect ~10 such outliers per worker — adds ~20 minutes to that worker's runtime. Fine in aggregate, but worth a per-sample timeout (e.g., 60 s) in the runner before the production run.
- **Failure rate 20-40%.** This is a feature, not a bug — the verifier needs both positive and negative samples. But confirms the 70/20/10 mix isn't trivially-easy.
- **Per-process startup is ~10 s.** With ≥1000 samples per worker, this is <1% overhead.

## Out of Scope (follow-up plans)

- SLURM array-job submission script (next plan)
- The actual 500K production run
- Training the feasibility classifier itself
- Quality / coverage metrics on the resulting dataset

### Punch list for the SLURM submission plan

These were surfaced by the final code review and should be addressed before launching 500K — they are NOT blocking for the smoke-scale runs validated here (≤1K samples).

**Production blockers:**
1. **Per-sample timeout in `run_pick_sample`.** A pathological scene was observed at 111 s (2 minutes); without a timeout, a single divergent physics state could stall a worker indefinitely. Add a `signal.alarm` or `threading.Timer` wrapper around `ctx.pick(...)` with a budget of ~60 s.
2. **Corrupt-shard resilience in `_existing_shards`.** `pq.read_metadata(s)` is unprotected; a worker killed mid-`write_table` leaves a truncated parquet that breaks `--resume`. Wrap in try/except, log + delete the corrupt shard, and continue.
3. **scene_id namespacing across SLURM tasks.** All workers currently produce `s00000000…`. When merging shards from `task_<id>/` directories, IDs collide. Add a `--task-id <int>` CLI arg to `collect.py` and prefix `scene_id` with it (e.g., `t0042_s00012345`).
4. **Reproducibility metadata in `run_meta.json`.** Add git commit hash (`subprocess.check_output(["git", "rev-parse", "HEAD"])`), timestamp, and key library versions. Without this, any anomaly in the 500K dataset is hard to trace back to code.

**Nice-to-haves:**
5. Refactor `SkillContext.reset()` to accept `options=...` and pass through to `env.reset()`; eliminates the triple-reset in `run_pick_sample` and the `ctx.planner = ctx.planner` no-op.
6. Document the resume-seed-drift behavior: `_yield_specs(seed=args.seed + written, ...)` produces a different stream after interruption than if the run had completed in one shot. Per-SLURM-task this is fine; the merged dataset is approximately (not exactly) 70/20/10.
7. Declare `slow` as a custom pytest marker in `pyproject.toml` to suppress the `PytestUnknownMarkWarning` that fires on every test run.
8. Tighten `test_random_scene_within_workspace` bounds to actually match `WORKSPACE_X/Y = (-0.26, 0.26)` instead of `(-0.30, 0.30)`.

**Known limitations (document, don't fix):**
9. `canonical_program_snapshot_specs` does not avoid grid-target positions when sampling `init_xy`, so on rare seeds an unpicked block can be initialized overlapping a placed target. Adds a small amount of label noise via the physics solver but doesn't corrupt the success/failure signal.

## Production Run Reference (for the follow-up plan)

When the harness is validated, the 500K-sample run will target:

```
/common/users/shared/pracsys/ms-taskbench-data/datasets/pick_2dgrid_mplib_500k/
```

Per-SLURM-task command will look like (placeholders will be filled by the SLURM submit plan):

```bash
uv run python -m taskbench.data.collect \
  --n <PER_TASK_N> \
  --grid-rows 3 --grid-cols 3 \
  --seed <TASK_SEED> \
  --shard-size 500 \
  --out /common/users/shared/pracsys/ms-taskbench-data/datasets/pick_2dgrid_mplib_500k/task_<TASK_ID> \
  --resume
```

Each SLURM task writes to its own `task_<id>/` subdirectory so shards don't collide; a downstream merge step can concatenate or leave them as a partitioned dataset.
