"""Unit tests for collect.py argument parsing, scene_id namespacing, and run_meta.

Covers Fix 3 (--task-id flag) and Fix 4 (git_hash + started_at in run_meta).
No real env or network calls required.
"""

from __future__ import annotations

import json

import pytest

from taskbench.data.collect import _build_parser, _build_run_meta


# ---------------------------------------------------------------------------
# Fix 3: --task-id argparse flag
# ---------------------------------------------------------------------------

class TestBuildParser:
    def setup_method(self):
        self.parser = _build_parser()

    def _action_dests(self):
        return {a.dest for a in self.parser._actions}

    def test_task_id_flag_exists(self):
        assert "task_id" in self._action_dests()

    def test_task_id_default_is_none(self):
        # Parse minimal required args.
        args = self.parser.parse_args(["--n", "10", "--out", "/tmp/x"])
        assert args.task_id is None

    def test_task_id_int(self):
        args = self.parser.parse_args(["--n", "10", "--out", "/tmp/x", "--task-id", "7"])
        assert args.task_id == 7

    def test_task_id_zero(self):
        args = self.parser.parse_args(["--n", "10", "--out", "/tmp/x", "--task-id", "0"])
        assert args.task_id == 0


# ---------------------------------------------------------------------------
# Fix 4: _build_run_meta returns git_hash + started_at
# ---------------------------------------------------------------------------

class TestBuildRunMeta:
    def _make_args(self, task_id=None):
        parser = _build_parser()
        argv = ["--n", "100", "--out", "/tmp/x"]
        if task_id is not None:
            argv += ["--task-id", str(task_id)]
        return parser.parse_args(argv)

    def test_git_hash_key_present(self):
        meta = _build_run_meta(self._make_args())
        assert "git_hash" in meta

    def test_git_hash_is_string(self):
        meta = _build_run_meta(self._make_args())
        assert isinstance(meta["git_hash"], str)
        assert len(meta["git_hash"]) > 0

    def test_started_at_key_present(self):
        meta = _build_run_meta(self._make_args())
        assert "started_at" in meta

    def test_started_at_is_iso8601_utc(self):
        meta = _build_run_meta(self._make_args())
        ts = meta["started_at"]
        assert isinstance(ts, str)
        assert ts.endswith("Z"), f"expected ISO-8601 UTC string ending in Z, got {ts!r}"
        # Must be parseable.
        import datetime
        datetime.datetime.fromisoformat(ts.rstrip("Z"))

    def test_task_id_none_in_meta(self):
        meta = _build_run_meta(self._make_args(task_id=None))
        assert meta["task_id"] is None

    def test_task_id_int_in_meta(self):
        meta = _build_run_meta(self._make_args(task_id=42))
        assert meta["task_id"] == 42

    def test_standard_fields_present(self):
        meta = _build_run_meta(self._make_args())
        for key in ("n_target", "grid_rows", "grid_cols", "seed", "shard_size", "mix"):
            assert key in meta, f"missing key: {key}"
