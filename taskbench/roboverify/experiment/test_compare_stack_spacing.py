"""Spacing sweeps must respect geometry, seeds, and explicit sampling budgets."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from taskbench.roboverify.cfg.demos import DemoTrace
from taskbench.roboverify.experiment.compare_stack_spacing import collect_one
from taskbench.roboverify.stack_reset import (
    XY_HIGH,
    XY_LOW,
    LayoutSamplingError,
    sample_layout,
)


class SpacingSamplerTests(unittest.TestCase):
    def test_sweep_has_complete_distinct_layouts_with_requested_clearance(self):
        for blocks in (3, 4):
            for separation in (0.08, 0.10, 0.12, 0.14, 0.16):
                layouts = set()
                for seed in range(100):
                    with self.subTest(blocks=blocks, separation=separation, seed=seed):
                        xy, stats = sample_layout(blocks, separation, seed)
                        self.assertEqual(xy.shape, (blocks, 2))
                        self.assertTrue(np.all(xy >= XY_LOW))
                        self.assertTrue(np.all(xy <= XY_HIGH))
                        self.assertGreaterEqual(stats["proposals"], blocks)
                        for i, a in enumerate(xy):
                            for b in xy[i + 1 :]:
                                self.assertGreaterEqual(
                                    np.max(np.abs(a - b)), separation
                                )
                        layouts.add(xy.tobytes())
                self.assertEqual(len(layouts), 100)

    def test_reproducible_and_independent_of_global_rng(self):
        before = np.random.get_state()
        xy, stats = sample_layout(4, 0.14, 67)
        repeated, repeated_stats = sample_layout(4, 0.14, 67)
        after = np.random.get_state()
        np.testing.assert_array_equal(xy, repeated)
        self.assertEqual(stats, repeated_stats)
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_impossible_layout_fails_without_relaxing_spacing(self):
        with self.assertRaisesRegex(LayoutSamplingError, "after 2 layouts"):
            sample_layout(4, 0.5, 0, max_layouts=2, max_trials=3)
        for separation in (0.079, np.nan, np.inf):
            with self.assertRaises(ValueError):
                sample_layout(4, separation, 0)

    def test_sampling_failure_is_counted_without_an_empty_trace_archive(self):
        trace = DemoTrace(
            (),
            actions=((), ()),
            metadata=dict(
                status="failed",
                reason="LayoutSamplingError: no layout",
                elapsed_seconds=0.1,
            ),
        )
        module = "taskbench.roboverify.experiment.compare_stack_spacing"
        with (
            TemporaryDirectory() as directory,
            patch(module + ".load_program"),
            patch(module + ".record_execution", return_value=trace),
        ):
            result = collect_one(4, 0.5, 0, Path(directory))
            self.assertFalse(result["valid"])
            self.assertEqual(result["failure_kind"], "sampling")
            self.assertIsNone(result["trace"])
            self.assertFalse(list(Path(directory).glob("**/*.npz")))


if __name__ == "__main__":
    unittest.main()
