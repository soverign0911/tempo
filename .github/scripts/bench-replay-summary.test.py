"""Replay estimates and intervals must use the same per-run matched blocks."""

import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/bench-replay-summary.py"
MODULE = runpy.run_path(str(SCRIPT))
METRICS = ("mean", "p50", "p90", "p99", "mgas_s", "wall_clock")


def block(number, latency_ms, wait_us=1000):
    return {
        "number": number, "gas_used": 1000000, "gas_limit": 30000000,
        "tx_count": 1, "new_payload_ms": latency_ms, "forkchoice_updated_ms": 0,
        "persistence_wait_us": wait_us,
    }


def report(baseline_runs, feature_runs):
    inputs = {}
    outputs = {}
    baseline_paths = [f"baseline-{i}.json" for i in range(len(baseline_runs))]
    feature_paths = [f"feature-{i}.json" for i in range(len(feature_runs))]
    for paths, runs in [(baseline_paths, baseline_runs), (feature_paths, feature_runs)]:
        for path, rows in zip(paths, runs):
            inputs[path] = {"blocks": rows}

    class Output(io.StringIO):
        def close(self):
            outputs[self.name] = self.getvalue()
            super().close()

    def memory_open(name, mode="r", *args, **kwargs):
        if mode == "r":
            return io.StringIO(json.dumps(inputs[name]))
        assert mode == "w"
        stream = Output()
        stream.name = name
        return stream

    argv = [str(SCRIPT), "--baseline-json", *baseline_paths,
            "--feature-json", *feature_paths, "--output-summary", "summary.json",
            "--output-markdown", "comment.md"]
    with (patch("builtins.open", memory_open), patch.object(sys, "argv", argv),
          contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())):
        MODULE["main"]()
    return json.loads(outputs["summary.json"]), outputs["comment.md"]


class MatchedCoverageTests(unittest.TestCase):
    def assert_neutral(self, summary, count):
        self.assertEqual(summary["baseline"]["stats"]["n"], count)
        self.assertEqual(summary["feature"]["stats"]["n"], count)
        self.assertEqual(summary["paired"]["n"], count)
        for metric in METRICS:
            with self.subTest(metric=metric):
                self.assertEqual(summary["changes"][metric],
                                 {"pct": 0.0, "ci_pct": 0.0, "sig": "neutral"})

    def test_unmatched_baseline_block_cannot_invent_improvement(self):
        summary, markdown = report([[block(1, 100), block(2, 900)]], [[block(1, 100)]])
        self.assert_neutral(summary, 1)
        self.assertNotIn("✅", markdown)

    def test_unmatched_feature_block_cannot_invent_regression(self):
        summary, _ = report([[block(1, 100)]], [[block(1, 100), block(2, 900)]])
        self.assert_neutral(summary, 1)

    def test_matching_is_per_run_pair(self):
        summary, _ = report(
            [[block(1, 100), block(2, 900)], [block(2, 200)]],
            [[block(1, 100)], [block(2, 200), block(1, 800)]],
        )
        self.assert_neutral(summary, 2)
        self.assertEqual(summary["baseline"]["stats"]["mean_ms"], 150.0)

    def test_duplicate_numbers_use_the_same_last_row_as_paired_intervals(self):
        summary, _ = report([[block(1, 900), block(1, 100)]], [[block(1, 100)]])
        self.assert_neutral(summary, 1)

    def test_disjoint_reports_still_fail(self):
        with self.assertRaises(SystemExit) as exited:
            report([[block(1, 100)]], [[block(2, 100)]])
        self.assertEqual(exited.exception.code, 1)

    def test_equal_coverage_retains_significant_changes(self):
        for feature_ms, significance in [(100, "neutral"), (50, "good"), (200, "bad")]:
            with self.subTest(feature_ms=feature_ms):
                summary, _ = report([[block(1, 100), block(2, 100)]],
                                    [[block(1, feature_ms), block(2, feature_ms)]])
                for metric in METRICS:
                    self.assertEqual(summary["changes"][metric]["sig"], significance)

    def test_wait_time_stats_also_exclude_unmatched_blocks(self):
        summary, _ = report([[block(1, 100), block(2, 900, wait_us=999000)]], [[block(1, 100)]])
        self.assertEqual(summary["baseline"]["stats"]["mean_persist_ms"], 1.0)
        waits = summary["wait_times"]["persistence_wait_us"]
        self.assertEqual(waits["baseline"], waits["feature"])
        self.assertEqual(summary["changes"]["persist_wait"]["sig"], "neutral")

    def test_empty_intersection_in_one_pair_cannot_contaminate_other_pairs(self):
        summary, _ = report([[block(1, 900)], [block(2, 100)]],
                            [[block(3, 800)], [block(2, 100)]])
        self.assert_neutral(summary, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
