#!/usr/bin/env python3

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS = Path(__file__).resolve().parents[1] / "data" / "derived"


class IntervalRadiusCaseStudyResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frames = pd.read_csv(RESULTS / "interval_radius_frame_metrics.csv")
        cls.trajectories = pd.read_csv(
            RESULTS / "interval_radius_trajectory_summary.csv"
        )
        cls.metadata = json.loads(
            (RESULTS / "interval_radius_case_study.json").read_text(encoding="utf-8")
        )

    def test_expected_analysis_grain(self) -> None:
        self.assertEqual(len(self.frames), 630)
        self.assertEqual(self.frames["run_id"].nunique(), 15)
        self.assertEqual(
            self.frames[["run_id", "simulation_step"]].drop_duplicates().shape[0],
            90,
        )
        self.assertFalse(
            self.frames.duplicated(
                [
                    "run_id",
                    "simulation_step",
                    "relative_radius_half_width",
                    "threshold_nm",
                ]
            ).any()
        )

    def test_all_exact_intervals_are_ordered(self) -> None:
        lower_columns = [column for column in self.frames if column.endswith("_lower")]
        self.assertGreater(len(lower_columns), 0)
        for lower_column in lower_columns:
            upper_column = lower_column[: -len("_lower")] + "_upper"
            self.assertIn(upper_column, self.frames)
            self.assertTrue(
                np.all(
                    self.frames[lower_column].to_numpy()
                    <= self.frames[upper_column].to_numpy() + 1e-14
                ),
                msg=lower_column,
            )

    def test_degenerate_intervals_collapse(self) -> None:
        point = self.frames[np.isclose(self.frames["relative_radius_half_width"], 0.0)]
        self.assertTrue((point["radius_dependent_edge_count"] == 0).all())
        width_columns = [column for column in point if column.endswith("_width")]
        self.assertTrue((point[width_columns].abs() <= 1e-14).all().all())

    def test_nonisolated_duplicate_metric_is_absent(self) -> None:
        self.assertFalse(any("recruited_fraction" in column for column in self.frames))
        self.assertIn("isolated_fraction_lower", self.frames)
        self.assertIn("isolated_fraction_upper", self.frames)

    def test_edge_partition_and_raw_trajectory_counts(self) -> None:
        np.testing.assert_array_equal(
            self.frames["possible_edge_count"].to_numpy()
            - self.frames["guaranteed_edge_count"].to_numpy(),
            self.frames["radius_dependent_edge_count"].to_numpy(),
        )
        self.assertIn("mean_guaranteed_edge_count", self.trajectories)
        self.assertIn("mean_possible_edge_count", self.trajectories)
        self.assertEqual(
            self.metadata["schema_version"],
            "simsv-interval-radius-case-study-1.1",
        )

    def test_source_completion_contract(self) -> None:
        quality = self.metadata["data_quality"]
        self.assertEqual(quality["complete_hdf5_sources"], 15)
        self.assertEqual(quality["available_frames"], 3000)
        self.assertTrue(quality["all_selected_runs_completed"])
        self.assertTrue(quality["all_expected_frames_saved"])
        self.assertEqual(len(self.metadata["included_sources"]), 15)
        for source in self.metadata["included_sources"]:
            self.assertRegex(source["sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
