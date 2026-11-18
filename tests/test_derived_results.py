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

    def test_critical_width_audit_partition_and_provenance(self) -> None:
        audit = json.loads((RESULTS / "critical_widths.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["schema"], "critical-width-analysis-1")
        self.assertEqual(audit["widths"], [0.0, 0.025, 0.05, 0.10])
        self.assertTrue(audit["source_hashes_checked"])
        self.assertTrue(audit["independent_endpoint_checks"])
        records = audit["records"]
        self.assertEqual(len(records), 15)
        self.assertEqual(len({record["run_id"] for record in records}), 15)
        for record in records:
            self.assertEqual(len(record["endpoints"]), 4)
            quotient = record["quotient_at_0p1"]
            endpoints = record["endpoints"][-1]
            self.assertEqual(
                quotient["guaranteed_components"] - quotient["possible_components"],
                quotient["component_mergers"],
            )
            self.assertEqual(
                quotient["ambiguous_edges_inside_guaranteed_components"]
                + quotient["ambiguous_edges_between_guaranteed_components"],
                endpoints["possible"]["edges"] - endpoints["guaranteed"]["edges"],
            )
            self.assertEqual(quotient["guaranteed_components"], endpoints["guaranteed"]["components"])
            self.assertEqual(quotient["possible_components"], endpoints["possible"]["components"])

    def test_width_sweep_benchmark_record_contract(self) -> None:
        report = json.loads((RESULTS / "width_sweep_benchmark.json").read_text(encoding="utf-8"))
        self.assertEqual(report["schema"], "critical-width-benchmark-1")
        records = report["records"]
        self.assertEqual(len(records), 6)
        self.assertEqual({row["n_nodes"] for row in records}, {3713, 7427, 14853})
        self.assertEqual({row["query_width_count"] for row in records}, {4, 40})
        for row in records:
            self.assertTrue(row["exact_summary_equality"])
            self.assertEqual(len(row["repeated_csr_seconds"]), 5)
            self.assertEqual(len(row["width_sweep_seconds"]), 5)
            self.assertAlmostEqual(
                row["median_time_ratio_repeated_over_sweep"],
                row["repeated_csr_median_seconds"] / row["width_sweep_median_seconds"],
            )

    def test_radius_baselines_respect_exact_envelopes(self) -> None:
        report = json.loads((RESULTS / "radius_baseline_comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(report["schema"], "radius-baselines-2")
        self.assertTrue(report["source_hashes_checked"])
        records = report["records"]
        self.assertEqual(len(records), 15)
        self.assertEqual(len({record["run_id"] for record in records}), 15)
        self.assertTrue(any(row["scalar_edge_disagreement"] > 0 for row in records))
        for row in records:
            self.assertEqual(row["draws"], 1024)
            self.assertTrue(row["type_pair_q_constant"])
            self.assertTrue(row["type_pair_possible_edges_equal"])
            self.assertTrue(row["type_pair_guaranteed_edges_equal"])
            for law in ("independent_uniform", "common_uniform"):
                short = row["sampling"][law]["summaries"]["64"]
                long = row["sampling"][law]["summaries"]["1024"]
                self.assertTrue(row["guaranteed_edges"] <= long["edge_min"] <= long["edge_max"] <= row["possible_edges"])
                self.assertTrue(row["guaranteed_lcc"] <= long["lcc_min"] <= long["lcc_max"] <= row["possible_lcc"])
                self.assertLessEqual(short["edge_max"] - short["edge_min"], long["edge_max"] - long["edge_min"])
                self.assertLessEqual(short["lcc_span"], long["lcc_span"])
            self.assertEqual(row["scalar_edges"], row["possible_edges"])

    def test_independent_mask_case_study_contract(self) -> None:
        report = json.loads((RESULTS / "bbbc039_radius_graph.json").read_text(encoding="utf-8"))
        self.assertEqual(report["schema"], "bbbc039-radius-graph-1")
        self.assertEqual(report["image_count"], 200)
        self.assertEqual(sum(row["n_objects"] for row in report["records"]), 23615)
        self.assertEqual(sum(row["n_objects"] >= 2 for row in report["records"]), 197)
        for row in report["records"]:
            for item in row["thresholds"]:
                self.assertLessEqual(item["guaranteed_edges"], item["possible_edges"])
                self.assertGreaterEqual(item["guaranteed_components"], item["possible_components"])
                self.assertLessEqual(item["guaranteed_lcc"], item["possible_lcc"])


if __name__ == "__main__":
    unittest.main()
