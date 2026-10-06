#!/usr/bin/env python3
"""Compare regenerated case-study tables and raw-source hashes with the archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "derived"
FRAME_KEYS = [
    "run_id",
    "simulation_step",
    "relative_radius_half_width",
    "threshold_nm",
]
TRAJECTORY_KEYS = ["run_id", "relative_radius_half_width", "threshold_nm"]


def compare_table(filename: str, keys: list[str], reproduced: Path) -> set[str]:
    current = pd.read_csv(reproduced / filename)
    reference = pd.read_csv(REFERENCE / filename)
    run_ids = set(current["run_id"])
    if not run_ids:
        raise AssertionError(f"No runs in {filename}")
    expected = reference[reference["run_id"].isin(run_ids)]
    current = current.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        current,
        expected,
        check_exact=False,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    print(f"TABLE_OK {filename} rows={len(current)} runs={len(run_ids)}")
    return run_ids


def compare_sources(reproduced: Path, run_ids: set[str]) -> None:
    actual_metadata = json.loads(
        (reproduced / "interval_radius_case_study.json").read_text(encoding="utf-8")
    )
    expected_metadata = json.loads(
        (REFERENCE / "interval_radius_case_study.json").read_text(encoding="utf-8")
    )
    actual = {
        item["run_id"]: (item["relative_path"], item["sha256"])
        for item in actual_metadata["included_sources"]
    }
    expected = {
        item["run_id"]: (item["relative_path"], item["sha256"])
        for item in expected_metadata["included_sources"]
        if item["run_id"] in run_ids
    }
    if actual != expected:
        raise AssertionError("Raw-source names or SHA-256 hashes differ")
    print(f"SOURCES_OK runs={len(actual)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "reproduced_dir",
        nargs="?",
        type=Path,
        default=ROOT / "data" / "reproduced",
    )
    args = parser.parse_args()
    frame_runs = compare_table(
        "interval_radius_frame_metrics.csv", FRAME_KEYS, args.reproduced_dir
    )
    trajectory_runs = compare_table(
        "interval_radius_trajectory_summary.csv",
        TRAJECTORY_KEYS,
        args.reproduced_dir,
    )
    if frame_runs != trajectory_runs:
        raise AssertionError("Run IDs differ between reproduced tables")
    compare_sources(args.reproduced_dir, frame_runs)
    print("REPRODUCTION_COMPARISON_OK")


if __name__ == "__main__":
    main()
