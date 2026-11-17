#!/usr/bin/env python3
"""Regenerate checksums for small archival artifacts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INCLUDED = [
    "analysis/analyze_bbbc039_masks.py",
    "analysis/compare_radius_uncertainty_baselines.py",
    "analysis/analyze_critical_widths.py",
    "analysis/benchmark_pair_search.py",
    "analysis/benchmark_width_sweep.py",
    "analysis/plot_critical_width_audit.py",
    "data/derived/critical_widths.json",
    "data/derived/bbbc039_radius_graph.json",
    "data/derived/radius_baseline_comparison.json",
    "data/derived/interval_radius_case_study.json",
    "data/derived/interval_radius_frame_metrics.csv",
    "data/derived/interval_radius_trajectory_summary.csv",
    "data/derived/pair_search_benchmark.json",
    "data/derived/width_sweep_benchmark.json",
    "data/provenance/RUN_PROVENANCE.csv",
    "src/interval_radius_graph.py",
    "src/critical_width_sweep.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    lines = ["# SHA-256 manifest for the archived release artifacts."]
    for relative in INCLUDED:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        lines.append(f"{sha256(path)}  {relative}")
    destination = ROOT / "MANIFEST.sha256"
    temporary = ROOT / ".MANIFEST.sha256.tmp"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    os.replace(temporary, destination)
    print("WROTE MANIFEST.sha256")


if __name__ == "__main__":
    main()
