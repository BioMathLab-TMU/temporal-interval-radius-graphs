#!/usr/bin/env python3
"""Compare exact interval bounds with illustrative sampling and a scalar cutoff.

The sampled law is a diagnostic assumption, not an inferred physical model.
All reported inputs are byte-checked against the provenance manifest.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from interval_radius_graph import _component_summary, build_interval_radius_graph  # noqa: E402
from benchmark_pair_search import read_frame, sha256  # noqa: E402


def compare_frame(positions, radii, types, *, seed: int, draws: int = 64,
                  width: float = 0.10, threshold: float = 2.0) -> dict:
    varied = types == 1
    lower = radii * (1 - width * varied)
    upper = radii * (1 + width * varied)
    graph = build_interval_radius_graph(positions, lower, upper, threshold)
    pairs = graph.candidate_pairs
    i, j = pairs[:, 0], pairs[:, 1]
    distances = np.linalg.norm(positions[i] - positions[j], axis=1)
    nominal_gap = distances - radii[i] - radii[j]
    possible = distances <= upper[i] + upper[j] + threshold
    guaranteed = distances <= lower[i] + lower[j] + threshold
    possible_count = int(np.sum(possible))

    # A single scalar cutoff is calibrated to approximately reproduce the
    # possible-edge count, making disagreement in edge identity informative.
    if possible_count:
        cutoff = float(np.partition(nominal_gap, possible_count - 1)[possible_count - 1])
        scalar = nominal_gap <= cutoff
    else:
        cutoff = float("-inf")
        scalar = np.zeros(len(pairs), dtype=bool)
    scalar_summary = _component_summary(len(radii), pairs[scalar])
    scalar_disagreement = int(np.count_nonzero(scalar != possible))

    rng = np.random.default_rng(seed)
    sample_lcc = []
    sample_components = []
    sample_edges = []
    for _ in range(draws):
        sampled = rng.uniform(lower, upper)
        mask = distances <= sampled[i] + sampled[j] + threshold
        count = int(np.count_nonzero(mask))
        summary = _component_summary(len(radii), pairs[mask])
        sample_edges.append(count)
        sample_lcc.append(summary["largest_component_fraction"])
        sample_components.append(summary["n_components"])
        if np.any(mask & ~possible) or np.any(guaranteed & ~mask):
            raise AssertionError("sampled edge set outside endpoint sandwich")
        if not len(graph.guaranteed_pairs) <= count <= possible_count:
            raise AssertionError("sampled edge count outside exact bounds")
        if not (graph.guaranteed_summary["largest_component_fraction"] <= sample_lcc[-1]
                <= graph.possible_summary["largest_component_fraction"]):
            raise AssertionError("sampled LCC outside exact bounds")
    return {
        "n_nodes": len(radii), "width": width, "threshold_nm": threshold,
        "draws": draws, "seed": seed,
        "guaranteed_edges": len(graph.guaranteed_pairs),
        "possible_edges": possible_count,
        "guaranteed_lcc": graph.guaranteed_summary["largest_component_fraction"],
        "possible_lcc": graph.possible_summary["largest_component_fraction"],
        "scalar_cutoff_nm": cutoff,
        "scalar_edges": int(np.sum(scalar)),
        "scalar_edge_disagreement": scalar_disagreement,
        "scalar_edge_disagreement_fraction_of_possible": scalar_disagreement / possible_count if possible_count else 0.0,
        "scalar_lcc": scalar_summary["largest_component_fraction"],
        "sample_edge_min": min(sample_edges), "sample_edge_max": max(sample_edges),
        "sample_lcc_min": min(sample_lcc), "sample_lcc_max": max(sample_lcc),
        "sample_lcc_mean": float(np.mean(sample_lcc)),
        "sample_components_min": min(sample_components),
        "sample_components_max": max(sample_components),
    }


def analyze(input_root: Path, draws: int = 64) -> dict:
    with (ROOT / "data/provenance/RUN_PROVENANCE.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    records = []
    for run_index, row in enumerate(rows):
        rel = PurePosixPath(row["relative_path"])
        if not rel.parts or rel.parts[0] != "builds" or ".." in rel.parts:
            raise ValueError("unsafe provenance path")
        source = input_root.joinpath(*rel.parts)
        if source.stat().st_size != int(row["bytes"]) or sha256(source) != row["sha256"]:
            raise ValueError("source checksum mismatch")
        positions, radii, types, frame = read_frame(source)
        record = compare_frame(positions, radii, types, seed=20261117 + run_index, draws=draws)
        record.update(run_id=row["run_id"], source_sha256=row["sha256"], frame=frame)
        records.append(record)
        print(f"BASELINE_OK {run_index+1}/{len(rows)} {row['run_id']}", flush=True)
    return {"schema": "radius-baselines-1", "radius_sampling_law":
            "independent uniform draws within each node interval; illustrative only",
            "frame_selection": "middle saved frame of each provenance-listed run",
            "source_hashes_checked": True, "records": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--output", type=Path, default=ROOT / "data/derived/radius_baseline_comparison.json")
    args = parser.parse_args()
    if args.draws < 1:
        parser.error("draws must be positive")
    result = analyze(args.input_root, args.draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
