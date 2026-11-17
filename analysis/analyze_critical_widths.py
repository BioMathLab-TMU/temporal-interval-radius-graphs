#!/usr/bin/env python3
"""Verify exact width events on one archived frame per particle count."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from critical_width_sweep import ambiguity_quotient, prepare_width_sweep  # noqa: E402
from interval_radius_graph import build_interval_radius_graph  # noqa: E402
from benchmark_pair_search import read_frame, sha256  # noqa: E402

WIDTHS = [0.0, 0.025, 0.05, 0.10]


def analyze(input_root: Path, all_runs: bool = False) -> dict:
    with (ROOT / "data/provenance/RUN_PROVENANCE.csv").open(encoding="utf-8-sig", newline="") as stream:
        sources = list(csv.DictReader(stream))
    chosen = {}
    for row in sources:
        chosen.setdefault(int(row["particle_count"]), row)
    selected = sources if all_runs else [chosen[n] for n in sorted(chosen)]
    records = []
    for row in selected:
        n = int(row["particle_count"])
        rel = PurePosixPath(row["relative_path"])
        if rel.parts[0] != "builds" or ".." in rel.parts:
            raise ValueError("unsafe provenance path")
        path = input_root.joinpath(*rel.parts)
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            raise ValueError(f"source checksum mismatch: {row['run_id']}")
        positions, radii, types, frame = read_frame(path)
        varied = types == 1
        sweep = prepare_width_sweep(positions, radii, varied, 2.0, 0.10)
        endpoints = sweep.evaluate(WIDTHS)
        for width, item in zip(WIDTHS, endpoints):
            graph = build_interval_radius_graph(
                positions, radii * (1 - width * varied),
                radii * (1 + width * varied), 2.0,
            )
            for name in ("guaranteed", "possible"):
                observed = getattr(graph, f"{name}_summary")
                predicted = item[name]
                if (predicted["edges"] != len(getattr(graph, f"{name}_pairs"))
                    or predicted["components"] != observed["n_components"]
                    or predicted["largest_component"] != observed["largest_component_size"]
                    or predicted["isolated_nodes"] != int(np.sum(observed["degree"] == 0))):
                    raise AssertionError(f"endpoint mismatch: {row['run_id']} {frame} {width} {name}")
        graph = build_interval_radius_graph(
            positions, radii * (1 - .1 * varied), radii * (1 + .1 * varied), 2.0,
        )
        quotient = ambiguity_quotient(n, graph.guaranteed_pairs, graph.possible_pairs)
        if quotient["possible_components"] != graph.possible_summary["n_components"]:
            raise AssertionError("quotient partition mismatch")
        transitions = sweep.component_transition_widths()
        if len(transitions["possible_component_mergers"]) != endpoints[0]["possible"]["components"] - endpoints[-1]["possible"]["components"]:
            raise AssertionError("possible-graph merger count mismatch")
        if len(transitions["guaranteed_component_splits"]) != endpoints[-1]["guaranteed"]["components"] - endpoints[0]["guaranteed"]["components"]:
            raise AssertionError("guaranteed-graph split count mismatch")
        records.append({
            "run_id": row["run_id"], "source_sha256": row["sha256"],
            "frame": frame, "n_nodes": n, "candidate_pairs": len(sweep.candidate_pairs),
            "possible_edge_birth_events": int(np.isfinite(sweep.possible_births).sum()),
            "guaranteed_edge_exit_events": int(np.isfinite(sweep.guaranteed_exits).sum()),
            "critical_width_count": len(sweep.critical_widths()),
            "possible_component_merger_events": len(transitions["possible_component_mergers"]),
            "guaranteed_component_split_events": len(transitions["guaranteed_component_splits"]),
            "endpoints": endpoints, "quotient_at_0p1": quotient,
        })
        print(f"WIDTH_SWEEP_OK n={n} run={row['run_id']} frame={frame}")
    return {"schema": "critical-width-analysis-1", "widths": WIDTHS,
            "threshold_nm": 2.0, "frame_selection": "middle saved frame of every manifest run" if all_runs else "middle frame of first manifest run per particle count",
            "source_hashes_checked": True, "independent_endpoint_checks": True, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--all-runs", action="store_true", help="analyze all 15 provenance-listed trajectories")
    parser.add_argument("--output", type=Path, default=ROOT / "data/derived/critical_widths.json")
    args = parser.parse_args()
    result = analyze(args.input_root.resolve(), all_runs=args.all_runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
