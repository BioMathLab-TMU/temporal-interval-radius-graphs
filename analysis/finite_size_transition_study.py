#!/usr/bin/env python3
"""Holdout finite-size disk-graph study at fixed density and fixed thresholds.

The three thresholds were chosen from an earlier three-seed pilot and are not
reselected for each holdout realization. This is a finite-size sensitivity
experiment, not a proof of an infinite-volume percolation threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from correlated_radius_continuum import analyse_geometry, frame_geometry
from exact_endpoint_edge_moments import exact_edge_moments

TARGET_COUNTS = (325, 650, 1300, 2600)
THRESHOLDS = (("sparse", 0.4), ("transition", 0.8), ("dense", 1.2))
HOLDOUT_SEEDS = tuple(range(4401, 4411))


def run(draws: int, width: float) -> dict:
    records = []
    for target_n in TARGET_COUNTS:
        box_side = 60.0 * np.sqrt(target_n / 650.0)
        for seed in HOLDOUT_SEEDS:
            rng = np.random.default_rng(seed + 10000 * target_n)
            n = int(rng.poisson(target_n))
            positions = rng.uniform(0.0, box_side, size=(n, 2))
            radii = np.ones(n, dtype=np.float64)
            varied = np.ones(n, dtype=bool)
            for stage, threshold in THRESHOLDS:
                nominal_geometry = frame_geometry(positions, radii, varied, 0.0, threshold)
                geometry = frame_geometry(positions, radii, varied, width, threshold)
                draw_seed = seed + 10000 * target_n + int(threshold * 100)
                sampled = analyse_geometry(geometry, seed=draw_seed, draws=draws, width=width)
                # All draws are reproducible from the geometry seed, draw seed,
                # and code. Keep summaries here to avoid redundant large JSON.
                for entry in sampled["rho_results"].values():
                    del entry["edge_count_draws"]
                    del entry["largest_component_fraction_draws"]
                exact = exact_edge_moments(geometry, width)
                records.append({
                    "target_n": target_n, "actual_n": n,
                    "box_side": float(box_side), "point_seed": seed,
                    "stage": stage, "threshold": threshold,
                    "nominal_lcc": nominal_geometry["lower_lcc"],
                    "sampled": sampled, "exact_edge_moments": exact,
                })
                print(f"HOLDOUT_OK target_n={target_n} seed={seed} stage={stage}", flush=True)
    return {
        "schema": "finite-size-correlation-holdout-1",
        "status": "exploratory; fixed-density, fixed-threshold holdout",
        "pilot_seeds_not_reused": [3301, 3302, 3303],
        "holdout_seeds": list(HOLDOUT_SEEDS),
        "nominal_density": 650.0 / (60.0 ** 2),
        "target_counts": list(TARGET_COUNTS),
        "thresholds_from_pilot": {stage: tau for stage, tau in THRESHOLDS},
        "boundary": "finite square; no periodic wrap",
        "width": width, "draws_per_case_per_rho": draws,
        "rho_grid": [float(x) for x in sampled["rho_results"].keys()],
        "draw_level_arrays_omitted": True,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=512)
    parser.add_argument("--width", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.draws < 2 or not 0.0 < args.width < 1.0:
        parser.error("draws must exceed one and width must be in (0,1)")
    result = run(args.draws, args.width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n",
                           encoding="utf-8", newline="\n")
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
