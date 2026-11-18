#!/usr/bin/env python3
"""Conditional Monte Carlo uncertainty for the variance interpolation.

Resamples graph draws jointly across rho, preserving the shared random-number
design. This assesses finite sampling error conditional on each fixed spatial
configuration; it is not a biological or spatial-replicate confidence interval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bootstrap_record(record: dict, rho_grid: list[float], *, rng, repeats: int) -> list[dict]:
    draws = int(record["draws"])
    index = rng.integers(0, draws, size=(repeats, draws))
    rows = []
    for observable, draw_key in (("edge_count", "edge_count_draws"),
                                 ("largest_component_fraction", "largest_component_fraction_draws")):
        sds = {}
        for rho in rho_grid:
            values = np.asarray(record["rho_results"][str(rho)][draw_key], dtype=np.float64)
            sds[rho] = np.std(values[index], axis=1, ddof=1)
        for rho in rho_grid[1:-1]:
            predicted = np.sqrt((1.0 - rho) * sds[0.0] ** 2 + rho * sds[1.0] ** 2)
            valid = predicted > 0.0
            if not np.all(valid):
                continue
            ratio = sds[rho] / predicted
            observed = record["rho_results"][str(rho)][observable]["observed_over_linearized_prediction"]
            rows.append({
                "rho": rho, "observable": observable,
                "observed_ratio": observed,
                "mc_bootstrap_q025": float(np.quantile(ratio, 0.025)),
                "mc_bootstrap_q975": float(np.quantile(ratio, 0.975)),
                "unit_ratio_outside_interval": bool(
                    np.quantile(ratio, 0.025) > 1.0 or np.quantile(ratio, 0.975) < 1.0),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261118)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    reports = []
    for source in args.sources:
        data = json.loads(source.read_text(encoding="utf-8"))
        if data["schema"] != "correlated-radius-continuum-1":
            raise ValueError(f"unexpected source schema: {source}")
        cases = []
        for group in ("archive", "synthetic"):
            for index, record in enumerate(data[group]):
                cases.append({
                    "group": group, "index": index,
                    "run_id": record.get("run_id"),
                    "synthetic_seed": record.get("synthetic_seed"),
                    "finite_size_stage": record.get("finite_size_stage"),
                    "rows": bootstrap_record(record, data["rho_grid"], rng=rng, repeats=args.repeats),
                })
        reports.append({"source": source.name, "width": data["width"],
                        "draws": data["archive"][0]["draws"], "cases": cases})
    result = {
        "schema": "correlated-radius-mc-bootstrap-1",
        "seed": args.seed, "repeats": args.repeats,
        "interpretation": "conditional Monte Carlo uncertainty only; draw indices jointly resampled across correlations",
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n",
                           encoding="utf-8", newline="\n")
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
