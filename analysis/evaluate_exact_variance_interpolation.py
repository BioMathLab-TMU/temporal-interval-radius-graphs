#!/usr/bin/env python3
"""Compare intermediate-correlated edge-count SD with exact endpoint moments.

Bootstrap intervals quantify finite Monte Carlo error conditional on each
fixed geometry. Endpoint moments contain no sampling approximation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def evaluate(moment_report: dict, sample_report: dict, *, repeats: int,
             rng: np.random.Generator) -> dict:
    if moment_report["schema"] != "exact-endpoint-edge-moments-1":
        raise ValueError("unexpected exact-moment schema")
    if sample_report["schema"] != "correlated-radius-continuum-1":
        raise ValueError("unexpected sampling schema")
    if moment_report["width"] != sample_report["width"]:
        raise ValueError("width mismatch")
    result = {"width": sample_report["width"], "groups": {}}
    for group in ("archive", "synthetic"):
        if len(moment_report[group]) != len(sample_report[group]):
            raise ValueError("case count mismatch")
        cases = []
        for exact, sampled in zip(moment_report[group], sample_report[group]):
            if group == "archive":
                if (exact["run_id"] != sampled["run_id"] or
                        exact["source_sha256"] != sampled["source_sha256"]):
                    raise ValueError("archived source mismatch")
            elif (exact["synthetic_seed"] != sampled["synthetic_seed"] or
                  exact["finite_size_stage"] != sampled["finite_size_stage"] or
                  exact["threshold"] != sampled["threshold"]):
                raise ValueError("synthetic case mismatch")
            v0 = exact["independent_uniform"]["variance_edges"]
            v1 = exact["common_uniform"]["variance_edges"]
            rows = []
            for rho in sample_report["rho_grid"]:
                samples = np.asarray(
                    sampled["rho_results"][str(rho)]["edge_count_draws"], dtype=float)
                predicted = float(np.sqrt((1.0 - rho) * v0 + rho * v1))
                if predicted == 0.0:
                    continue
                observed = float(np.std(samples, ddof=1) / predicted)
                index = rng.integers(0, len(samples), size=(repeats, len(samples)))
                bootstrap = np.std(samples[index], axis=1, ddof=1) / predicted
                lower, upper = np.quantile(bootstrap, (0.025, 0.975))
                rows.append({
                    "rho": rho,
                    "predicted_sd_from_exact_endpoints": predicted,
                    "observed_sd_over_prediction": observed,
                    "mc_bootstrap_q025": float(lower),
                    "mc_bootstrap_q975": float(upper),
                    "unit_ratio_outside_interval": bool(lower > 1.0 or upper < 1.0),
                })
            cases.append({
                "run_id": sampled.get("run_id"),
                "synthetic_seed": sampled.get("synthetic_seed"),
                "finite_size_stage": sampled.get("finite_size_stage"),
                "rows": rows,
                "exact_independent_mean": exact["independent_uniform"]["mean_edges"],
                "exact_independent_sd": exact["independent_uniform"]["sd_edges"],
                "exact_common_mean": exact["common_uniform"]["mean_edges"],
                "exact_common_sd": exact["common_uniform"]["sd_edges"],
            })
        result["groups"][group] = cases
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, nargs=2, required=True)
    parser.add_argument("--moments", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261119)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    reports = []
    for sample_path, moment_path in zip(args.samples, args.moments):
        samples = json.loads(sample_path.read_text(encoding="utf-8"))
        moments = json.loads(moment_path.read_text(encoding="utf-8"))
        reports.append(evaluate(moments, samples, repeats=args.repeats, rng=rng))
    output = {
        "schema": "exact-endpoint-variance-interpolation-test-1",
        "seed": args.seed, "bootstrap_repeats": args.repeats,
        "model_status": "hypothesis; no intermediate-correlation fitted parameters",
        "mc_interval_scope": "finite Monte Carlo draws conditional on each fixed geometry",
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, separators=(",", ":")) + "\n",
                           encoding="utf-8", newline="\n")
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
