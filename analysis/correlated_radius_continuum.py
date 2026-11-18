#!/usr/bin/env python3
"""Explore cross-node radius correlation without leaving declared intervals.

This is an exploratory, falsifiable analysis, not a fitted physical radius law.
The Gaussian copula makes every radius factor uniform on [-1, 1] and sets its
pairwise Pearson correlation to ``rho``. The two endpoints reproduce the
independent- and common-uniform laws used in the manuscript.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path, PurePosixPath

import numpy as np
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from interval_radius_graph import build_interval_radius_graph, _component_summary  # noqa: E402
from benchmark_pair_search import read_frame, sha256  # noqa: E402
from compare_radius_uncertainty_baselines import _quotient_summary  # noqa: E402

RHO_GRID = (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)


def uniform_factors(epsilon: np.ndarray, shared: np.ndarray, rho: float) -> np.ndarray:
    """Uniform marginals with exact pairwise Pearson correlation ``rho``.

    For Gaussian-copula latent correlation c, uniform-factor correlation is
    6/pi * arcsin(c/2); inversion gives c=2*sin(pi*rho/6).
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    # Exact endpoints avoid a tiny spurious independent term from rounding
    # 2*sin(pi/6) just below one in floating-point arithmetic.
    c = 0.0 if rho == 0.0 else (1.0 if rho == 1.0 else 2.0 * np.sin(np.pi * rho / 6.0))
    latent = np.sqrt(c) * shared[:, None] + np.sqrt(1.0 - c) * epsilon
    return 2.0 * ndtr(latent) - 1.0


def frame_geometry(positions, radii, varied, width: float, threshold: float) -> dict:
    lower = radii * (1.0 - width * varied)
    upper = radii * (1.0 + width * varied)
    graph = build_interval_radius_graph(positions, lower, upper, threshold)
    pairs = graph.candidate_pairs
    i, j = pairs[:, 0], pairs[:, 1]
    distances = np.linalg.norm(positions[i] - positions[j], axis=1)
    guaranteed = distances <= lower[i] + lower[j] + threshold
    possible = distances <= upper[i] + upper[j] + threshold
    uncertain_pairs = pairs[possible & ~guaranteed]
    ui, uj = uncertain_pairs[:, 0], uncertain_pairs[:, 1]
    active_nodes = np.unique(uncertain_pairs.reshape(-1))
    local_index = np.full(len(radii), -1, dtype=np.int64)
    local_index[active_nodes] = np.arange(len(active_nodes))
    labels = graph.guaranteed_summary["labels"]
    cross = labels[ui] != labels[uj]
    uncertain_degree = np.bincount(uncertain_pairs.reshape(-1), minlength=len(radii))
    # Each independent variable can change at most its incident uncertain
    # edges. This supplies a valid bounded-differences constant for edge count.
    mcdiarmid_sum_squares = int(np.sum(uncertain_degree[varied].astype(np.int64) ** 2))
    incidence_sum = int(np.sum(uncertain_degree[varied]))
    degree_effective_count = (incidence_sum ** 2 / mcdiarmid_sum_squares
                              if mcdiarmid_sum_squares else None)
    return {
        "nominal": radii[active_nodes],
        "active_varied": varied[active_nodes],
        "ui": local_index[ui], "uj": local_index[uj],
        "required_sums": distances[possible & ~guaranteed] - threshold,
        "cross_pairs": uncertain_pairs[cross], "cross": cross,
        "labels": labels, "sizes": graph.guaranteed_summary["component_sizes"],
        "n_nodes": len(radii), "n_varied": int(np.count_nonzero(varied)),
        "n_active": int(np.count_nonzero(varied[active_nodes])),
        "base_edges": int(np.count_nonzero(guaranteed)),
        "upper_edges": int(np.count_nonzero(possible)),
        "lower_lcc": graph.guaranteed_summary["largest_component_fraction"],
        "upper_lcc": graph.possible_summary["largest_component_fraction"],
        "mcdiarmid_sum_squares": mcdiarmid_sum_squares,
        "ambiguous_incidence_sum": incidence_sum,
        "degree_effective_count": degree_effective_count,
    }


def summarize(values: np.ndarray, exact_span: float) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "min": float(values.min()), "max": float(values.max()),
        "exact_span": float(exact_span),
        "sd_over_exact_span": float(values.std(ddof=1) / exact_span) if exact_span else None,
        "sample_span_over_exact_span": float(np.ptp(values) / exact_span) if exact_span else None,
    }


def analyse_geometry(geometry: dict, *, seed: int, draws: int,
                     width: float, rho_grid=RHO_GRID, batch_size: int = 64) -> dict:
    if draws < 2:
        raise ValueError("at least two draws required for sample SD")
    # Fixed-radius nodes can occur in uncertain edges but have no random
    # factor. Their nominal radius remains unchanged in every draw.
    rng = np.random.default_rng(seed)
    # Shared random numbers across rho reduce comparison noise. Process in
    # batches so the largest frame does not retain a draws-by-nodes matrix.
    edge_values = {rho: np.empty(draws, dtype=np.int64) for rho in rho_grid}
    lcc_values = {rho: np.empty(draws, dtype=np.float64) for rho in rho_grid}
    for start in range(0, draws, batch_size):
        stop = min(start + batch_size, draws)
        epsilon = rng.standard_normal((stop - start, len(geometry["nominal"])))
        shared = rng.standard_normal(stop - start)
        for rho in rho_grid:
            factor = uniform_factors(epsilon, shared, rho)
            sampled = geometry["nominal"][None, :] * (
                1.0 + width * factor * geometry["active_varied"][None, :])
            active = sampled[:, geometry["ui"]] + sampled[:, geometry["uj"]] >= geometry["required_sums"]
            edge_values[rho][start:stop] = geometry["base_edges"] + active.sum(axis=1)
            for local_draw, flags in enumerate(active):
                lcc, _ = _quotient_summary(
                    geometry["labels"], geometry["sizes"], geometry["cross_pairs"],
                    flags[geometry["cross"]], geometry["n_nodes"])
                lcc_values[rho][start + local_draw] = lcc
    edge_span = geometry["upper_edges"] - geometry["base_edges"]
    lcc_span = geometry["upper_lcc"] - geometry["lower_lcc"]
    results = {}
    for rho in rho_grid:
        edges, lcc = edge_values[rho], lcc_values[rho]
        if np.any((edges < geometry["base_edges"]) | (edges > geometry["upper_edges"])):
            raise AssertionError("sampled edge count outside exact envelope")
        if np.any((lcc < geometry["lower_lcc"] - 1e-14) | (lcc > geometry["upper_lcc"] + 1e-14)):
            raise AssertionError("sampled component size outside exact envelope")
        results[str(rho)] = {
            "edge_count": summarize(edges, edge_span),
            "largest_component_fraction": summarize(lcc, lcc_span),
            "edge_count_draws": edges.tolist(),
            "largest_component_fraction_draws": lcc.tolist(),
        }
    for observable in ("edge_count", "largest_component_fraction"):
        v0 = results["0.0"][observable]["sd"] ** 2
        v1 = results["1.0"][observable]["sd"] ** 2
        for rho in rho_grid:
            observed = results[str(rho)][observable]["sd"]
            variance_interpolation = (1.0 - rho) * v0 + rho * v1
            predicted = np.sqrt(variance_interpolation)
            results[str(rho)][observable]["linearized_sd_prediction"] = float(predicted)
            results[str(rho)][observable]["observed_over_linearized_prediction"] = (
                float(observed / predicted) if predicted else None)
            if v1 > 0:
                naive = np.sqrt(v1) * np.sqrt(rho + (1.0 - rho) / geometry["n_varied"])
                results[str(rho)][observable]["naive_n_var_sd_prediction"] = float(naive)
                results[str(rho)][observable]["observed_over_naive_prediction"] = (
                    float(observed / naive) if naive else None)
    return {
        "seed": seed, "draws": draws, "rho_results": results,
        "n_nodes": geometry["n_nodes"], "n_varied_nodes": geometry["n_varied"],
        "n_active_varied_nodes": geometry["n_active"],
        "guaranteed_edges": geometry["base_edges"], "possible_edges": geometry["upper_edges"],
        "guaranteed_lcc": geometry["lower_lcc"], "possible_lcc": geometry["upper_lcc"],
        "mcdiarmid_sum_squares": geometry["mcdiarmid_sum_squares"],
        "ambiguous_incidence_sum": geometry["ambiguous_incidence_sum"],
        "degree_effective_count": geometry["degree_effective_count"],
    }


def archived_frames(input_root: Path, draws: int, width: float) -> list[dict]:
    with (ROOT / "data/provenance/RUN_PROVENANCE.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    records = []
    for index, row in enumerate(rows):
        rel = PurePosixPath(row["relative_path"])
        if not rel.parts or rel.parts[0] != "builds" or ".." in rel.parts:
            raise ValueError("unsafe provenance path")
        path = input_root.joinpath(*rel.parts)
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            raise ValueError("source checksum mismatch")
        positions, radii, types, frame = read_frame(path)
        geometry = frame_geometry(positions, radii, types == 1, width=width, threshold=2.0)
        result = analyse_geometry(geometry, seed=20261118 + index, draws=draws, width=width)
        result.update(run_id=row["run_id"], source_sha256=row["sha256"], frame=frame)
        records.append(result)
        print(f"CORRELATION_ARCHIVE_OK {index+1}/{len(rows)} {row['run_id']}", flush=True)
    return records


def synthetic_frames(draws: int, width: float) -> list[dict]:
    records = []
    for seed in (3301, 3302, 3303):
        rng = np.random.default_rng(seed)
        n = int(rng.poisson(650))
        positions = rng.uniform(0.0, 60.0, size=(n, 2))
        radii = np.ones(n, dtype=np.float64)
        nominal = []
        for tau in np.arange(-0.8, 2.01, 0.2):
            graph = frame_geometry(positions, radii, np.ones(n, dtype=bool), 0.0, float(tau))
            nominal.append((float(tau), graph["lower_lcc"]))
        selected = []
        for target in (0.10, 0.50, 0.90):
            selected.append(min(nominal, key=lambda x: abs(x[1] - target)))
        for stage, (tau, nominal_lcc) in zip(("sparse", "transition", "dense"), selected):
            geometry = frame_geometry(positions, radii, np.ones(n, dtype=bool), width, tau)
            result = analyse_geometry(geometry, seed=seed + int(round((tau + 10) * 100)),
                                      draws=draws, width=width)
            result.update(synthetic_seed=seed, point_count=n, box_side=60.0,
                          nominal_radius=1.0, radius_half_width=width,
                          threshold=tau, nominal_lcc=nominal_lcc,
                          finite_size_stage=stage,
                          threshold_selection="closest nominal LCC to 0.1, 0.5, or 0.9 on grid -0.8:0.2:2.0")
            records.append(result)
            print(f"CORRELATION_SYNTHETIC_OK seed={seed} stage={stage} tau={tau}", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1024)
    parser.add_argument("--width", type=float, default=0.10)
    parser.add_argument("--output", type=Path, default=ROOT / "data/derived/correlated_radius_continuum.json")
    args = parser.parse_args()
    if not 0.0 < args.width < 1.0:
        parser.error("width must be in (0, 1)")
    report = {
        "schema": "correlated-radius-continuum-1",
        "status": "exploratory hypothesis test; not a calibrated biological radius law",
        "rho_definition": "pairwise Pearson correlation of uniform [-1,1] node factors within one graph",
        "copula_latent_c": "2*sin(pi*rho/6)",
        "rho_grid": list(RHO_GRID), "width": args.width,
        "source_hashes_checked": True, "archive": archived_frames(args.input_root, args.draws, args.width),
        "synthetic": synthetic_frames(args.draws, args.width),
        "synthetic_model": "Poisson count; independent uniform centres in finite 60x60 square; equal nominal disk radii; no periodic boundary",
        "inference_limit": "size groups in archive differ in simulation conditions; finite synthetic thresholds are selected from the same realizations",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    print(f"CORRELATION_REPORT {args.output}", flush=True)


if __name__ == "__main__":
    main()
