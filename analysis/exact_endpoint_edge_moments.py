#!/usr/bin/env python3
"""Exact first two edge-count moments at independent and shared radius laws.

The input is one fixed interval-radius geometry. All varied radius factors are
uniform on [-1, 1]. The independent law uses one factor per node; the shared
law uses one factor for all varied nodes. Piecewise-linear conditional edge
probabilities and a threshold sweep replace Monte Carlo at these two limits.
Floating-point arithmetic is used for breakpoints and quadrature.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from benchmark_pair_search import read_frame, sha256  # noqa: E402
from correlated_radius_continuum import frame_geometry  # noqa: E402


def conditional_probability(u: np.ndarray, b_center: float,
                            b_other: float, threshold: float) -> np.ndarray:
    """P(edge | U_center=u) for one independent uniform neighbour factor."""
    if b_center <= 0 or b_other < 0:
        raise ValueError("invalid radius half-width")
    if b_other == 0:
        return np.asarray(b_center * u >= threshold, dtype=np.float64)
    return np.clip((b_center * u + b_other - threshold) / (2.0 * b_other), 0.0, 1.0)


def _breakpoints(b_center: float, b_other: float, threshold: float) -> list[float]:
    if b_other == 0:
        raw = [threshold / b_center]
    else:
        raw = [(threshold - b_other) / b_center,
               (threshold + b_other) / b_center]
    return [float(x) for x in raw if -1.0 < x < 1.0]


def uniform_integral_of_product(first: tuple[float, float, float],
                                second: tuple[float, float, float] | None = None) -> float:
    """Integrate one or two conditional probabilities over U~Uniform[-1,1].

    On each interval between all breakpoints, every factor is constant or
    linear. Two-point Gauss-Legendre quadrature integrates their product,
    a polynomial of degree at most two, exactly up to floating-point error.
    """
    specs = [first] if second is None else [first, second]
    cuts = sorted(set([-1.0, 1.0, *[x for b, c, t in specs
                                     for x in _breakpoints(b, c, t)]]))
    total = 0.0
    offset = 1.0 / np.sqrt(3.0)
    for left, right in zip(cuts[:-1], cuts[1:]):
        if right <= left:
            continue
        midpoint = (left + right) / 2.0
        half = (right - left) / 2.0
        values = []
        for abscissa in (midpoint - offset * half, midpoint + offset * half):
            product = 1.0
            for b_center, b_other, threshold in specs:
                product *= float(conditional_probability(np.asarray(abscissa),
                                                         b_center, b_other, threshold))
            values.append(product)
        total += half * (values[0] + values[1]) / 2.0
    return float(np.clip(total, 0.0, 1.0))


def exact_edge_moments(geometry: dict, width: float) -> dict:
    """Return exact edge-count means and variances conditional on geometry."""
    nodes = geometry["nominal"]
    b = width * nodes * geometry["active_varied"]
    ui, uj = geometry["ui"], geometry["uj"]
    thresholds = geometry["required_sums"] - nodes[ui] - nodes[uj]
    m = len(ui)
    incident: list[list[int]] = [[] for _ in range(len(nodes))]
    probs = np.empty(m, dtype=np.float64)
    for edge in range(m):
        i, j = int(ui[edge]), int(uj[edge])
        if b[i] > 0:
            probs[edge] = uniform_integral_of_product((b[i], b[j], thresholds[edge]))
            incident[i].append(edge)
            if b[j] > 0:
                incident[j].append(edge)
        elif b[j] > 0:
            probs[edge] = uniform_integral_of_product((b[j], b[i], thresholds[edge]))
            incident[j].append(edge)
        else:
            raise AssertionError("an uncertain edge must have a varying endpoint")
    independent_variance = float(np.sum(probs * (1.0 - probs)))
    covariance_sum = 0.0
    covariance_pairs = 0
    for node, edges in enumerate(incident):
        if len(edges) < 2:
            continue
        for a in range(len(edges)):
            e = edges[a]
            other_e = int(uj[e] if int(ui[e]) == node else ui[e])
            spec_e = (float(b[node]), float(b[other_e]), float(thresholds[e]))
            for f in edges[a + 1:]:
                other_f = int(uj[f] if int(ui[f]) == node else ui[f])
                spec_f = (float(b[node]), float(b[other_f]), float(thresholds[f]))
                joint = uniform_integral_of_product(spec_e, spec_f)
                covariance_sum += joint - probs[e] * probs[f]
                covariance_pairs += 1
    independent_variance += 2.0 * covariance_sum
    if independent_variance < -1e-8:
        raise AssertionError("negative independent variance")
    independent_variance = max(0.0, independent_variance)
    independent_mean = geometry["base_edges"] + float(np.sum(probs))

    common_thresholds = thresholds / (b[ui] + b[uj])
    common_thresholds = np.sort(np.clip(common_thresholds, -1.0, 1.0))
    interval_lengths = np.diff(np.r_[-1.0, common_thresholds, 1.0])
    levels = geometry["base_edges"] + np.arange(m + 1, dtype=np.float64)
    common_mean = float(np.dot(interval_lengths, levels) / 2.0)
    common_variance = float(np.dot(interval_lengths, (levels - common_mean) ** 2) / 2.0)
    return {
        "uncertain_edges": m,
        "independent_uniform": {
            "mean_edges": independent_mean,
            "variance_edges": independent_variance,
            "sd_edges": float(np.sqrt(independent_variance)),
            "covariance_pairs": covariance_pairs,
        },
        "common_uniform": {
            "mean_edges": common_mean,
            "variance_edges": common_variance,
            "sd_edges": float(np.sqrt(common_variance)),
        },
    }


def analyse_archive(input_root: Path, width: float) -> list[dict]:
    with (ROOT / "data/provenance/RUN_PROVENANCE.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    records = []
    for index, row in enumerate(rows):
        rel = PurePosixPath(row["relative_path"])
        if not rel.parts or rel.parts[0] != "builds" or ".." in rel.parts:
            raise ValueError("unsafe provenance path")
        source = input_root.joinpath(*rel.parts)
        if source.stat().st_size != int(row["bytes"]) or sha256(source) != row["sha256"]:
            raise ValueError("source checksum mismatch")
        positions, radii, types, frame = read_frame(source)
        geometry = frame_geometry(positions, radii, types == 1, width, 2.0)
        record = exact_edge_moments(geometry, width)
        record.update(run_id=row["run_id"], source_sha256=row["sha256"], frame=frame,
                      n_nodes=geometry["n_nodes"], n_varied_nodes=geometry["n_varied"])
        records.append(record)
        print(f"EXACT_MOMENTS_ARCHIVE_OK {index + 1}/{len(rows)}", flush=True)
    return records


def analyse_synthetic(width: float) -> list[dict]:
    records = []
    for seed in (3301, 3302, 3303):
        rng = np.random.default_rng(seed)
        n = int(rng.poisson(650))
        positions = rng.uniform(0.0, 60.0, size=(n, 2))
        radii = np.ones(n)
        candidates = []
        for tau in np.arange(-0.8, 2.01, 0.2):
            geometry = frame_geometry(positions, radii, np.ones(n, dtype=bool), 0.0, float(tau))
            candidates.append((float(tau), geometry["lower_lcc"]))
        for stage, target in (("sparse", 0.1), ("transition", 0.5), ("dense", 0.9)):
            tau, nominal_lcc = min(candidates, key=lambda x: abs(x[1] - target))
            geometry = frame_geometry(positions, radii, np.ones(n, dtype=bool), width, tau)
            record = exact_edge_moments(geometry, width)
            record.update(synthetic_seed=seed, finite_size_stage=stage,
                          threshold=tau, nominal_lcc=nominal_lcc, n_nodes=n)
            records.append(record)
            print(f"EXACT_MOMENTS_SYNTHETIC_OK {seed} {stage}", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--width", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 < args.width < 1.0:
        parser.error("width must be in (0, 1)")
    report = {
        "schema": "exact-endpoint-edge-moments-1",
        "width": args.width,
        "method": "exact piecewise-polynomial integration at independence; exact threshold sweep at full common factor",
        "source_hashes_checked": True,
        "archive": analyse_archive(args.input_root, args.width),
        "synthetic": analyse_synthetic(args.width),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
