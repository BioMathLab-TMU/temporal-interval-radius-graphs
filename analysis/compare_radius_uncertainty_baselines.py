#!/usr/bin/env python3
"""Compare exact interval bounds with sampling laws and cutoff baselines.

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


def _quotient_summary(component_labels, component_sizes, cross_pairs, active, n_nodes):
    """Compute component observables after adding uncertain quotient edges."""
    parent = np.arange(len(component_sizes), dtype=np.int64)
    weights = component_sizes.copy()
    component_count = len(parent)
    largest = int(weights.max())

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for left, right in cross_pairs[active]:
        a, b = find(int(component_labels[left])), find(int(component_labels[right]))
        if a != b:
            if weights[a] < weights[b]:
                a, b = b, a
            parent[b] = a
            weights[a] += weights[b]
            largest = max(largest, int(weights[a]))
            component_count -= 1
    return largest / n_nodes, component_count


def _sampling_summary(lcc, components, edges, exact_span, lower_lcc, upper_lcc):
    span = max(lcc) - min(lcc)
    return {
        "edge_min": min(edges), "edge_max": max(edges),
        "lcc_min": min(lcc), "lcc_max": max(lcc),
        "lcc_mean": float(np.mean(lcc)), "lcc_span": span,
        "lcc_span_fraction_of_exact": span / exact_span if exact_span > 0 else None,
        "lower_lcc_attained": bool(np.isclose(min(lcc), lower_lcc, rtol=0, atol=1e-14)),
        "upper_lcc_attained": bool(np.isclose(max(lcc), upper_lcc, rtol=0, atol=1e-14)),
        "components_min": min(components), "components_max": max(components),
    }


def compare_frame(positions, radii, types, *, seed: int, draws: int = 1024,
                  width: float = 0.10, threshold: float = 2.0) -> dict:
    if draws < 64:
        raise ValueError("draws must be at least 64 for nested 64-versus-N comparison")
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

    # A type-pair cutoff is an exact comparator when the radius contribution
    # q_ij is constant for each observed pair of node types.
    q = varied[i] * radii[i] + varied[j] * radii[j]
    type_cutoffs = {}
    type_possible = np.zeros(len(pairs), dtype=bool)
    type_guaranteed = np.zeros(len(pairs), dtype=bool)
    type_constant = True
    for left_type in np.unique(types):
        for right_type in np.unique(types):
            if right_type < left_type:
                continue
            group = ((types[i] == left_type) & (types[j] == right_type)) | ((types[i] == right_type) & (types[j] == left_type))
            if not np.any(group):
                continue
            group_q = q[group]
            constant = bool(np.allclose(group_q, group_q[0], rtol=0, atol=1e-12))
            type_constant &= constant
            type_cutoffs[f"{int(left_type)}-{int(right_type)}"] = {
                "q_nm": float(group_q[0]) if constant else None,
                "constant_q": constant,
            }
            if constant:
                type_possible[group] = nominal_gap[group] <= threshold + width * group_q[0]
                type_guaranteed[group] = nominal_gap[group] <= threshold - width * group_q[0]
    if type_constant and (not np.array_equal(type_possible, possible) or not np.array_equal(type_guaranteed, guaranteed)):
        raise AssertionError("type-pair cutoffs should reproduce both endpoint edge sets")

    # Only uncertain pairs can change across draws. Contract guaranteed
    # components and update connectivity using the uncertain cross edges.
    uncertain = possible & ~guaranteed
    uncertain_pairs = pairs[uncertain]
    ui, uj = uncertain_pairs[:, 0], uncertain_pairs[:, 1]
    required_sums = distances[uncertain] - threshold
    base_count = int(np.count_nonzero(guaranteed))
    component_labels = graph.guaranteed_summary["labels"]
    component_sizes = graph.guaranteed_summary["component_sizes"]
    cross = component_labels[ui] != component_labels[uj]
    cross_pairs = uncertain_pairs[cross]
    lower_lcc = graph.guaranteed_summary["largest_component_fraction"]
    upper_lcc = graph.possible_summary["largest_component_fraction"]
    exact_span = upper_lcc - lower_lcc
    sampling = {}
    for law, law_seed in (("independent_uniform", seed), ("common_uniform", seed + 1000000)):
        rng = np.random.default_rng(law_seed)
        lcc, components, edges = [], [], []
        summaries = {}
        for draw_index in range(1, draws + 1):
            if law == "independent_uniform":
                sampled = rng.uniform(lower, upper)
            else:
                z = rng.uniform(-1.0, 1.0)
                sampled = radii * (1.0 + width * varied * z)
            active = sampled[ui] + sampled[uj] >= required_sums
            edge_count = base_count + int(np.count_nonzero(active))
            largest, n_components = _quotient_summary(component_labels, component_sizes,
                                                       cross_pairs, active[cross], len(radii))
            if not (base_count <= edge_count <= possible_count and lower_lcc - 1e-14 <= largest <= upper_lcc + 1e-14):
                raise AssertionError("sampled observable outside exact endpoint interval")
            lcc.append(largest)
            components.append(n_components)
            edges.append(edge_count)
            if draw_index in (64, draws):
                summaries[str(draw_index)] = _sampling_summary(lcc, components, edges,
                                                               exact_span, lower_lcc, upper_lcc)
        sampling[law] = {"seed": law_seed, "summaries": summaries}
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
        "type_pair_cutoffs": type_cutoffs,
        "type_pair_q_constant": bool(type_constant),
        "type_pair_possible_edges_equal": bool(type_constant and np.array_equal(type_possible, possible)),
        "type_pair_guaranteed_edges_equal": bool(type_constant and np.array_equal(type_guaranteed, guaranteed)),
        "sampling": sampling,
    }


def analyze(input_root: Path, draws: int = 1024) -> dict:
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
    return {"schema": "radius-baselines-2", "radius_sampling_laws": [
                "independent uniform per varied node",
                "one common uniform scale for all varied nodes"],
            "sampling_prefixes": [64, draws],
            "frame_selection": "middle saved frame of each provenance-listed run",
            "source_hashes_checked": True, "records": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=ROOT / "data/derived/radius_baseline_comparison.json")
    args = parser.parse_args()
    if args.draws < 64:
        parser.error("draws must be at least 64")
    result = analyze(args.input_root, args.draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
