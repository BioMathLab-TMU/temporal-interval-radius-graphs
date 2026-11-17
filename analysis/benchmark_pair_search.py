#!/usr/bin/env python3
"""Compare exact interval-radius pair searches on archived simulation frames.

This benchmark times only pair generation and endpoint edge classification.
Component summaries are deliberately excluded from both timings. The blocked
all-pairs reference uses bounded memory but still evaluates every pair. It is
not a stochastic approximation and serves as an independent correctness check.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path, PurePosixPath

import numpy as np
import scipy
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "provenance" / "RUN_PROVENANCE.csv"
SYNAPSIN_TYPE = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_frame(source: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    import h5py

    with h5py.File(source, "r") as handle:
        frames = sorted(
            (int(key[5:]), key)
            for key in handle.keys()
            if key.startswith("step_") and key[5:].isdigit()
        )
        if not frames:
            raise ValueError(f"No step groups in {source}")
        _, name = frames[len(frames) // 2]
        group = handle[name]
        if "frame_write_complete" in group and int(np.asarray(group["frame_write_complete"][...]).reshape(-1)[0]) != 1:
            raise ValueError(f"Incomplete benchmark frame: {source}:{name}")
        return (
            np.asarray(group["positions"][...], dtype=np.float64),
            np.asarray(group["radii"][...], dtype=np.float64),
            np.asarray(group["types"][...]),
            name,
        )


def radius_bounds(
    nominal: np.ndarray, node_types: np.ndarray, half_width: float
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(half_width) or not 0 <= half_width < 1:
        raise ValueError("half_width must be in [0, 1)")
    lower = nominal.copy()
    upper = nominal.copy()
    selected = node_types == SYNAPSIN_TYPE
    lower[selected] *= 1 - half_width
    upper[selected] *= 1 + half_width
    return lower, upper


def _lexsorted(pairs: np.ndarray) -> np.ndarray:
    if not len(pairs):
        return np.empty((0, 2), dtype=np.int64)
    return pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]


def tree_search(
    positions: np.ndarray, lower: np.ndarray, upper: np.ndarray, threshold: float
) -> tuple[int, np.ndarray, np.ndarray]:
    reach = 2 * float(np.max(upper)) + threshold
    if reach < 0:
        empty = np.empty((0, 2), dtype=np.int64)
        return 0, empty, empty
    candidates = np.asarray(
        cKDTree(positions).query_pairs(reach, output_type="ndarray"), dtype=np.int64
    ).reshape(-1, 2)
    if not len(candidates):
        return 0, candidates, candidates
    i, j = candidates[:, 0], candidates[:, 1]
    distance = np.linalg.norm(positions[i] - positions[j], axis=1)
    guaranteed = distance <= lower[i] + lower[j] + threshold
    possible = distance <= upper[i] + upper[j] + threshold
    return len(candidates), _lexsorted(candidates[guaranteed]), _lexsorted(candidates[possible])


def all_pairs_search(
    positions: np.ndarray, lower: np.ndarray, upper: np.ndarray, threshold: float
) -> tuple[int, np.ndarray, np.ndarray]:
    """Enumerate all pairs in bounded-size vectorized rows."""

    n = len(positions)
    guaranteed_blocks: list[np.ndarray] = []
    possible_blocks: list[np.ndarray] = []
    for i in range(n - 1):
        j = np.arange(i + 1, n, dtype=np.int64)
        distance = np.linalg.norm(positions[j] - positions[i], axis=1)
        guaranteed = distance <= lower[i] + lower[j] + threshold
        possible = distance <= upper[i] + upper[j] + threshold
        if np.any(guaranteed):
            selected = j[guaranteed]
            guaranteed_blocks.append(np.column_stack((np.full(len(selected), i), selected)))
        if np.any(possible):
            selected = j[possible]
            possible_blocks.append(np.column_stack((np.full(len(selected), i), selected)))
    empty = np.empty((0, 2), dtype=np.int64)
    guaranteed_pairs = np.concatenate(guaranteed_blocks) if guaranteed_blocks else empty
    possible_pairs = np.concatenate(possible_blocks) if possible_blocks else empty
    return n * (n - 1) // 2, guaranteed_pairs, possible_pairs


def timed(callable_search, repeats: int) -> tuple[float, list[float], tuple[int, np.ndarray, np.ndarray]]:
    durations: list[float] = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = callable_search()
        durations.append(time.perf_counter() - start)
    assert result is not None
    return float(np.median(durations)), durations, result


def benchmark(
    *, input_root: Path, half_width: float, threshold: float, repeats: int, max_reference_nodes: int
) -> dict[str, object]:
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        sources = list(csv.DictReader(handle))
    chosen: dict[int, dict[str, str]] = {}
    for row in sources:
        count = int(row["particle_count"])
        chosen.setdefault(count, row)
    records: list[dict[str, object]] = []
    for n, row in sorted(chosen.items()):
        relative = PurePosixPath(row["relative_path"])
        if not relative.parts or relative.parts[0] != "builds" or ".." in relative.parts:
            raise ValueError(f"Unsafe manifest path: {relative}")
        source = input_root.joinpath(*relative.parts)
        if source.stat().st_size != int(row["bytes"]) or sha256(source) != row["sha256"]:
            raise ValueError(f"Raw source differs from manifest: {row['run_id']}")
        positions, nominal, types, frame_name = read_frame(source)
        if len(positions) != n or not np.all(np.isfinite(positions)) or np.any(nominal <= 0):
            raise ValueError(f"Invalid geometry in {row['run_id']}:{frame_name}")
        lower, upper = radius_bounds(nominal, types, half_width)
        tree_seconds, tree_durations, tree_result = timed(
            lambda: tree_search(positions, lower, upper, threshold), repeats
        )
        record: dict[str, object] = {
            "run_id": row["run_id"],
            "source_sha256": row["sha256"],
            "frame": frame_name,
            "n_nodes": n,
            "all_pair_count": n * (n - 1) // 2,
            "tree_candidate_count": tree_result[0],
            "guaranteed_edge_count": len(tree_result[1]),
            "possible_edge_count": len(tree_result[2]),
            "tree_pair_classification_seconds_median": tree_seconds,
            "tree_pair_classification_seconds_repeats": tree_durations,
            "reference_seconds_median": None,
            "reference_seconds_repeats": [],
            "reference_equal": None,
            "reference_repeats": 0,
        }
        if n <= max_reference_nodes:
            reference_seconds, reference_durations, reference_result = timed(
                lambda: all_pairs_search(positions, lower, upper, threshold), repeats
            )
            equal = bool(
                np.array_equal(tree_result[1], reference_result[1])
                and np.array_equal(tree_result[2], reference_result[2])
            )
            if not equal:
                raise AssertionError(f"Endpoint edges differ for {row['run_id']}")
            record.update(
                reference_seconds_median=reference_seconds,
                reference_seconds_repeats=reference_durations,
                reference_equal=equal,
                reference_repeats=repeats,
            )
        records.append(record)
    return {
        "schema": "interval-radius-pair-benchmark-1",
        "benchmark_scope": "pair generation and endpoint classification only; no component analysis",
        "reference_scope": "blocked all-pairs exact enumeration",
        "frame_selection": "middle saved frame of the first manifest run at each node count",
        "relative_half_width": half_width,
        "surface_gap_threshold_nm": threshold,
        "repeats": repeats,
        "max_reference_nodes": max_reference_nodes,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-half-width", type=float, default=0.10)
    parser.add_argument("--threshold-nm", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-reference-nodes", type=int, default=7500)
    args = parser.parse_args()
    if args.repeats < 1 or args.max_reference_nodes < 2:
        parser.error("repeats must be positive and max-reference-nodes at least two")
    result = benchmark(
        input_root=args.input_root,
        half_width=args.relative_half_width,
        threshold=args.threshold_nm,
        repeats=args.repeats,
        max_reference_nodes=args.max_reference_nodes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
