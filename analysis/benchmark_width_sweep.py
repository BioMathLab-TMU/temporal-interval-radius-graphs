#!/usr/bin/env python3
"""Compare width-event union-find with repeated endpoint CSR calculations.

Both methods use one exact safe KD-tree candidate search and return the same
four component/edge observables at every requested width. The baseline
rebuilds sparse endpoint graphs at each width; the event method sorts edge
thresholds once and sweeps. Raw-file hashing and HDF5 reading are excluded
from timed sections. Timings are conditional on this machine and workload.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path, PurePosixPath

import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from critical_width_sweep import prepare_width_sweep  # noqa: E402
from benchmark_pair_search import read_frame, sha256  # noqa: E402


def _summary(n: int, pairs: np.ndarray) -> dict[str, int]:
    degree = np.bincount(pairs.reshape(-1), minlength=n)
    if len(pairs):
        rows = np.concatenate((pairs[:, 0], pairs[:, 1]))
        columns = np.concatenate((pairs[:, 1], pairs[:, 0]))
        matrix = csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, columns)), shape=(n, n))
    else:
        matrix = csr_matrix((n, n), dtype=np.uint8)
    count, labels = connected_components(matrix, directed=False, return_labels=True)
    sizes = np.bincount(labels, minlength=count)
    return {"edges": len(pairs), "components": int(count),
            "largest_component": int(sizes.max()), "isolated_nodes": int(np.sum(degree == 0))}


def repeated_endpoints(positions: np.ndarray, radii: np.ndarray, varied: np.ndarray,
                       widths: np.ndarray, threshold: float) -> list[dict[str, object]]:
    maximum = float(np.max(radii * (1 + widths[-1] * varied)))
    reach = 2 * maximum + threshold
    pairs = np.asarray(cKDTree(positions).query_pairs(reach, output_type="ndarray"), dtype=np.int64).reshape(-1, 2)
    i, j = pairs[:, 0], pairs[:, 1]
    distances = np.linalg.norm(positions[i] - positions[j], axis=1)
    output = []
    for width in widths:
        lower = radii * (1 - width * varied)
        upper = radii * (1 + width * varied)
        guaranteed = pairs[distances <= lower[i] + lower[j] + threshold]
        possible = pairs[distances <= upper[i] + upper[j] + threshold]
        output.append({"width": float(width), "guaranteed": _summary(len(radii), guaranteed),
                       "possible": _summary(len(radii), possible)})
    return output


def benchmark(input_root: Path, repeats: int) -> dict[str, object]:
    with (ROOT / "data/provenance/RUN_PROVENANCE.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = {}
    for row in rows:
        selected.setdefault(int(row["particle_count"]), row)
    records = []
    for n, row in sorted(selected.items()):
        relative = PurePosixPath(row["relative_path"])
        if relative.parts[0] != "builds" or ".." in relative.parts:
            raise ValueError("unsafe provenance path")
        source = input_root.joinpath(*relative.parts)
        if source.stat().st_size != int(row["bytes"]) or sha256(source) != row["sha256"]:
            raise ValueError(f"source mismatch: {row['run_id']}")
        positions, radii, types, frame = read_frame(source)
        varied = types == 1
        for count in (4, 40):
            widths = np.linspace(0, 0.10, count)
            run_sweep = lambda: prepare_width_sweep(positions, radii, varied, 2.0, .10).evaluate(widths)
            run_baseline = lambda: repeated_endpoints(positions, radii, varied, widths, 2.0)
            reference = run_baseline()
            observed = run_sweep()
            if observed != reference:
                raise AssertionError(f"endpoint mismatch: {row['run_id']} Q={count}")
            elapsed = {"repeated_csr": [], "width_sweep": []}
            for repetition in range(repeats):
                order = ("repeated_csr", "width_sweep") if repetition % 2 == 0 else ("width_sweep", "repeated_csr")
                for method in order:
                    start = time.perf_counter()
                    result = run_baseline() if method == "repeated_csr" else run_sweep()
                    elapsed[method].append(time.perf_counter() - start)
                    if result != reference:
                        raise AssertionError(f"repeat mismatch: {row['run_id']} {method}")
            baseline_median = float(np.median(elapsed["repeated_csr"]))
            sweep_median = float(np.median(elapsed["width_sweep"]))
            records.append({"run_id": row["run_id"], "source_sha256": row["sha256"],
                            "frame": frame, "n_nodes": n, "query_width_count": count,
                            "repeats": repeats, "exact_summary_equality": True,
                            "repeated_csr_seconds": elapsed["repeated_csr"],
                            "width_sweep_seconds": elapsed["width_sweep"],
                            "repeated_csr_median_seconds": baseline_median,
                            "width_sweep_median_seconds": sweep_median,
                            "median_time_ratio_repeated_over_sweep": baseline_median / sweep_median})
            print(f"WIDTH_BENCH_OK n={n} Q={count} ratio={baseline_median/sweep_median:.2f}", flush=True)
    return {"schema": "critical-width-benchmark-1", "scope": "safe KD-tree candidate generation plus four endpoint observables; excludes file read and hash",
            "baseline": "one KD-tree search, then repeated sparse CSR endpoint calculations",
            "width_sweep": "one KD-tree search, event sort, union-find queries",
            "frame_selection": "middle saved frame of first provenance-listed run per node count",
            "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "platform": platform.platform(), "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    result = benchmark(args.input_root.resolve(), args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
