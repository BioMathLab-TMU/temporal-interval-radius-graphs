#!/usr/bin/env python3
"""Exact graph envelopes for nodes with interval-valued radii.

At one observation time, node ``i`` has a fixed centre ``x_i`` and an
admissible radius interval ``[r_i^-, r_i^+]``.  Nodes ``i`` and ``j`` are
adjacent when their surface gap is no greater than ``tau``::

    ||x_i - x_j|| - r_i - r_j <= tau.

The module constructs two endpoint graphs:

* ``guaranteed``: edges present for every admissible radius assignment;
* ``possible``: edges present for at least one admissible assignment.

Because increasing any radius can only add edges, both endpoints are exactly
attained: all lower radii realise the guaranteed graph and all upper radii
realise the possible graph.  The interval bounds are supplied mathematical
inputs at each observation time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.optimize import linprog
from scipy.spatial import cKDTree


INTERVAL_RADIUS_GRAPH_SCHEMA = "simsv-interval-radius-graph-1.1"

EDGE_IMPOSSIBLE = np.int8(0)
EDGE_RADIUS_DEPENDENT = np.int8(1)
EDGE_GUARANTEED = np.int8(2)


@dataclass(frozen=True)
class RadiusPatternRealizability:
    """Certificate returned by :func:`realize_edge_pattern`.

    A finite strict margin certifies that every requested non-edge remains a
    non-edge.  The margin is infinite for a feasible complete target graph,
    because there are no strict non-edge inequalities to constrain it.
    """

    realizable: bool
    strict_margin_nm: float
    radii_nm: np.ndarray | None


def _normalise_pairs(pairs: np.ndarray) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    return pairs.reshape(-1, 2)


def _validate_simple_pairs(count: int, pairs: np.ndarray) -> np.ndarray:
    """Normalize an undirected edge list and reject loops or duplicates."""

    pairs = _normalise_pairs(pairs)
    if len(pairs) == 0:
        return pairs
    if np.any(pairs < 0) or np.any(pairs >= count):
        raise ValueError("edge endpoint is outside the node range")
    pairs = np.sort(pairs, axis=1)
    if np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError("self-loops are not allowed")
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    if len(pairs) > 1 and np.any(np.all(pairs[1:] == pairs[:-1], axis=1)):
        raise ValueError("duplicate edges are not allowed")
    return pairs


def endpoint_interleaving_shift(
    radius_lower_nm: np.ndarray,
    radius_upper_nm: np.ndarray,
) -> float:
    """Return the sharp uniform shift between the two edge filtrations.

    For at least two nodes this is the sum of the two largest interval widths,
    equivalently ``max_{i<j}[(r_i^+-r_i^-)+(r_j^+-r_j^-)]``.
    """

    lower = np.asarray(radius_lower_nm, dtype=np.float64).reshape(-1)
    upper = np.asarray(radius_upper_nm, dtype=np.float64).reshape(-1)
    if len(lower) != len(upper):
        raise ValueError("radius bounds must have equal length")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("radius bounds must be finite")
    if np.any(lower <= 0.0) or np.any(upper < lower):
        raise ValueError("invalid radius interval")
    if len(lower) < 2:
        return 0.0
    widths = upper - lower
    two_largest = np.partition(widths, len(widths) - 2)[-2:]
    return float(np.sum(two_largest))


def h0_minimum_spanning_forest_deaths(
    n_nodes: int,
    pairs: np.ndarray,
    edge_birth_values: np.ndarray,
) -> np.ndarray:
    """Return finite ``H_0`` death parameters from a weighted edge list.

    Kruskal's algorithm adds an edge exactly when two connected components
    merge.  A complete edge list therefore returns ``n_nodes - 1`` finite
    deaths; a truncated list returns the deaths visible in its filtration
    window.
    """

    n_nodes = int(n_nodes)
    if n_nodes < 1:
        raise ValueError("n_nodes must be positive")
    # Validation may sort undirected pairs; the supplied birth values refer to
    # the original row order, so keep that order for Kruskal's algorithm.
    pairs = _normalise_pairs(pairs)
    _validate_simple_pairs(n_nodes, pairs)
    births = np.asarray(edge_birth_values, dtype=np.float64).reshape(-1)
    if len(births) != len(pairs):
        raise ValueError("one birth value is required for every edge")
    if not np.all(np.isfinite(births)):
        raise ValueError("edge birth values must be finite")

    parent = np.arange(n_nodes, dtype=np.int64)
    rank = np.zeros(n_nodes, dtype=np.int8)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    deaths: list[float] = []
    for index in np.argsort(births, kind="mergesort"):
        left, right = (int(value) for value in pairs[index])
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            continue
        if rank[root_left] < rank[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        if rank[root_left] == rank[root_right]:
            rank[root_left] += 1
        deaths.append(float(births[index]))
    return np.asarray(deaths, dtype=np.float64)


def realize_edge_pattern(
    positions: np.ndarray,
    radius_lower_nm: np.ndarray,
    radius_upper_nm: np.ndarray,
    threshold_nm: float,
    target_pairs: np.ndarray,
    *,
    strict_tolerance_nm: float = 1.0e-9,
) -> RadiusPatternRealizability:
    """Numerically test realizability of a prescribed intermediate graph by LP.

    Requested edges satisfy ``r_i+r_j >= d_ij-threshold``.  Requested
    non-edges satisfy the strict reverse inequality.  Maximizing a common
    non-edge slack converts the finite strict system into a linear program;
    the target is mathematically realizable exactly when the optimal slack is
    positive. This floating-point implementation uses ``strict_tolerance_nm``
    when deciding whether the computed slack is positive.
    """

    if not np.isfinite(strict_tolerance_nm) or strict_tolerance_nm < 0.0:
        raise ValueError("strict_tolerance_nm must be finite and nonnegative")

    positions, lower, upper, threshold_nm = _validate_inputs(
        positions, radius_lower_nm, radius_upper_nm, threshold_nm
    )
    count = len(positions)
    target_pairs = _validate_simple_pairs(count, target_pairs)
    pair_i, pair_j = np.triu_indices(count, k=1)
    all_pairs = np.column_stack([pair_i, pair_j]).astype(np.int64, copy=False)
    target_set = {tuple(pair) for pair in target_pairs.tolist()}
    target_mask = np.fromiter(
        (tuple(pair) in target_set for pair in all_pairs),
        dtype=bool,
        count=len(all_pairs),
    )
    distances = np.linalg.norm(
        positions[all_pairs[:, 0]] - positions[all_pairs[:, 1]], axis=1
    )
    required_sums = distances - threshold_nm

    edge_pairs = all_pairs[target_mask]
    edge_bounds = required_sums[target_mask]
    nonedge_pairs = all_pairs[~target_mask]
    nonedge_bounds = required_sums[~target_mask]

    if len(nonedge_pairs) == 0:
        matrix = np.zeros((len(edge_pairs), count), dtype=np.float64)
        if len(edge_pairs):
            rows = np.arange(len(edge_pairs))
            matrix[rows, edge_pairs[:, 0]] = -1.0
            matrix[rows, edge_pairs[:, 1]] = -1.0
        result = linprog(
            np.zeros(count),
            A_ub=matrix if len(matrix) else None,
            b_ub=-edge_bounds if len(matrix) else None,
            bounds=list(zip(lower, upper)),
            method="highs",
        )
        return RadiusPatternRealizability(
            realizable=bool(result.success),
            strict_margin_nm=float("inf") if result.success else 0.0,
            radii_nm=np.asarray(result.x, dtype=np.float64) if result.success else None,
        )

    # Variables are (r_1,...,r_n, epsilon), where epsilon is the common
    # strict slack assigned to every requested non-edge.
    variable_count = count + 1
    matrix = np.zeros((len(edge_pairs) + len(nonedge_pairs), variable_count))
    bounds_vector = np.empty(len(matrix), dtype=np.float64)
    if len(edge_pairs):
        rows = np.arange(len(edge_pairs))
        matrix[rows, edge_pairs[:, 0]] = -1.0
        matrix[rows, edge_pairs[:, 1]] = -1.0
        bounds_vector[rows] = -edge_bounds
    nonedge_rows = np.arange(len(edge_pairs), len(matrix))
    matrix[nonedge_rows, nonedge_pairs[:, 0]] = 1.0
    matrix[nonedge_rows, nonedge_pairs[:, 1]] = 1.0
    matrix[nonedge_rows, -1] = 1.0
    bounds_vector[nonedge_rows] = nonedge_bounds
    objective = np.zeros(variable_count)
    objective[-1] = -1.0
    result = linprog(
        objective,
        A_ub=matrix,
        b_ub=bounds_vector,
        bounds=[*zip(lower, upper), (0.0, None)],
        method="highs",
    )
    margin = float(result.x[-1]) if result.success else 0.0
    realizable = bool(result.success and margin > float(strict_tolerance_nm))
    return RadiusPatternRealizability(
        realizable=realizable,
        strict_margin_nm=margin,
        radii_nm=np.asarray(result.x[:-1], dtype=np.float64) if realizable else None,
    )


def _validate_inputs(
    positions: np.ndarray,
    radius_lower: np.ndarray,
    radius_upper: np.ndarray,
    threshold_nm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    positions = np.asarray(positions, dtype=np.float64)
    radius_lower = np.asarray(radius_lower, dtype=np.float64).reshape(-1)
    radius_upper = np.asarray(radius_upper, dtype=np.float64).reshape(-1)
    threshold_nm = float(threshold_nm)

    if positions.ndim != 2 or positions.shape[1] not in (2, 3):
        raise ValueError("positions must have shape (n, 2) or (n, 3)")
    if len(positions) == 0:
        raise ValueError("at least one node is required")
    if len(radius_lower) != len(positions) or len(radius_upper) != len(positions):
        raise ValueError("positions and radius bounds must have equal length")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions must be finite")
    if not np.all(np.isfinite(radius_lower)) or not np.all(np.isfinite(radius_upper)):
        raise ValueError("radius bounds must be finite")
    if np.any(radius_lower <= 0.0):
        raise ValueError("lower radius bounds must be strictly positive")
    if np.any(radius_upper < radius_lower):
        raise ValueError("every upper radius bound must be at least its lower bound")
    if not np.isfinite(threshold_nm):
        raise ValueError("threshold_nm must be finite")
    return positions, radius_lower, radius_upper, threshold_nm


def classify_edge_flags(
    centre_distances_nm: np.ndarray,
    radius_lower_i_nm: np.ndarray,
    radius_lower_j_nm: np.ndarray,
    radius_upper_i_nm: np.ndarray,
    radius_upper_j_nm: np.ndarray,
    threshold_nm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify pairwise edges and return flags plus extremal surface gaps.

    Flag values are ``0`` (impossible), ``1`` (radius-dependent), and ``2``
    (guaranteed).  The minimum possible gap uses both upper radii; the maximum
    possible gap uses both lower radii.
    """

    distance = np.asarray(centre_distances_nm, dtype=np.float64)
    lower_i = np.asarray(radius_lower_i_nm, dtype=np.float64)
    lower_j = np.asarray(radius_lower_j_nm, dtype=np.float64)
    upper_i = np.asarray(radius_upper_i_nm, dtype=np.float64)
    upper_j = np.asarray(radius_upper_j_nm, dtype=np.float64)
    minimum_gap = distance - upper_i - upper_j
    maximum_gap = distance - lower_i - lower_j
    possible = minimum_gap <= float(threshold_nm)
    guaranteed = maximum_gap <= float(threshold_nm)
    flags = np.full(np.broadcast_shapes(distance.shape, lower_i.shape), EDGE_IMPOSSIBLE, dtype=np.int8)
    flags[possible] = EDGE_RADIUS_DEPENDENT
    flags[guaranteed] = EDGE_GUARANTEED
    return flags, minimum_gap, maximum_gap


def _component_summary(count: int, pairs: np.ndarray) -> Dict[str, object]:
    pairs = _normalise_pairs(pairs)
    if len(pairs):
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        columns = np.concatenate([pairs[:, 1], pairs[:, 0]])
        adjacency = csr_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
            shape=(count, count),
        )
        degree = np.bincount(pairs.reshape(-1), minlength=count)
    else:
        adjacency = csr_matrix((count, count), dtype=np.uint8)
        degree = np.zeros(count, dtype=np.int64)
    n_components, labels = connected_components(adjacency, directed=False, return_labels=True)
    sizes = np.bincount(labels, minlength=n_components)
    return {
        "n_components": int(n_components),
        "labels": labels.astype(np.int64, copy=False),
        "component_sizes": sizes.astype(np.int64, copy=False),
        "degree": degree.astype(np.int64, copy=False),
        "largest_component_size": int(sizes.max()),
        "largest_component_fraction": float(sizes.max() / count),
        "isolated_fraction": float(np.mean(degree == 0)),
        "mean_degree": float(np.mean(degree)),
    }


@dataclass(frozen=True)
class IntervalRadiusGraph:
    """Exact lower/upper graph endpoints for one temporal snapshot."""

    threshold_nm: float
    candidate_pairs: np.ndarray
    candidate_flags: np.ndarray
    minimum_gap_nm: np.ndarray
    maximum_gap_nm: np.ndarray
    guaranteed_pairs: np.ndarray
    possible_pairs: np.ndarray
    radius_dependent_pairs: np.ndarray
    guaranteed_summary: Mapping[str, object]
    possible_summary: Mapping[str, object]

    @property
    def n_nodes(self) -> int:
        return int(len(self.guaranteed_summary["labels"]))

    def scalar_intervals(self) -> Dict[str, Dict[str, float]]:
        """Return exact intervals for standard component observables."""

        lower_graph = self.guaranteed_summary
        upper_graph = self.possible_summary
        return {
            "edge_count": {
                "lower": float(len(self.guaranteed_pairs)),
                "upper": float(len(self.possible_pairs)),
            },
            "largest_component_size": {
                "lower": float(lower_graph["largest_component_size"]),
                "upper": float(upper_graph["largest_component_size"]),
            },
            "largest_component_fraction": {
                "lower": float(lower_graph["largest_component_fraction"]),
                "upper": float(upper_graph["largest_component_fraction"]),
            },
            "mean_degree": {
                "lower": float(lower_graph["mean_degree"]),
                "upper": float(upper_graph["mean_degree"]),
            },
            # Adding edges can only reduce these two observables.
            "component_count": {
                "lower": float(upper_graph["n_components"]),
                "upper": float(lower_graph["n_components"]),
            },
            "isolated_fraction": {
                "lower": float(upper_graph["isolated_fraction"]),
                "upper": float(lower_graph["isolated_fraction"]),
            },
        }

    def to_record(self) -> Dict[str, float | int | str]:
        record: Dict[str, float | int | str] = {
            "schema_version": INTERVAL_RADIUS_GRAPH_SCHEMA,
            "threshold_nm": float(self.threshold_nm),
            "n_nodes": self.n_nodes,
            "candidate_pair_count": int(len(self.candidate_pairs)),
            "guaranteed_edge_count": int(len(self.guaranteed_pairs)),
            "possible_edge_count": int(len(self.possible_pairs)),
            "radius_dependent_edge_count": int(len(self.radius_dependent_pairs)),
            "radius_dependent_fraction_of_possible_edges": (
                float(len(self.radius_dependent_pairs) / len(self.possible_pairs))
                if len(self.possible_pairs)
                else 0.0
            ),
        }
        for name, interval in self.scalar_intervals().items():
            record[f"{name}_lower"] = interval["lower"]
            record[f"{name}_upper"] = interval["upper"]
            record[f"{name}_width"] = interval["upper"] - interval["lower"]
        return record


def build_interval_radius_graph(
    positions: np.ndarray,
    radius_lower_nm: np.ndarray,
    radius_upper_nm: np.ndarray,
    threshold_nm: float,
) -> IntervalRadiusGraph:
    """Construct exact guaranteed and possible graphs for one snapshot."""

    positions, radius_lower, radius_upper, threshold_nm = _validate_inputs(
        positions,
        radius_lower_nm,
        radius_upper_nm,
        threshold_nm,
    )
    count = len(positions)
    maximum_reach = 2.0 * float(np.max(radius_upper)) + threshold_nm
    if count < 2 or maximum_reach < 0.0:
        candidate_pairs = np.empty((0, 2), dtype=np.int64)
    else:
        candidate_pairs = _normalise_pairs(
            cKDTree(positions).query_pairs(maximum_reach, output_type="ndarray")
        )

    if len(candidate_pairs):
        i = candidate_pairs[:, 0]
        j = candidate_pairs[:, 1]
        distances = np.linalg.norm(positions[i] - positions[j], axis=1)
        flags, minimum_gap, maximum_gap = classify_edge_flags(
            distances,
            radius_lower[i],
            radius_lower[j],
            radius_upper[i],
            radius_upper[j],
            threshold_nm,
        )
    else:
        flags = np.empty(0, dtype=np.int8)
        minimum_gap = np.empty(0, dtype=np.float64)
        maximum_gap = np.empty(0, dtype=np.float64)

    guaranteed_pairs = candidate_pairs[flags == EDGE_GUARANTEED]
    radius_dependent_pairs = candidate_pairs[flags == EDGE_RADIUS_DEPENDENT]
    possible_pairs = candidate_pairs[flags >= EDGE_RADIUS_DEPENDENT]
    guaranteed_summary = _component_summary(count, guaranteed_pairs)
    possible_summary = _component_summary(count, possible_pairs)

    return IntervalRadiusGraph(
        threshold_nm=threshold_nm,
        candidate_pairs=candidate_pairs,
        candidate_flags=flags,
        minimum_gap_nm=minimum_gap,
        maximum_gap_nm=maximum_gap,
        guaranteed_pairs=guaranteed_pairs,
        possible_pairs=possible_pairs,
        radius_dependent_pairs=radius_dependent_pairs,
        guaranteed_summary=guaranteed_summary,
        possible_summary=possible_summary,
    )
