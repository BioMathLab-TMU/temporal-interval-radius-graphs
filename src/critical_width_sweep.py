"""Exact endpoint-connectivity queries over a radius-uncertainty width.

The nominal centres and radii are fixed for one observed frame.  Nodes marked
``varied`` have radii ``[(1-a)r_i, (1+a)r_i]`` for ``0 <= a < 1``; other
nodes retain their nominal radii.  This is a sensitivity calculation, not a
model of radius dynamics between observed frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


class _Components:
    def __init__(self, count: int) -> None:
        self.parent = np.arange(count, dtype=np.int64)
        self.size = np.ones(count, dtype=np.int64)
        self.degree = np.zeros(count, dtype=np.int64)
        self.n_components = count
        self.largest = 1
        self.isolated = count
        self.n_edges = 0

    def find(self, node: int) -> int:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = int(self.parent[node])
        return node

    def add(self, left: int, right: int) -> bool:
        if self.degree[left] == 0:
            self.isolated -= 1
        if self.degree[right] == 0:
            self.isolated -= 1
        self.degree[left] += 1
        self.degree[right] += 1
        self.n_edges += 1
        a, b = self.find(left), self.find(right)
        if a == b:
            return False
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]
        self.n_components -= 1
        self.largest = max(self.largest, int(self.size[a]))
        return True

    def snapshot(self) -> dict[str, int]:
        return {
            "edges": self.n_edges,
            "components": self.n_components,
            "largest_component": self.largest,
            "isolated_nodes": self.isolated,
        }


@dataclass(frozen=True)
class WidthSweep:
    """Candidate pairs and exact endpoint-edge event widths for one frame.

    ``possible_births`` and ``guaranteed_exits`` use infinity for pairs that
    never switch within the specified domain.  These are edge events; a graph
    observable changes only when an event merges or separates components.
    """

    n_nodes: int
    max_width: float
    candidate_pairs: np.ndarray
    possible_births: np.ndarray
    guaranteed_exits: np.ndarray
    nominal_edges: np.ndarray
    maximum_width_guaranteed_edges: np.ndarray

    def critical_widths(self) -> np.ndarray:
        births = self.possible_births[np.isfinite(self.possible_births)]
        exits = self.guaranteed_exits[np.isfinite(self.guaranteed_exits)]
        return np.unique(np.concatenate(([0.0, self.max_width], births, exits)))

    def component_transition_widths(self) -> dict[str, np.ndarray]:
        """Return event widths that actually change endpoint component counts.

        Possible-graph values are births in ascending order. Guaranteed-graph
        values are exits in descending order: their reverse-sweep unions become
        component splits when uncertainty width increases past an exit.
        """

        possible = _Components(self.n_nodes)
        guaranteed = _Components(self.n_nodes)
        pairs = self.candidate_pairs
        for left, right in pairs[self.nominal_edges]:
            possible.add(int(left), int(right))
        for left, right in pairs[self.maximum_width_guaranteed_edges]:
            guaranteed.add(int(left), int(right))
        birth_indices = np.flatnonzero(np.isfinite(self.possible_births))
        birth_indices = birth_indices[np.argsort(self.possible_births[birth_indices], kind="mergesort")]
        exit_indices = np.flatnonzero(np.isfinite(self.guaranteed_exits))
        exit_indices = exit_indices[np.argsort(-self.guaranteed_exits[exit_indices], kind="mergesort")]
        birth_mergers = []
        exit_splits = []
        for index in birth_indices:
            left, right = pairs[index]
            if possible.add(int(left), int(right)):
                birth_mergers.append(float(self.possible_births[index]))
        for index in exit_indices:
            left, right = pairs[index]
            if guaranteed.add(int(left), int(right)):
                exit_splits.append(float(self.guaranteed_exits[index]))
        return {
            "possible_component_mergers": np.asarray(birth_mergers, dtype=np.float64),
            "guaranteed_component_splits": np.asarray(exit_splits, dtype=np.float64),
        }

    def evaluate(self, widths: np.ndarray | list[float]) -> list[dict[str, object]]:
        """Evaluate both endpoint graphs at requested widths, including ties.

        Possible edges enter at their birth width.  Guaranteed edges remain
        present at their exit width and disappear immediately above it.
        Sorting edge events and queries costs O(M log M + Q log Q)
        for M candidate pairs and Q query widths, plus near-linear unions.
        """

        queries = np.asarray(widths, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(queries)) or np.any(queries < 0) or np.any(queries > self.max_width):
            raise ValueError("query widths must be finite and inside [0, max_width]")
        if len(queries) == 0:
            return []
        pairs = self.candidate_pairs
        possible = _Components(self.n_nodes)
        guaranteed = _Components(self.n_nodes)
        for left, right in pairs[self.nominal_edges]:
            possible.add(int(left), int(right))
        for left, right in pairs[self.maximum_width_guaranteed_edges]:
            guaranteed.add(int(left), int(right))

        birth_order = np.argsort(self.possible_births, kind="mergesort")
        exit_order = np.argsort(-self.guaranteed_exits, kind="mergesort")
        results: list[dict[str, object] | None] = [None] * len(queries)
        possible_at: list[dict[str, int] | None] = [None] * len(queries)
        birth_cursor = 0
        for index in np.argsort(queries, kind="mergesort"):
            width = float(queries[index])
            while birth_cursor < len(birth_order) and self.possible_births[birth_order[birth_cursor]] <= width:
                pair_index = birth_order[birth_cursor]
                left, right = pairs[pair_index]
                possible.add(int(left), int(right))
                birth_cursor += 1
            possible_at[index] = possible.snapshot()

        exit_cursor = 0
        for index in np.argsort(-queries, kind="mergesort"):
            width = float(queries[index])
            while exit_cursor < len(exit_order) and self.guaranteed_exits[exit_order[exit_cursor]] >= width:
                pair_index = exit_order[exit_cursor]
                left, right = pairs[pair_index]
                guaranteed.add(int(left), int(right))
                exit_cursor += 1
            results[index] = {
                "width": width,
                "guaranteed": guaranteed.snapshot(),
                "possible": possible_at[index],
            }
        return [result for result in results if result is not None]


def prepare_width_sweep(
    positions_nm: np.ndarray,
    nominal_radii_nm: np.ndarray,
    varied: np.ndarray,
    threshold_nm: float,
    max_width: float,
) -> WidthSweep:
    """Construct all edge-event widths with an exact, safe spatial search."""

    positions = np.asarray(positions_nm, dtype=np.float64)
    radii = np.asarray(nominal_radii_nm, dtype=np.float64).reshape(-1)
    varied = np.asarray(varied, dtype=bool).reshape(-1)
    threshold = float(threshold_nm)
    max_width = float(max_width)
    if positions.ndim != 2 or positions.shape[1] not in (2, 3) or len(positions) == 0:
        raise ValueError("positions must be a nonempty (n, 2) or (n, 3) array")
    if len(radii) != len(positions) or len(varied) != len(positions):
        raise ValueError("radii and varied mask must match the positions")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(radii)) or np.any(radii <= 0):
        raise ValueError("positions must be finite and nominal radii positive and finite")
    if not np.isfinite(threshold) or not np.isfinite(max_width) or not 0 <= max_width < 1:
        raise ValueError("threshold must be finite and max_width must be in [0, 1)")
    upper = radii * (1 + max_width * varied)
    reach = 2 * float(np.max(upper)) + threshold
    if len(radii) < 2 or reach < 0:
        pairs = np.empty((0, 2), dtype=np.int64)
    else:
        pairs = np.asarray(cKDTree(positions).query_pairs(reach, output_type="ndarray"), dtype=np.int64).reshape(-1, 2)
    if not len(pairs):
        empty_float = np.empty(0, dtype=np.float64)
        empty_bool = np.empty(0, dtype=bool)
        return WidthSweep(len(radii), max_width, pairs, empty_float, empty_float, empty_bool, empty_bool)
    left, right = pairs[:, 0], pairs[:, 1]
    distances = np.linalg.norm(positions[left] - positions[right], axis=1)
    nominal_sum = radii[left] + radii[right]
    variable_sum = radii[left] * varied[left] + radii[right] * varied[right]
    nominal_edges = distances <= nominal_sum + threshold
    maximum_guaranteed = distances <= nominal_sum - max_width * variable_sum + threshold
    possible_at_maximum = distances <= nominal_sum + max_width * variable_sum + threshold
    births = np.full(len(pairs), np.inf)
    entering = (~nominal_edges) & possible_at_maximum & (variable_sum > 0)
    births[entering] = (distances[entering] - nominal_sum[entering] - threshold) / variable_sum[entering]
    exits = np.full(len(pairs), -np.inf)
    leaving = nominal_edges & (~maximum_guaranteed) & (variable_sum > 0)
    exits[leaving] = (nominal_sum[leaving] + threshold - distances[leaving]) / variable_sum[leaving]
    return WidthSweep(len(radii), max_width, pairs, births, exits, nominal_edges, maximum_guaranteed)


def ambiguity_quotient(
    n_nodes: int, guaranteed_pairs: np.ndarray, possible_pairs: np.ndarray
) -> dict[str, int]:
    """Count uncertain edges within and between guaranteed components.

    The quotient of guaranteed components by cross-component uncertain edges
    has exactly the possible graph's component partition.  Cross-component
    edges are candidates for mergers; they need not individually be bridges.
    """

    lower = np.asarray(guaranteed_pairs, dtype=np.int64).reshape(-1, 2)
    upper = np.asarray(possible_pairs, dtype=np.int64).reshape(-1, 2)
    if n_nodes < 1 or np.any(lower < 0) or np.any(upper < 0) or np.any(lower >= n_nodes) or np.any(upper >= n_nodes):
        raise ValueError("invalid node count or edge endpoint")
    lower_set = {tuple(sorted(map(int, pair))) for pair in lower}
    upper_set = {tuple(sorted(map(int, pair))) for pair in upper}
    if not lower_set.issubset(upper_set):
        raise ValueError("guaranteed edges must be a subset of possible edges")
    components = _Components(n_nodes)
    for left, right in lower_set:
        components.add(left, right)
    guaranteed_component_count = components.n_components
    inside = outside = 0
    for left, right in upper_set - lower_set:
        if components.find(left) == components.find(right):
            inside += 1
        else:
            outside += 1
    for left, right in upper_set - lower_set:
        components.add(left, right)
    return {
        "ambiguous_edges_inside_guaranteed_components": inside,
        "ambiguous_edges_between_guaranteed_components": outside,
        "guaranteed_components": guaranteed_component_count,
        "possible_components": components.n_components,
        "component_mergers": guaranteed_component_count - components.n_components,
    }
