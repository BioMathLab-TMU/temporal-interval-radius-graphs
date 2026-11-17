#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from critical_width_sweep import ambiguity_quotient, prepare_width_sweep


def brute_force(positions, radii, varied, threshold, width):
    lower = radii * (1 - width * varied)
    upper = radii * (1 + width * varied)
    n = len(radii)
    edge_lists = [[], []]
    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance <= lower[i] + lower[j] + threshold:
                edge_lists[0].append((i, j))
            if distance <= upper[i] + upper[j] + threshold:
                edge_lists[1].append((i, j))

    def summary(edges):
        parent = list(range(n))
        degree = [0] * n

        def root(i):
            while parent[i] != i:
                i = parent[i]
            return i

        for i, j in edges:
            degree[i] += 1
            degree[j] += 1
            parent[root(i)] = root(j)
        sizes = {r: sum(root(i) == r for i in range(n)) for r in {root(i) for i in range(n)}}
        return {
            "edges": len(edges),
            "components": len(sizes),
            "largest_component": max(sizes.values()),
            "isolated_nodes": degree.count(0),
        }

    return summary(edge_lists[0]), summary(edge_lists[1])


class CriticalWidthSweepTests(unittest.TestCase):
    def test_exact_birth_and_exit_conventions(self):
        positions = np.array([[0.0, 0.0], [2.5, 0.0]])
        radii = np.ones(2)
        sweep = prepare_width_sweep(positions, radii, [True, True], 0.0, 0.5)
        self.assertEqual(sweep.critical_widths().tolist(), [0.0, 0.25, 0.5])
        below, at, above = sweep.evaluate([0.24, 0.25, 0.26])
        self.assertEqual(below["possible"]["edges"], 0)
        self.assertEqual(at["possible"]["edges"], 1)
        self.assertEqual(above["possible"]["edges"], 1)
        self.assertEqual(at["guaranteed"]["edges"], 0)

        # At nominal equality, a guaranteed edge is present at width zero and
        # vanishes immediately above zero.
        positions[1, 0] = 2.0
        sweep = prepare_width_sweep(positions, radii, [True, True], 0.0, 0.5)
        at_zero, above_zero = sweep.evaluate([0.0, 0.01])
        self.assertEqual(at_zero["guaranteed"]["edges"], 1)
        self.assertEqual(above_zero["guaranteed"]["edges"], 0)

    def test_random_geometry_matches_independent_all_pairs(self):
        rng = np.random.default_rng(20261116)
        for _ in range(12):
            positions = rng.uniform(-2, 2, (9, 3))
            radii = rng.uniform(0.2, 1.0, 9)
            varied = rng.integers(0, 2, 9).astype(bool)
            threshold = float(rng.uniform(-0.3, 0.4))
            sweep = prepare_width_sweep(positions, radii, varied, threshold, 0.5)
            events = sweep.critical_widths()
            # At a computed critical ratio, separate floating-point operation
            # orders can disagree by one ULP. Test all open intervals here;
            # exact equality is tested above with representable values.
            probe = np.unique(np.concatenate(([0.0, 0.025, 0.05, 0.1, 0.5], (events[:-1] + events[1:]) / 2)))
            for result in sweep.evaluate(probe):
                guaranteed, possible = brute_force(
                    positions, radii, varied, threshold, result["width"]
                )
                self.assertEqual(result["guaranteed"], guaranteed)
                self.assertEqual(result["possible"], possible)

    def test_static_radii_ties_and_query_order(self):
        positions = np.array([[0, 0], [2, 0], [4, 0]], dtype=float)
        radii = np.ones(3)
        sweep = prepare_width_sweep(positions, radii, [False] * 3, 0.0, 0.5)
        results = sweep.evaluate([0.5, 0.0, 0.5])
        self.assertEqual(results[0], results[2])
        self.assertEqual(results[0]["guaranteed"], results[1]["guaranteed"])
        self.assertEqual(results[0]["possible"], results[1]["possible"])

    def test_component_transitions_filter_redundant_edge_events(self):
        positions = np.array([[0, 0], [2.5, 0], [5, 0]], dtype=float)
        sweep = prepare_width_sweep(positions, np.ones(3), [True] * 3, 0, 0.5)
        transitions = sweep.component_transition_widths()
        np.testing.assert_allclose(transitions["possible_component_mergers"], [0.25, 0.25])
        self.assertEqual(len(transitions["guaranteed_component_splits"]), 0)

        positions[:, 0] = [0, 1.5, 3]
        sweep = prepare_width_sweep(positions, np.ones(3), [True] * 3, 0, 0.5)
        transitions = sweep.component_transition_widths()
        np.testing.assert_allclose(transitions["guaranteed_component_splits"], [0.25, 0.25])

    def test_ambiguity_quotient_distinguishes_redundancy(self):
        guaranteed = np.array([[0, 1], [2, 3]])
        possible = np.array([[0, 1], [2, 3], [0, 2], [1, 3], [0, 3]])
        result = ambiguity_quotient(4, guaranteed, possible)
        self.assertEqual(result["ambiguous_edges_between_guaranteed_components"], 3)
        self.assertEqual(result["component_mergers"], 1)

    def test_invalid_domain_and_query(self):
        with self.assertRaises(ValueError):
            prepare_width_sweep(np.zeros((2, 3)), np.ones(2), [True, True], 0, 1)
        sweep = prepare_width_sweep(np.zeros((2, 3)), np.ones(2), [True, True], 0, 0.5)
        with self.assertRaises(ValueError):
            sweep.evaluate([0.6])


if __name__ == "__main__":
    unittest.main()
