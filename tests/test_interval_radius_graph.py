#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PIPELINE_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PIPELINE_PATH))
import interval_radius_graph as MODULE


class IntervalRadiusGraphTests(unittest.TestCase):
    def test_three_edge_flags(self) -> None:
        distances = np.array([1.5, 2.5, 4.0])
        lower = np.ones(3)
        upper = np.full(3, 1.5)
        flags, minimum_gap, maximum_gap = MODULE.classify_edge_flags(
            distances, lower, lower, upper, upper, 0.0
        )
        np.testing.assert_array_equal(
            flags,
            np.array(
                [MODULE.EDGE_GUARANTEED, MODULE.EDGE_RADIUS_DEPENDENT, MODULE.EDGE_IMPOSSIBLE],
                dtype=np.int8,
            ),
        )
        np.testing.assert_allclose(minimum_gap, [-1.5, -0.5, 1.0])
        np.testing.assert_allclose(maximum_gap, [-0.5, 0.5, 2.0])

    def test_graph_endpoints_are_realised_by_radius_endpoints(self) -> None:
        positions = np.array([[0.0, 0.0], [2.4, 0.0], [5.0, 0.0], [20.0, 0.0]])
        lower = np.array([1.0, 1.0, 1.0, 0.5])
        upper = np.array([1.5, 1.4, 1.5, 0.75])
        result = MODULE.build_interval_radius_graph(positions, lower, upper, 0.0)
        lower_graph = MODULE.build_interval_radius_graph(positions, lower, lower, 0.0)
        upper_graph = MODULE.build_interval_radius_graph(positions, upper, upper, 0.0)
        np.testing.assert_array_equal(result.guaranteed_pairs, lower_graph.guaranteed_pairs)
        np.testing.assert_array_equal(result.possible_pairs, upper_graph.possible_pairs)

    def test_every_sampled_graph_is_sandwiched(self) -> None:
        rng = np.random.default_rng(20260926)
        positions = rng.uniform(-4.0, 4.0, size=(18, 3))
        lower = rng.uniform(0.25, 0.7, size=18)
        upper = lower + rng.uniform(0.0, 0.4, size=18)
        result = MODULE.build_interval_radius_graph(positions, lower, upper, 0.3)
        guaranteed = {tuple(pair) for pair in result.guaranteed_pairs.tolist()}
        possible = {tuple(pair) for pair in result.possible_pairs.tolist()}
        for _ in range(100):
            radii = rng.uniform(lower, upper)
            realised = {
                (i, j)
                for i in range(len(radii))
                for j in range(i + 1, len(radii))
                if np.linalg.norm(positions[i] - positions[j]) - radii[i] - radii[j] <= 0.3
            }
            self.assertTrue(guaranteed.issubset(realised))
            self.assertTrue(realised.issubset(possible))

    def test_component_intervals_have_correct_monotone_order(self) -> None:
        positions = np.array([[0.0, 0.0], [2.5, 0.0], [5.1, 0.0], [30.0, 0.0]])
        lower = np.ones(4)
        upper = np.full(4, 1.5)
        result = MODULE.build_interval_radius_graph(positions, lower, upper, 0.0)
        for interval in result.scalar_intervals().values():
            self.assertLessEqual(interval["lower"], interval["upper"])
        self.assertEqual(result.scalar_intervals()["component_count"], {"lower": 2.0, "upper": 4.0})
        self.assertNotIn("recruited_fraction", result.scalar_intervals())
        self.assertIn("isolated_fraction", result.scalar_intervals())

    def test_exact_threshold_is_guaranteed(self) -> None:
        result = MODULE.build_interval_radius_graph(
            np.array([[0.0, 0.0], [3.0, 0.0]]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            1.0,
        )
        np.testing.assert_array_equal(result.guaranteed_pairs, np.array([[0, 1]]))
        self.assertEqual(len(result.radius_dependent_pairs), 0)

    def test_invalid_radius_interval_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.build_interval_radius_graph(
                np.array([[0.0, 0.0], [1.0, 0.0]]),
                np.array([1.0, 1.0]),
                np.array([0.9, 1.1]),
                0.0,
            )

    def test_sharp_interleaving_shift_is_sum_of_two_largest_widths(self) -> None:
        shift = MODULE.endpoint_interleaving_shift(
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.1, 2.7, 3.4, 4.2]),
        )
        self.assertAlmostEqual(shift, 1.1)

    def test_h0_deaths_are_kruskal_merge_weights(self) -> None:
        pairs = np.array([[0, 1], [0, 2], [1, 2], [2, 3]])
        births = np.array([1.0, 3.0, 2.0, 5.0])
        deaths = MODULE.h0_minimum_spanning_forest_deaths(4, pairs, births)
        np.testing.assert_allclose(deaths, [1.0, 2.0, 5.0])

    def test_h0_keeps_weights_attached_to_unsorted_edges(self) -> None:
        pairs = np.array([[1, 3], [0, 1], [0, 2], [2, 3], [1, 2]])
        births = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        deaths = MODULE.h0_minimum_spanning_forest_deaths(4, pairs, births)
        np.testing.assert_allclose(deaths, [1.0, 2.0, 3.0])

    def test_complete_target_pattern_is_realizable(self) -> None:
        positions = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        )
        pairs = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
        result = MODULE.realize_edge_pattern(
            positions,
            np.full(4, 0.2),
            np.full(4, 1.0),
            0.0,
            pairs,
        )
        self.assertTrue(result.realizable)
        self.assertTrue(np.isinf(result.strict_margin_nm))

    def test_two_disjoint_edges_can_be_infeasible_inside_graph_sandwich(self) -> None:
        # Vertices of a regular tetrahedron have equal pairwise distance 1.
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, np.sqrt(3.0) / 2.0, 0.0],
                [0.5, np.sqrt(3.0) / 6.0, np.sqrt(2.0 / 3.0)],
            ]
        )
        result = MODULE.realize_edge_pattern(
            positions,
            np.full(4, 0.2),
            np.full(4, 0.8),
            0.0,
            np.array([[0, 1], [2, 3]]),
        )
        self.assertFalse(result.realizable)
        self.assertAlmostEqual(result.strict_margin_nm, 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
