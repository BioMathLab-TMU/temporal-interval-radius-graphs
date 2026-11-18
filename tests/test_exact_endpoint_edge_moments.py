"""Analytic graph examples for exact independent/shared edge-count moments."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from correlated_radius_continuum import frame_geometry  # noqa: E402
from exact_endpoint_edge_moments import exact_edge_moments  # noqa: E402


class ExactMomentTests(unittest.TestCase):
    def test_one_symmetric_uncertain_edge(self):
        positions = np.array([[0.0, 0.0], [2.0, 0.0]])
        radii = np.ones(2)
        geometry = frame_geometry(positions, radii, np.ones(2, dtype=bool), 0.2, 0.0)
        result = exact_edge_moments(geometry, 0.2)
        self.assertEqual(result["uncertain_edges"], 1)
        for law in ("independent_uniform", "common_uniform"):
            self.assertAlmostEqual(result[law]["mean_edges"], 0.5, places=12)
            self.assertAlmostEqual(result[law]["variance_edges"], 0.25, places=12)

    def test_two_edges_sharing_one_varied_node(self):
        positions = np.array([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
        radii = np.ones(3)
        varied = np.array([False, True, False])
        geometry = frame_geometry(positions, radii, varied, 0.2, 0.0)
        result = exact_edge_moments(geometry, 0.2)
        self.assertEqual(result["uncertain_edges"], 2)
        self.assertEqual(result["independent_uniform"]["covariance_pairs"], 1)
        for law in ("independent_uniform", "common_uniform"):
            self.assertAlmostEqual(result[law]["mean_edges"], 1.0, places=12)
            self.assertAlmostEqual(result[law]["variance_edges"], 1.0, places=12)

    def test_disjoint_uncertain_edges(self):
        positions = np.array([[0.0, 0.0], [2.0, 0.0], [100.0, 0.0], [102.0, 0.0]])
        radii = np.ones(4)
        varied = np.array([True, False, True, False])
        geometry = frame_geometry(positions, radii, varied, 0.2, 0.0)
        result = exact_edge_moments(geometry, 0.2)
        self.assertEqual(result["uncertain_edges"], 2)
        self.assertEqual(result["independent_uniform"]["covariance_pairs"], 0)
        self.assertAlmostEqual(result["independent_uniform"]["mean_edges"], 1.0, places=12)
        self.assertAlmostEqual(result["independent_uniform"]["variance_edges"], 0.5, places=12)
        self.assertAlmostEqual(result["common_uniform"]["variance_edges"], 1.0, places=12)

    def test_shared_variable_with_two_random_neighbours(self):
        positions = np.array([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
        radii = np.ones(3)
        geometry = frame_geometry(positions, radii, np.ones(3, dtype=bool), 0.2, 0.0)
        result = exact_edge_moments(geometry, 0.2)
        self.assertEqual(result["uncertain_edges"], 2)
        self.assertAlmostEqual(result["independent_uniform"]["mean_edges"], 1.0, places=12)
        # Each edge has variance 1/4. Conditional on the shared middle-node
        # radius, E[p(U)^2]=1/3, so the covariance is 1/12.
        self.assertAlmostEqual(result["independent_uniform"]["variance_edges"], 2.0 / 3.0, places=12)
        self.assertAlmostEqual(result["common_uniform"]["variance_edges"], 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
