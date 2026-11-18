"""Checks for the exploratory correlation-continuum analysis."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from correlated_radius_continuum import (  # noqa: E402
    analyse_geometry, frame_geometry, uniform_factors,
)


class CopulaTests(unittest.TestCase):
    def test_uniform_support_and_target_correlation(self):
        rng = np.random.default_rng(47)
        epsilon = rng.standard_normal((50000, 2))
        common = rng.standard_normal(50000)
        for rho in (0.0, 0.25, 0.5, 0.9, 1.0):
            factors = uniform_factors(epsilon, common, rho)
            self.assertTrue(np.all((-1.0 <= factors) & (factors <= 1.0)))
            self.assertAlmostEqual(float(np.corrcoef(factors.T)[0, 1]), rho, delta=0.015)
            self.assertAlmostEqual(float(factors[:, 0].mean()), 0.0, delta=0.015)
        np.testing.assert_allclose(
            uniform_factors(epsilon, common, 1.0)[:, 0],
            uniform_factors(epsilon, common, 1.0)[:, 1], atol=1e-12,
        )

    def test_invalid_rho(self):
        with self.assertRaises(ValueError):
            uniform_factors(np.zeros((2, 2)), np.zeros(2), -0.1)

    def test_sampled_graphs_remain_within_exact_endpoints(self):
        positions = np.array([[0.0, 0.0], [1.9, 0.0], [3.8, 0.0], [5.7, 0.0]])
        radii = np.ones(4)
        geometry = frame_geometry(positions, radii, np.ones(4, dtype=bool), 0.2, 0.0)
        report = analyse_geometry(geometry, seed=19, draws=128, width=0.2)
        for entry in report["rho_results"].values():
            self.assertGreaterEqual(entry["edge_count"]["min"], geometry["base_edges"])
            self.assertLessEqual(entry["edge_count"]["max"], geometry["upper_edges"])
            self.assertGreaterEqual(entry["largest_component_fraction"]["min"], geometry["lower_lcc"])
            self.assertLessEqual(entry["largest_component_fraction"]["max"], geometry["upper_lcc"])


if __name__ == "__main__":
    unittest.main()
