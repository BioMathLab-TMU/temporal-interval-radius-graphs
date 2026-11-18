#!/usr/bin/env python3
"""Check quotient-based sampling against full graph reconstruction."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "src"))

from compare_radius_uncertainty_baselines import compare_frame  # noqa: E402
from interval_radius_graph import _component_summary, build_interval_radius_graph  # noqa: E402


class SamplingLawsTests(unittest.TestCase):
    def test_sampled_quotient_matches_full_graph(self):
        rng = np.random.default_rng(106)
        positions = rng.uniform(0, 14, size=(32, 2))
        types = np.array([1] * 24 + [2] * 8)
        radii = np.where(types == 1, 1.7, rng.uniform(1.0, 2.0, size=32))
        width, threshold, seed = 0.10, 0.4, 317
        result = compare_frame(positions, radii, types, seed=seed,
                               draws=64, width=width, threshold=threshold)
        varied = types == 1
        lower = radii * (1 - width * varied)
        upper = radii * (1 + width * varied)
        graph = build_interval_radius_graph(positions, lower, upper, threshold)
        pairs = graph.candidate_pairs
        left, right = pairs[:, 0], pairs[:, 1]
        distances = np.linalg.norm(positions[left] - positions[right], axis=1)
        for law, law_seed in (("independent_uniform", seed),
                              ("common_uniform", seed + 1000000)):
            draw_rng = np.random.default_rng(law_seed)
            edge_counts, lcc, components = [], [], []
            for _ in range(64):
                if law == "independent_uniform":
                    sampled = draw_rng.uniform(lower, upper)
                else:
                    sampled = radii * (1 + width * varied * draw_rng.uniform(-1, 1))
                mask = distances <= sampled[left] + sampled[right] + threshold
                summary = _component_summary(len(radii), pairs[mask])
                edge_counts.append(int(np.count_nonzero(mask)))
                lcc.append(summary["largest_component_fraction"])
                components.append(summary["n_components"])
            observed = result["sampling"][law]["summaries"]["64"]
            self.assertEqual((min(edge_counts), max(edge_counts)),
                             (observed["edge_min"], observed["edge_max"]))
            self.assertEqual((min(lcc), max(lcc)),
                             (observed["lcc_min"], observed["lcc_max"]))
            self.assertEqual((min(components), max(components)),
                             (observed["components_min"], observed["components_max"]))
        self.assertTrue(result["type_pair_possible_edges_equal"])
        self.assertTrue(result["type_pair_guaranteed_edges_equal"])


if __name__ == "__main__":
    unittest.main()
