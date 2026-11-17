#!/usr/bin/env python3
"""Small exactness checks for the independent benchmark reference."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from benchmark_pair_search import all_pairs_search, tree_search  # noqa: E402


class PairSearchBenchmarkTests(unittest.TestCase):
    def test_heterogeneous_random_geometry(self) -> None:
        rng = np.random.default_rng(20261010)
        positions = rng.uniform(-20, 20, size=(100, 3))
        lower = rng.uniform(0.5, 2.0, size=100)
        upper = lower + rng.uniform(0, 0.8, size=100)
        for threshold in (-0.5, 0.0, 2.0):
            tree = tree_search(positions, lower, upper, threshold)
            all_pairs = all_pairs_search(positions, lower, upper, threshold)
            np.testing.assert_array_equal(tree[1], all_pairs[1])
            np.testing.assert_array_equal(tree[2], all_pairs[2])

    def test_empty_candidate_search(self) -> None:
        positions = np.array([[0.0, 0.0], [50.0, 0.0]])
        lower = np.array([1.0, 1.0])
        upper = np.array([1.5, 1.5])
        tree = tree_search(positions, lower, upper, 0.0)
        all_pairs = all_pairs_search(positions, lower, upper, 0.0)
        self.assertEqual(tree[0], 0)
        np.testing.assert_array_equal(tree[1], all_pairs[1])
        np.testing.assert_array_equal(tree[2], all_pairs[2])


if __name__ == "__main__":
    unittest.main()
