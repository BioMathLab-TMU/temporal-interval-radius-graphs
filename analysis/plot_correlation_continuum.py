#!/usr/bin/env python3
"""Plot exploratory radius-dependence diagnostics from derived JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def quantile_band(records, rho, observable, key):
    y = np.asarray([r["rho_results"][str(rho)][observable][key]
                    for r in records], dtype=float)
    return np.quantile(y, [0.25, 0.5, 0.75])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.source.read_text(encoding="utf-8"))
    if report["schema"] != "correlated-radius-continuum-1":
        raise ValueError("unexpected source schema")
    rho = np.asarray(report["rho_grid"])
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.1), constrained_layout=True)
    colors = {"archive": "#24476b", "synthetic": "#bd6b32"}
    for group in ("archive", "synthetic"):
        records = report[group]
        q = np.asarray([quantile_band(records, x, "edge_count", "sd_over_exact_span") for x in rho])
        axes[0].plot(rho, q[:, 1], marker="o", label=group, color=colors[group])
        axes[0].fill_between(rho, q[:, 0], q[:, 2], alpha=0.16, color=colors[group])
        e = np.asarray([quantile_band(records, x, "edge_count", "observed_over_linearized_prediction") for x in rho])
        axes[1].plot(rho, e[:, 1], marker="o", label=group, color=colors[group])
        axes[1].fill_between(rho, e[:, 0], e[:, 2], alpha=0.16, color=colors[group])
    for stage, color in (("sparse", "#3d8c70"), ("transition", "#9a4d86"), ("dense", "#bd6b32")):
        records = [r for r in report["synthetic"] if r["finite_size_stage"] == stage]
        s = np.asarray([quantile_band(records, x, "largest_component_fraction", "sd") for x in rho])
        axes[2].plot(rho, s[:, 1], marker="o", label=stage, color=color)
        axes[2].fill_between(rho, s[:, 0], s[:, 2], alpha=0.16, color=color)
        q = np.asarray([quantile_band(records, x, "largest_component_fraction", "observed_over_linearized_prediction") for x in rho])
        axes[3].plot(rho, q[:, 1], marker="o", label=stage, color=color)
        axes[3].fill_between(rho, q[:, 0], q[:, 2], alpha=0.16, color=color)
    axes[0].set_title("A  Edge-count dispersion")
    axes[0].set_ylabel("SD / exact endpoint span")
    axes[1].set_title("B  Edge-count variance law")
    axes[2].set_title("C  Component sensitivity")
    axes[2].set_ylabel("Largest-component SD")
    axes[3].set_title("D  Component variance law")
    for ax in (axes[1], axes[3]):
        ax.axhline(1.0, color="0.35", ls="--", lw=1)
        ax.set_ylabel("Observed SD / interpolated SD")
    for ax in axes:
        ax.set_xlabel("Pairwise radius-factor correlation ρ")
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
