#!/usr/bin/env python3
"""Plot fixed-threshold finite-size holdout diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STAGES = (("sparse", "#3d8c70"), ("transition", "#9a4d86"), ("dense", "#bd6b32"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rho", type=float, default=0.25)
    args = parser.parse_args()
    data = json.loads(args.source.read_text(encoding="utf-8"))
    if data["schema"] != "finite-size-correlation-holdout-1":
        raise ValueError("unexpected source schema")
    rho_key = str(args.rho)
    sizes = data["target_counts"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), constrained_layout=True)
    for stage, color in STAGES:
        lcc_q, sd_q = [], []
        for n in sizes:
            records = [r for r in data["records"] if r["target_n"] == n and r["stage"] == stage]
            lcc_q.append(np.quantile([r["nominal_lcc"] for r in records], [0.25, 0.5, 0.75]))
            sd_q.append(np.quantile([
                r["sampled"]["rho_results"][rho_key]["largest_component_fraction"]["sd"]
                for r in records], [0.25, 0.5, 0.75]))
        for ax, quantiles in ((axes[0], np.asarray(lcc_q)), (axes[1], np.asarray(sd_q))):
            ax.plot(sizes, quantiles[:, 1], marker="o", color=color, label=stage)
            ax.fill_between(sizes, quantiles[:, 0], quantiles[:, 2], alpha=0.14, color=color)
    peak_counts = []
    for n in sizes:
        count = 0
        for seed in data["holdout_seeds"]:
            cases = {r["stage"]: r for r in data["records"] if r["target_n"] == n and r["point_seed"] == seed}
            sd = lambda stage: cases[stage]["sampled"]["rho_results"][rho_key]["largest_component_fraction"]["sd"]
            count += sd("transition") > max(sd("sparse"), sd("dense"))
        peak_counts.append(count)
    axes[2].bar([str(n) for n in sizes], peak_counts, color="#9a4d86", width=0.6)
    axes[2].set_ylim(0, len(data["holdout_seeds"]) + 1)
    axes[2].set_yticks(range(0, len(data["holdout_seeds"]) + 1, 2))
    axes[0].set_title("A  Nominal connectivity")
    axes[0].set_ylabel("Largest-component fraction")
    axes[1].set_title(f"B  Size sensitivity at ρ={args.rho:g}")
    axes[1].set_ylabel("Largest-component SD")
    axes[2].set_title("C  Middle regime most variable")
    axes[2].set_ylabel("Point-process seeds / 10")
    for ax in axes[:2]:
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(n) for n in sizes])
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.18)
    for ax in axes:
        ax.set_xlabel("Target point count")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
