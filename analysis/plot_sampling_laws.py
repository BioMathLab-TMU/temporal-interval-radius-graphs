#!/usr/bin/env python3
"""Plot observed Monte Carlo spans against attainable interval endpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE = "#276FBF"
ORANGE = "#F28E2B"
INK = "#151A2D"


def plot(source: Path, output: Path) -> None:
    report = json.loads(source.read_text(encoding="utf-8"))
    if report["schema"] != "radius-baselines-2" or len(report["records"]) != 15:
        raise ValueError("expected the 15-frame sampling-law comparison")
    counts = [64, 1024]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), constrained_layout=True)
    for ax, measure in zip(axes, ("edge", "lcc")):
        for law, color, label in (("independent_uniform", BLUE, "Independent node radii"),
                                  ("common_uniform", ORANGE, "Common scale factor")):
            arrays = []
            for count in counts:
                values = []
                for row in report["records"]:
                    sample = row["sampling"][law]["summaries"][str(count)]
                    if measure == "edge":
                        denominator = row["possible_edges"] - row["guaranteed_edges"]
                        value = ((sample["edge_max"] - sample["edge_min"]) / denominator
                                 if denominator else np.nan)
                    else:
                        value = sample["lcc_span_fraction_of_exact"]
                    values.append(value)
                arrays.append(np.asarray(values, dtype=float))
            medians = [np.nanmedian(values) for values in arrays]
            q1 = [np.nanpercentile(values, 25) for values in arrays]
            q3 = [np.nanpercentile(values, 75) for values in arrays]
            ax.errorbar(counts, medians,
                        yerr=[np.asarray(medians) - q1, np.asarray(q3) - medians],
                        marker="o", markersize=4, color=color, linewidth=1.7,
                        capsize=3, label=label)
        ax.axhline(1, color="#7A8194", linestyle="--", linewidth=0.8)
        ax.set_xticks(counts, [str(count) for count in counts])
        ax.set_xlim(20, 1068)
        ax.set_ylim(0, 1.06)
        ax.set_xlabel("Nested Monte Carlo draws per frame")
        ax.grid(axis="y", color="#DDE1EA", linewidth=0.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=INK, labelsize=8)
        ax.set_ylabel("Sampled span / exact endpoint span")
    axes[0].set_title("a  Edge count", loc="left", color=INK, fontsize=9, fontweight="bold")
    axes[1].set_title("b  Largest-component fraction", loc="left", color=INK, fontsize=9, fontweight="bold")
    axes[1].legend(frameon=False, fontsize=7.5, loc="lower right")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/derived/radius_baseline_comparison.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.source, args.output)
    print(f"WROTE {args.output}")
