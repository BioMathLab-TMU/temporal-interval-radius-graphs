#!/usr/bin/env python3
"""Render a two-panel structural audit figure from the derived JSON."""

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
GREY = "#7A8194"


def plot(source: Path, output: Path) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))
    if data["schema"] != "critical-width-analysis-1" or len(data["records"]) != 15:
        raise ValueError("expected the validated 15-run critical-width audit")
    records = sorted(data["records"], key=lambda row: (row["n_nodes"], row["run_id"]))
    sizes = [3713, 7427, 14853]
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.25), constrained_layout=True)
    for group_index, n in enumerate(sizes):
        group = [row for row in records if row["n_nodes"] == n]
        offsets = np.linspace(-0.12, 0.12, len(group))
        for offset, row in zip(offsets, group):
            quotient = row["quotient_at_0p1"]
            inside = quotient["ambiguous_edges_inside_guaranteed_components"]
            outside = quotient["ambiguous_edges_between_guaranteed_components"]
            cross_pct = 100 * outside / (inside + outside)
            axes[0].scatter(group_index + offset, cross_pct, s=27, color=BLUE,
                            edgecolor="white", linewidth=0.45, zorder=3)
            possible_pct = 100 * row["possible_component_merger_events"] / row["possible_edge_birth_events"]
            guaranteed_pct = 100 * row["guaranteed_component_split_events"] / row["guaranteed_edge_exit_events"]
            axes[1].scatter(group_index + offset - 0.055, possible_pct, s=25, color=BLUE,
                            marker="o", edgecolor="white", linewidth=0.35, zorder=3)
            axes[1].scatter(group_index + offset + 0.055, guaranteed_pct, s=25, color=ORANGE,
                            marker="s", edgecolor="white", linewidth=0.35, zorder=3)
    for ax in axes:
        ax.set_xlim(-0.4, 2.4)
        ax.set_xticks(range(3), [f"{n:,}\n({[3,9,3][i]} runs)" for i, n in enumerate(sizes)])
        ax.set_xlabel("Nodes per frame")
        ax.set_ylim(0, 4)
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.grid(axis="y", color="#DDE1EA", linewidth=0.55, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=INK, labelsize=7.5)
        ax.xaxis.label.set_color(INK)
        ax.yaxis.label.set_color(INK)
    axes[0].set_title("a  Cross-component ambiguity", loc="left", color=INK, fontsize=9, fontweight="bold")
    axes[0].set_ylabel("Cross-component edges (%)")
    axes[1].set_title("b  Events changing component count", loc="left", color=INK, fontsize=9, fontweight="bold")
    axes[1].set_ylabel("Component-changing edge events (%)")
    axes[1].scatter([], [], color=BLUE, marker="o", s=25, label="Possible-graph mergers")
    axes[1].scatter([], [], color=ORANGE, marker="s", s=25, label="Guaranteed-graph splits")
    axes[1].legend(frameon=False, fontsize=7, loc="upper right")
    for ax in axes:
        ax.set_axisbelow(True)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/derived/critical_widths.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.source, args.output)
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
