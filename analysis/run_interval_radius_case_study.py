#!/usr/bin/env python3
"""Analyse completed article trajectories as interval-radius temporal graphs.

The particle trajectories are treated as fixed observed inputs.  At each saved
time, selected node radii are allowed to vary inside declared intervals.  The
script reports exact guaranteed/possible graph endpoints; it does not alter or
rerun the physical simulation and it does not infer a stochastic radius law.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPOSITORY_ROOT / "src"
sys.path.insert(0, str(SOURCE_DIR))

from interval_radius_graph import build_interval_radius_graph  # noqa: E402


STEP_PATTERN = re.compile(r"^step_(\d+)$")
SYNAPSIN_TYPE = 1
PALETTE = {
    "ink": "#151A2D",
    "blue": "#276FBF",
    "cyan": "#18A7B5",
    "orange": "#F28E2B",
    "red": "#D1495B",
    "grey": "#7A8194",
    "light": "#DDE1EA",
}


def scalar(group: h5py.Group, key: str) -> float | int:
    return np.asarray(group[key][...]).reshape(-1)[0].item()


def relative_path(path: Path, base: Path) -> str:
    """Return a portable path without recording a contributor's home path."""

    return path.resolve().relative_to(base.resolve()).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def complete_step_groups(handle: h5py.File) -> list[tuple[int, str]]:
    groups: list[tuple[int, str]] = []
    for name in handle.keys():
        match = STEP_PATTERN.match(name)
        if not match:
            continue
        group = handle[name]
        if "frame_write_complete" in group and int(scalar(group, "frame_write_complete")) != 1:
            continue
        groups.append((int(match.group(1)), name))
    return sorted(groups)


def inspect_hdf5(path: Path, input_root: Path, source_prefix: str) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        metadata = handle["run_metadata"]
        completion = handle["run_completion"]
        frames = complete_step_groups(handle)
        run_id = path.parent.name.split("----", 1)[0]
        return {
            "run_id": run_id,
            "path": path,
            "relative_path": f"{source_prefix}/{relative_path(path, input_root)}",
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "run_completed": int(scalar(completion, "run_completed")),
            "all_expected_frames_saved": int(scalar(completion, "all_expected_frames_saved")),
            "frame_count": len(frames),
            "expected_frame_count": int(scalar(metadata, "expected_saved_frame_count")),
            "particle_count": int(scalar(metadata, "particle_count")),
            "synapsin_count": int(scalar(metadata, "synapsin_count")),
            "vesicle_count": int(scalar(metadata, "vesicle_count")),
            "delta_t_ms": float(scalar(metadata, "delta_t_ms")),
            "final_time_ms": float(scalar(completion, "final_physical_time_ms")),
        }


def discover_sources(
    input_root: Path,
    source_prefix: str,
    manifest_path: Path,
    run_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No raw sources in manifest: {manifest_path}")
    inspected: list[dict[str, Any]] = []
    for row in rows:
        if run_ids is not None and row["run_id"] not in run_ids:
            continue
        relative = PurePosixPath(row["relative_path"])
        if (
            not relative.parts
            or relative.parts[0] != source_prefix
            or ".." in relative.parts
        ):
            raise ValueError(f"Unsafe or mismatched source path: {relative}")
        path = input_root.joinpath(*relative.parts[1:])
        if not path.is_file():
            raise FileNotFoundError(path)
        source = inspect_hdf5(path, input_root, source_prefix)
        if source["run_id"] != row["run_id"]:
            raise ValueError(f"Run ID mismatch for {path}")
        if source["bytes"] != int(row["bytes"]):
            raise ValueError(f"Byte-size mismatch for {path}")
        if source["sha256"] != row["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {path}")
        inspected.append(source)
    if run_ids is not None:
        missing = run_ids - {item["run_id"] for item in inspected}
        if missing:
            raise ValueError(f"Run IDs absent from manifest: {sorted(missing)}")
    complete: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in inspected:
        reasons: list[str] = []
        if source["run_id"] in seen:
            reasons.append("duplicate_run_id")
        if source["run_completed"] != 1:
            reasons.append("run_not_completed")
        if source["all_expected_frames_saved"] != 1:
            reasons.append("not_all_expected_frames_saved")
        if source["frame_count"] != source["expected_frame_count"]:
            reasons.append("frame_count_mismatch")
        if reasons:
            excluded.append({**source, "exclusion_reasons": reasons})
        else:
            complete.append(source)
            seen.add(source["run_id"])
    return complete, excluded


def choose_even_frames(
    frames: list[tuple[int, str]],
    samples_per_run: int,
) -> list[tuple[int, str]]:
    if len(frames) <= samples_per_run:
        return frames
    indices = np.linspace(0, len(frames) - 1, samples_per_run, dtype=int)
    return [frames[index] for index in np.unique(indices)]


def radius_interval(
    nominal_radii: np.ndarray,
    node_types: np.ndarray,
    relative_half_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a symmetric declared interval to Synapsin nodes only."""

    lower = nominal_radii.astype(np.float64, copy=True)
    upper = nominal_radii.astype(np.float64, copy=True)
    selected = node_types == SYNAPSIN_TYPE
    lower[selected] *= 1.0 - relative_half_width
    upper[selected] *= 1.0 + relative_half_width
    return lower, upper


def analyse_frame(
    *,
    source: dict[str, Any],
    group: h5py.Group,
    step: int,
    sampled_frame_index: int,
    relative_half_width: float,
    threshold_nm: float,
) -> dict[str, Any]:
    positions = group["positions"][...]
    nominal_radii = group["radii"][...]
    node_types = group["types"][...]
    lower, upper = radius_interval(nominal_radii, node_types, relative_half_width)
    graph = build_interval_radius_graph(
        positions,
        lower,
        upper,
        threshold_nm,
    )
    time_ms = (
        float(scalar(group, "physical_time_ms"))
        if "physical_time_ms" in group
        else step * source["delta_t_ms"]
    )
    return {
        "source_file": source["relative_path"],
        "run_id": source["run_id"],
        "simulation_step": step,
        "time_ms": time_ms,
        "sampled_frame_index": sampled_frame_index,
        "relative_radius_half_width": relative_half_width,
        "synapsin_radius_lower_nm": float(np.min(lower[node_types == SYNAPSIN_TYPE])),
        "synapsin_radius_upper_nm": float(np.max(upper[node_types == SYNAPSIN_TYPE])),
        "varied_node_type": SYNAPSIN_TYPE,
        **graph.to_record(),
    }


def validate_frame_identity(
    group: h5py.Group,
    reference_ids: np.ndarray | None,
    reference_types: np.ndarray | None,
    source_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    particle_ids = group["ids"][...]
    node_types = group["types"][...]
    positions = group["positions"][...]
    radii = group["radii"][...]
    if len(np.unique(particle_ids)) != len(particle_ids):
        raise ValueError(f"Duplicate IDs in {source_path}:{group.name}")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(radii)):
        raise ValueError(f"Non-finite geometry in {source_path}:{group.name}")
    if np.any(radii <= 0.0):
        raise ValueError(f"Non-positive radii in {source_path}:{group.name}")
    if reference_ids is not None and not np.array_equal(particle_ids, reference_ids):
        raise ValueError(f"Particle IDs changed in {source_path}:{group.name}")
    if reference_types is not None and not np.array_equal(node_types, reference_types):
        raise ValueError(f"Particle types changed in {source_path}:{group.name}")
    return particle_ids.copy(), node_types.copy()


def analyse_sources(
    sources: list[dict[str, Any]],
    *,
    relative_half_widths: Iterable[float],
    thresholds_nm: Iterable[float],
    primary_threshold_nm: float,
    samples_per_run: int,
) -> pd.DataFrame:
    widths = sorted(set(float(value) for value in relative_half_widths))
    thresholds = sorted(set(float(value) for value in thresholds_nm))
    maximum_width = max(widths)
    conditions = {(width, primary_threshold_nm) for width in widths}
    conditions.update((maximum_width, threshold) for threshold in thresholds)
    rows: list[dict[str, Any]] = []
    for source_index, source in enumerate(sources, start=1):
        print(f"ANALYSE_RUN {source_index}/{len(sources)} {source['run_id']}", flush=True)
        with h5py.File(source["path"], "r") as handle:
            frames = choose_even_frames(complete_step_groups(handle), samples_per_run)
            reference_ids: np.ndarray | None = None
            reference_types: np.ndarray | None = None
            previous_time = -np.inf
            for sampled_frame_index, (step, name) in enumerate(frames):
                group = handle[name]
                reference_ids, reference_types = validate_frame_identity(
                    group, reference_ids, reference_types, source["path"]
                )
                time_ms = float(scalar(group, "physical_time_ms"))
                if time_ms <= previous_time:
                    raise ValueError(f"Sampled times are not increasing in {source['path']}")
                previous_time = time_ms
                for width, threshold in sorted(conditions):
                    rows.append(
                        analyse_frame(
                            source=source,
                            group=group,
                            step=step,
                            sampled_frame_index=sampled_frame_index,
                            relative_half_width=width,
                            threshold_nm=threshold,
                        )
                    )
    return pd.DataFrame(rows)


def trajectory_summary(frame_table: pd.DataFrame) -> pd.DataFrame:
    grouping = ["run_id", "relative_radius_half_width", "threshold_nm"]
    return (
        frame_table.groupby(grouping, dropna=False)
        .agg(
            sampled_frames=("simulation_step", "size"),
            particle_count=("n_nodes", "first"),
            mean_guaranteed_edge_count=("guaranteed_edge_count", "mean"),
            mean_possible_edge_count=("possible_edge_count", "mean"),
            mean_radius_dependent_edge_count=("radius_dependent_edge_count", "mean"),
            maximum_radius_dependent_edge_count=("radius_dependent_edge_count", "max"),
            mean_radius_dependent_fraction=("radius_dependent_fraction_of_possible_edges", "mean"),
            maximum_largest_component_fraction_width=("largest_component_fraction_width", "max"),
            mean_largest_component_fraction_width=("largest_component_fraction_width", "mean"),
            maximum_component_count_width=("component_count_width", "max"),
            mean_component_count_width=("component_count_width", "mean"),
        )
        .reset_index()
        .sort_values(grouping)
    )


def draw_schematic(axis: plt.Axes) -> None:
    axis.set_aspect("equal")
    axis.axis("off")
    centres = [(0.0, 0.0), (2.1, 0.0), (4.9, 0.0)]
    lower = [0.7, 0.7, 0.7]
    upper = [1.0, 1.0, 1.0]
    for index, ((x, y), r0, r1) in enumerate(zip(centres, lower, upper), start=1):
        axis.add_patch(Circle((x, y), r1, facecolor=PALETTE["cyan"], alpha=0.12, edgecolor=PALETTE["cyan"], linestyle="--", linewidth=1.3))
        axis.add_patch(Circle((x, y), r0, facecolor="white", edgecolor=PALETTE["ink"], linewidth=1.3))
        axis.plot(x, y, "o", color=PALETTE["ink"], markersize=3)
        axis.text(x, -1.35, rf"$v_{index}$", ha="center", fontsize=9)
    axis.plot([centres[0][0], centres[1][0]], [0, 0], color=PALETTE["blue"], linewidth=2.5)
    axis.plot([centres[1][0], centres[2][0]], [0, 0], color=PALETTE["orange"], linewidth=2.5, linestyle="--")
    axis.text(1.05, 1.25, "guaranteed", color=PALETTE["blue"], ha="center", fontsize=8)
    axis.text(3.5, 1.25, "radius-dependent", color=PALETTE["orange"], ha="center", fontsize=8)
    axis.set_xlim(-1.2, 6.1)
    axis.set_ylim(-1.6, 1.7)
    axis.set_title("a", loc="left", fontweight="bold")


def make_figure(frame_table: pd.DataFrame, output: Path, primary_threshold_nm: float) -> None:
    primary = frame_table[np.isclose(frame_table["threshold_nm"], primary_threshold_nm)].copy()
    widths = sorted(primary["relative_radius_half_width"].unique())
    run_means = (
        primary.groupby(["run_id", "relative_radius_half_width"], as_index=False)
        .agg(
            radius_dependent_fraction_of_possible_edges=(
                "radius_dependent_fraction_of_possible_edges",
                "mean",
            ),
            largest_component_fraction_width=(
                "largest_component_fraction_width",
                "mean",
            ),
        )
    )
    grouped_runs = run_means.groupby("relative_radius_half_width")
    grouped_frames = primary.groupby("relative_radius_half_width")

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    draw_schematic(axes[0, 0])
    for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color=PALETTE["light"], linewidth=0.7, alpha=0.8)

    median = grouped_runs["radius_dependent_fraction_of_possible_edges"].median().reindex(widths)
    q1 = grouped_runs["radius_dependent_fraction_of_possible_edges"].quantile(0.25).reindex(widths)
    q3 = grouped_runs["radius_dependent_fraction_of_possible_edges"].quantile(0.75).reindex(widths)
    x = np.asarray(widths) * 100.0
    axes[0, 1].plot(x, median, marker="o", color=PALETTE["blue"], linewidth=2)
    axes[0, 1].fill_between(x, q1, q3, color=PALETTE["blue"], alpha=0.2)
    axes[0, 1].set_xlabel("Synapsin radius half-width (%)")
    axes[0, 1].set_ylabel("Radius-dependent / possible edges")
    axes[0, 1].set_title("b", loc="left", fontweight="bold")

    component_width = grouped_runs["largest_component_fraction_width"].median().reindex(widths)
    component_max = grouped_frames["largest_component_fraction_width"].max().reindex(widths)
    axes[1, 0].plot(x, component_width, marker="o", color=PALETTE["orange"], linewidth=2, label="median run mean")
    axes[1, 0].plot(x, component_max, marker="s", color=PALETTE["red"], linewidth=1.5, label="maximum frame")
    axes[1, 0].set_xlabel("Synapsin radius half-width (%)")
    axes[1, 0].set_ylabel("Largest-component fraction interval width")
    axes[1, 0].set_title("c", loc="left", fontweight="bold")
    axes[1, 0].legend(frameon=False, fontsize=8)

    representative_run = sorted(primary["run_id"].unique())[0]
    representative = primary[
        (primary["run_id"] == representative_run)
        & np.isclose(primary["relative_radius_half_width"], max(widths))
    ].sort_values("time_ms")
    axes[1, 1].fill_between(
        representative["time_ms"],
        representative["component_count_lower"],
        representative["component_count_upper"],
        color=PALETTE["cyan"],
        alpha=0.3,
        label="exact interval",
    )
    axes[1, 1].plot(representative["time_ms"], representative["component_count_lower"], color=PALETTE["blue"], linewidth=1.6)
    axes[1, 1].plot(representative["time_ms"], representative["component_count_upper"], color=PALETTE["orange"], linewidth=1.6)
    axes[1, 1].set_xlabel("Saved physical time (ms)")
    axes[1, 1].set_ylabel("Number of connected components")
    axes[1, 1].set_title("d", loc="left", fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(fig)


def serialisable_source(source: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in source.items() if key != "path"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "raw",
        help="Root containing article_s1_* directories with HDF5 trajectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "reproduced",
    )
    parser.add_argument(
        "--source-prefix",
        default="builds",
        help="Portable prefix retained in the published source_file column.",
    )
    parser.add_argument(
        "--provenance-manifest",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "provenance" / "RUN_PROVENANCE.csv",
        help="Exact list and SHA-256 digests of raw files used in the manuscript.",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        default=None,
        help="Limit processing to this run ID; may be specified more than once.",
    )
    parser.add_argument("--relative-half-widths", type=float, nargs="+", default=[0.0, 0.025, 0.05, 0.10])
    parser.add_argument("--thresholds-nm", type=float, nargs="+", default=[0.0, 2.0, 4.0, 6.0])
    parser.add_argument("--primary-threshold-nm", type=float, default=2.0)
    parser.add_argument("--samples-per-run", type=int, default=6)
    args = parser.parse_args()
    if not args.source_prefix or "/" in args.source_prefix or "\\" in args.source_prefix:
        raise ValueError("source-prefix must be one nonempty path component")
    if args.samples_per_run < 2:
        raise ValueError("samples-per-run must be at least 2")
    if any(value < 0.0 or value >= 1.0 for value in args.relative_half_widths):
        raise ValueError("relative half-widths must be in [0, 1)")

    requested_run_ids = set(args.run_id) if args.run_id else None
    sources, excluded = discover_sources(
        args.input_root,
        args.source_prefix,
        args.provenance_manifest,
        requested_run_ids,
    )
    if not sources:
        raise RuntimeError(f"No complete article HDF5 sources found under {args.input_root}")
    frame_table = analyse_sources(
        sources,
        relative_half_widths=args.relative_half_widths,
        thresholds_nm=args.thresholds_nm,
        primary_threshold_nm=args.primary_threshold_nm,
        samples_per_run=args.samples_per_run,
    )
    summary_table = trajectory_summary(frame_table)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = args.output_dir / "interval_radius_frame_metrics.csv"
    summary_path = args.output_dir / "interval_radius_trajectory_summary.csv"
    figure_path = args.output_dir / "interval_radius_graph_figure.png"
    metadata_path = args.output_dir / "interval_radius_case_study.json"
    frame_table.to_csv(frame_path, index=False)
    summary_table.to_csv(summary_path, index=False)
    make_figure(frame_table, figure_path, args.primary_threshold_nm)

    missing_output_directories = []
    for directory in sorted(args.input_root.glob("article_s1_*")):
        if not list(directory.glob("simulation_Rawdata*.h5")):
            missing_output_directories.append(
                f"{args.source_prefix}/{relative_path(directory, args.input_root)}"
            )
    metadata = {
        "schema_version": "simsv-interval-radius-case-study-1.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_scope": {
            "positions": "fixed saved coordinates at each analysed frame",
            "radius_model": "independent interval-valued radii at each saved time",
            "varied_species": "Synapsin (type 1) only in this case study",
            "relative_half_widths": sorted(set(args.relative_half_widths)),
            "thresholds_nm": sorted(set(args.thresholds_nm)),
            "primary_threshold_nm": args.primary_threshold_nm,
            "samples_per_run": args.samples_per_run,
            "interpretation": "prescribed interval envelopes evaluated independently at each observation time",
        },
        "data_quality": {
            "complete_hdf5_sources": len(sources),
            "excluded_hdf5_sources": len(excluded),
            "available_frames": int(sum(source["frame_count"] for source in sources)),
            "analysed_unique_run_frame_pairs": int(frame_table[["run_id", "simulation_step"]].drop_duplicates().shape[0]),
            "all_selected_runs_completed": all(source["run_completed"] == 1 for source in sources),
            "all_expected_frames_saved": all(source["all_expected_frames_saved"] == 1 for source in sources),
            "missing_hdf5_output_directories": missing_output_directories,
        },
        "included_sources": [serialisable_source(source) for source in sources],
        "excluded_sources": [serialisable_source(source) for source in excluded],
        "outputs": {
            "frame_metrics": frame_path.name,
            "trajectory_summary": summary_path.name,
            "figure": figure_path.name,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"WROTE {frame_path}")
    print(f"WROTE {summary_path}")
    print(f"WROTE {figure_path}")
    print(f"WROTE {metadata_path}")


if __name__ == "__main__":
    main()
