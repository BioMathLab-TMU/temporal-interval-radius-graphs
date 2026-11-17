#!/usr/bin/env python3
"""Apply interval-radius graph summaries to independent BBBC039 nuclei masks.

The mask archive is the official CC0 BBBC039v1 source. Pixel-area-equivalent
radii are analysis choices; interval widths are sensitivity scenarios.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass, label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from interval_radius_graph import build_interval_radius_graph  # noqa: E402

SOURCE_SHA256 = "f9e6043d8ca56344a4886f96a700d804d6ee982f31e2b2cd3194af2a053c2710"


def objects_from_mask(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(Image.open(io.BytesIO(payload)))
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("BBBC039 mask is not RGB/RGBA")
    channel = image[:, :, 0]
    centres: list[tuple[float, float]] = []
    radii: list[float] = []
    # Distinct nonzero color values can touch. Label each value separately,
    # as in the BBBC039 decoding example, rather than merging all foreground.
    for value in np.unique(channel):
        if value == 0:
            continue
        labels, count = label(channel == value, structure=np.ones((3, 3), dtype=np.uint8))
        if count == 0:
            continue
        areas = np.bincount(labels.ravel(), minlength=count + 1)[1:]
        positions = center_of_mass(np.ones(channel.shape), labels, range(1, count + 1))
        for (y, x), area in zip(positions, areas):
            if area > 0:
                centres.append((float(x), float(y)))
                radii.append(float(np.sqrt(area / np.pi)))
    return np.asarray(centres, dtype=np.float64).reshape(-1, 2), np.asarray(radii)


def analyze(source: Path) -> dict:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("official BBBC039 mask archive SHA-256 mismatch")
    records = []
    with zipfile.ZipFile(source) as archive:
        names = sorted(name for name in archive.namelist()
                       if name.startswith("masks/") and name.lower().endswith(".png"))
        if len(names) != 200:
            raise ValueError(f"expected 200 masks; found {len(names)}")
        for name in names:
            positions, radii = objects_from_mask(archive.read(name))
            record = {"mask": name, "n_objects": len(radii), "thresholds": []}
            if len(radii) >= 2:
                for threshold in (0.0, 2.0):
                    graph = build_interval_radius_graph(
                        positions, radii * .9, radii * 1.1, threshold)
                    record["thresholds"].append({
                        "threshold_px": threshold,
                        "guaranteed_edges": len(graph.guaranteed_pairs),
                        "possible_edges": len(graph.possible_pairs),
                        "guaranteed_components": graph.guaranteed_summary["n_components"],
                        "possible_components": graph.possible_summary["n_components"],
                        "guaranteed_lcc": graph.guaranteed_summary["largest_component_fraction"],
                        "possible_lcc": graph.possible_summary["largest_component_fraction"],
                    })
            records.append(record)
    return {
        "schema": "bbbc039-radius-graph-1", "source": "BBBC039v1 masks.zip",
        "source_url": "https://data.broadinstitute.org/bbbc/BBBC039/masks.zip",
        "source_sha256": digest, "source_license": "CC0",
        "image_count": len(records), "radius_definition": "sqrt(pixel_area/pi)",
        "radius_half_width": 0.10, "thresholds_px": [0.0, 2.0],
        "mask_decoding": "8-connected components separately for each nonzero red-channel label",
        "records": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masks-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data/derived/bbbc039_radius_graph.json")
    args = parser.parse_args()
    result = analyze(args.masks_zip)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"BBBC039_OK fields={result['image_count']} objects={sum(r['n_objects'] for r in result['records'])}")
