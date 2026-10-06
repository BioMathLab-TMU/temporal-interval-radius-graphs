#!/usr/bin/env python3
"""Regenerate checksums for small archival artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INCLUDED = [
    "data/derived/interval_radius_case_study.json",
    "data/derived/interval_radius_frame_metrics.csv",
    "data/derived/interval_radius_trajectory_summary.csv",
    "data/provenance/RUN_PROVENANCE.csv",
    "src/interval_radius_graph.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    lines = ["# SHA-256 manifest for the archived release artifacts."]
    for relative in INCLUDED:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        lines.append(f"{sha256(path)}  {relative}")
    with (ROOT / "MANIFEST.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    print("WROTE MANIFEST.sha256")


if __name__ == "__main__":
    main()
