#!/usr/bin/env python3
"""Verify checksums, tests, and the internal contract of this release."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest() -> None:
    manifest = ROOT / "MANIFEST.sha256"
    failures: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append(f"checksum mismatch: {relative}")
    if failures:
        raise RuntimeError("manifest verification failed:\n" + "\n".join(failures))
    print("MANIFEST_OK")


def run_tests() -> None:
    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_*.py",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    print("TESTS_OK")


def main() -> None:
    verify_manifest()
    run_tests()
    print("RELEASE_VERIFICATION_OK")


if __name__ == "__main__":
    main()
