# Temporal interval-radius graphs

Code, derived data, and provenance for temporal interval-radius graph analysis.
The manuscript, supplementary text, publisher files, and all article figures
are deliberately excluded from this repository and will be submitted directly
to the journal.

This is the laboratory-hosted reproducibility repository. The mathematical
formulation, analysis software, simulations/data analysis, and visualizations
were developed by [Shokoofeh Akbari](https://github.com/shokoofe-akbari),
with supervision and methodological input from Yousef Jamali. See
[`AUTHORS.md`](AUTHORS.md) and [`CITATION.cff`](CITATION.cff) for the full
contribution and citation records.

The method considers fixed node centres at each observation time and a closed
radius interval for every node. It constructs exact guaranteed and possible
proximity graphs, classifies radius-dependent edges, bounds monotone graph
observables, tests intermediate-graph realizability by linear programming,
computes the sharp endpoint-filtration shift, and obtains finite
zero-dimensional persistence deaths with Kruskal's algorithm.

## Repository contents

| Path | Purpose |
|---|---|
| `src/interval_radius_graph.py` | Maintained implementation of the mathematical method |
| `tests/` | Unit tests and validation tests for the published derived results |
| `analysis/run_interval_radius_case_study.py` | Standalone HDF5 analysis and local figure-generation workflow |
| `data/derived/` | Frame-level metrics, trajectory summaries, and analysis metadata |
| `data/provenance/` | File-level run provenance and SHA-256 identifiers for the raw trajectories |
| `data/raw/README.md` | Raw-data scope and archival policy |
| `scripts/verify_release.py` | One-command integrity and regression verification |
| `MANIFEST.sha256` | Checksums for the small archival artifacts |

## Quick start

Python 3.12 was used for the archived analysis. Create a clean environment and
install the package with its analysis dependencies:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[analysis]"
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[analysis]"
```

Run all checks:

```bash
python scripts/verify_release.py
```

The command verifies the archival checksums, runs the mathematical unit tests,
and validates the shape and internal invariants of the published derived data.

## Reproduce the case-study data

The raw HDF5 files are intentionally excluded from ordinary Git history. Place
them under `data/raw/article_s1_*/` using the paths listed in
`data/provenance/RUN_PROVENANCE.csv`, or pass another root explicitly. The runner
reads exactly those 15 manifest entries and checks each byte size and SHA-256:

```bash
python analysis/run_interval_radius_case_study.py \
  --input-root /path/to/raw/article-runs \
  --output-dir data/reproduced
python scripts/compare_reproduced.py data/reproduced
```

The archived sensitivity design is:

- relative Synapsin radius half-widths: `0`, `0.025`, `0.05`, and `0.10`;
- surface-gap thresholds: `0`, `2`, `4`, and `6` nm;
- primary threshold: `2` nm;
- six evenly spaced saved frames per trajectory.

These interval widths are declared sensitivity scenarios, not fitted
probability distributions and not direct experimental estimates of Synapsin
radius dynamics.
Use `--run-id RUN_ID` to reproduce one trajectory first; the comparison script
accepts either a subset or all 15 runs.

## Data layers

The repository distinguishes two committed data layers:

1. **Raw trajectories:** large HDF5 outputs, excluded from Git and identified by
   SHA-256 in `RUN_PROVENANCE.csv`.
2. **Derived data:** the CSV and JSON files in `data/derived/`, sufficient to
   reproduce the reported summary statistics and inspect every analysed
   frame-condition record.
Generated figures are written only to the ignored local output directory; they
are not committed to this repository.

See `REPRODUCIBILITY.md` and `DATA_DICTIONARY.md` for the full contract.

## Citation

Software citation metadata are provided in `CITATION.cff`. A journal citation
and versioned software DOI may be added when those identifiers exist.

## Funding

This work is based upon research funded by Iran National Science Foundation
(INSF) under project No. 4029556.

## License status

No reuse license is currently asserted. Please contact the authors before
reusing code or data. The authors will select an explicit code and data license
after confirming institutional and funder requirements.
