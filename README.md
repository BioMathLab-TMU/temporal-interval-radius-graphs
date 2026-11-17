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
| `analysis/benchmark_pair_search.py` | Exact spatial-search benchmark against blocked all-pairs enumeration |
| `src/critical_width_sweep.py` | Exact pairwise width events, endpoint component sweeps, and ambiguity quotient |
| `analysis/analyze_critical_widths.py` | Hash-checked structural audit of 15 saved frames |
| `analysis/plot_critical_width_audit.py` | Rebuild the private manuscript's structural audit figure from derived JSON |
| `analysis/benchmark_width_sweep.py` | Compare width-event union-find with repeated sparse endpoint summaries |
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

## Critical radius widths and structural ambiguity

The symmetric one-parameter interval family can be evaluated at any declared
width in `[0, 0.1]`. The sweep computes the pairwise widths at which possible
edges enter or guaranteed edges leave and uses union-find for component
summaries. At a threshold of 2 nm, the separate structural audit analyses
the middle saved frame of each of the 15 provenance-listed runs. It checks
the source hashes and agrees with independent endpoint constructions at
widths `0`, `0.025`, `0.05`, and `0.1` for edge count, component count,
largest-component size, and isolated-node count. It does not interpret
frames as independent biological replicates.

To reproduce this audit from the raw HDF5 archive:

```bash
python analysis/analyze_critical_widths.py \
  --input-root /path/to/raw/article-runs --all-runs \
  --output data/derived/critical_widths.json
```

The quotient count contracts each guaranteed component and counts uncertain
edges between contracted vertices. Its possible-component partition is
checked against the direct possible graph. Exact event ratios are defined
mathematically; floating-point representations at an event itself can differ
in the last bit, so the validation checks report widths separately from
critical-ratio equality tests.
The figure used in the manuscript can be regenerated locally with
`python analysis/plot_critical_width_audit.py --output /path/to/Fig2.png`.
The image itself is excluded from this code repository.

The width-sweep benchmark is reproduced with:

```bash
python analysis/benchmark_width_sweep.py \
  --input-root /path/to/raw/article-runs \
  --output data/derived/width_sweep_benchmark.json --repeats 5
```

It checks equality of four endpoint observables for 4 and 40 widths on three
hash-checked middle frames. On the recorded Windows run, the event sweep was
slower for 4 widths but 4.43-4.58 times faster for 40 widths. These timings
are conditional on the chosen implementation, hardware, and data; the
benchmark does not claim a universal speedup.

See `REPRODUCIBILITY.md` and `DATA_DICTIONARY.md` for the full contract.

## Compare with radius sampling and a common cutoff

```bash
python analysis/compare_radius_uncertainty_baselines.py \
  --input-root /path/to/raw/article-runs --draws 64 \
  --output data/derived/radius_baseline_comparison.json
```

The script checks all 15 raw HDF5 hashes, draws independent uniform radii
within the declared intervals, and verifies every sampled edge set lies
between the exact endpoint edge sets. It also chooses one scalar cutoff per
frame to match the possible-edge count and reports the symmetric difference
of edge identities. Neither comparator estimates a biological radius law.

## Independent microscopy-mask application

The public BBBC039v1 mask archive is kept outside this repository. Its
official source is `https://data.broadinstitute.org/bbbc/BBBC039/masks.zip`
and its expected SHA-256 is recorded in the analysis script. Reproduce the
per-field analysis with:

```bash
python analysis/analyze_bbbc039_masks.py \
  --masks-zip /path/to/BBBC039/masks.zip \
  --output data/derived/bbbc039_radius_graph.json
```

The code uses independently annotated two-dimensional nuclei masks as
spatial input. Equivalent-circle radii and a 10% interval are explicit
analysis choices, not measured uncertainties in the original dataset.

## Reproduce the pair-search benchmark

With the raw trajectories placed under the path layout described above, run:

```bash
python analysis/benchmark_pair_search.py \
  --input-root /path/to/raw/article-runs \
  --output data/derived/pair_search_benchmark.json \
  --repeats 5 --max-reference-nodes 15000
```

The benchmark selects the middle saved frame from the first provenance-listed
trajectory at each recorded node count. It checks raw SHA-256 identifiers,
times exact tree-based pair generation and endpoint classification, and
compares the endpoint edge sets with a blocked all-pairs reference at all
three archived sizes. Timings
exclude HDF5 loading, source hashing, and component calculations. The JSON
records individual repeats and software versions; timings depend on hardware
and workload.

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
