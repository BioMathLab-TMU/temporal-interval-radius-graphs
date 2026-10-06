# Reproducibility contract

## Mathematical layer

`src/interval_radius_graph.py` implements the graph definitions and algorithms:

- exact guaranteed and possible graph endpoints;
- impossible, radius-dependent, and guaranteed edge flags;
- exact intervals for monotone and antitone component observables;
- the sharp uniform endpoint-filtration shift;
- finite zero-dimensional persistence deaths by Kruskal's algorithm;
- linear-programming certificates for prescribed intermediate graphs.

The corresponding regression tests are in
`tests/test_interval_radius_graph.py`.

## Case-study layer

The case study treats saved particle centres as fixed observations. Only the
effective radii of Synapsin nodes are replaced by symmetric declared intervals
at each selected frame. No trajectory is rerun, no within-frame radius process
is assumed, and no probability distribution is assigned within an interval.

The analysis includes 15 complete HDF5 trajectories. Six equally spaced saved
frames are selected from every trajectory. The union of the archived
primary-width and threshold-sensitivity conditions produces 630 distinct
frame-condition records.

The runner reads the exact 15 paths in `RUN_PROVENANCE.csv` and enforces the
following data-quality checks before analysis:

1. the simulation reports successful completion;
2. every expected frame is present and marked complete;
3. particle IDs and types are invariant across sampled frames;
4. coordinates and radii are finite;
5. radii are strictly positive;
6. sampled physical times increase strictly;
7. every included raw source is hashed with SHA-256.
8. every raw source matches its archived byte size and SHA-256 digest.

## Verification levels

### Level 1: repository integrity

```bash
python scripts/verify_release.py
```

This verifies the archival manifest, mathematical tests, and published derived
data invariants without requiring the multi-gigabyte raw trajectories.

### Level 2: full numerical reproduction

After retrieving the HDF5 files listed in
`data/provenance/RUN_PROVENANCE.csv`, run:

```bash
python analysis/run_interval_radius_case_study.py \
  --input-root /path/to/raw/article-runs \
  --output-dir data/reproduced
python scripts/compare_reproduced.py data/reproduced
```

The comparison checks the CSV values and the included raw-source SHA-256 values.
The JSON generation timestamp is expected to differ. PNG byte hashes can vary
across Matplotlib or font versions even when the plotted values are identical,
which is why `requirements-lock.txt` records the validated environment.

## Claim boundary

The repository establishes exact interval graph envelopes and validates the
reported graph-level case study. It does not claim that the declared radius
intervals are measured probability distributions, that microscopic radius
dynamics have been inferred, or that the biological simulation itself has been
experimentally validated by this graph analysis.
