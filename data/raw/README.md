# Raw HDF5 trajectories

The raw simulation trajectories are not committed to ordinary Git history.
The 15 included HDF5 files total several gigabytes, while GitHub is intended
here only for code, small derived tables, and provenance.

Every raw file used by the case study is identified in
`../provenance/RUN_PROVENANCE.csv` by:

- run ID and original project-relative path;
- byte size;
- SHA-256 digest;
- completion and expected-frame flags;
- frame and particle counts;
- simulation time step and final physical time.

Until a DOI-backed raw-data deposit is released, the files can be obtained from
the corresponding author. For local reproduction, preserve the directory form

```text
data/raw/article_s1_<run-id>----<timestamp>/simulation_Rawdata*.h5
```

or pass a different root with `--input-root` to the analysis runner. Before
making a DOI archive public, verify each uploaded file against the SHA-256
digest in the provenance table.
