# Data dictionary

## `data/derived/interval_radius_frame_metrics.csv`

Each row represents one `(run, sampled frame, radius half-width, threshold)`
condition.

### Identity and analysis condition

- `source_file`: repository-independent path recorded for the source HDF5 file.
- `run_id`: simulation run identifier.
- `simulation_step`, `time_ms`, `sampled_frame_index`: temporal identifiers.
- `relative_radius_half_width`: declared symmetric fractional half-width applied
  to Synapsin radii.
- `synapsin_radius_lower_nm`, `synapsin_radius_upper_nm`: realised endpoint
  radii in nanometres for the varied node class.
- `varied_node_type`: numerical particle-type identifier; `1` denotes Synapsin.
- `threshold_nm`: surface-gap threshold in nanometres.
- `n_nodes`: number of graph nodes.
- `schema_version`: analysis-record schema.

### Edge partition

- `candidate_pair_count`: pairs returned by the exact spatial candidate search.
- `guaranteed_edge_count`: edges present for every admissible radius vector.
- `possible_edge_count`: edges present for at least one admissible radius vector.
- `radius_dependent_edge_count`: possible minus guaranteed edges.
- `radius_dependent_fraction_of_possible_edges`: dependent edges divided by
  possible edges, or zero when there are no possible edges.

### Exact observable intervals

The suffixes `_lower`, `_upper`, and `_width` denote the exact endpoint interval
and its width. They are supplied for:

- `edge_count`;
- `largest_component_size`;
- `largest_component_fraction`;
- `mean_degree`;
- `component_count`;
- `isolated_fraction`.

For observables that decrease when edges are added, the lower observable bound
comes from the possible graph and the upper bound from the guaranteed graph.

## `data/derived/interval_radius_trajectory_summary.csv`

Each row aggregates the selected frames for one
`(run_id, relative_radius_half_width, threshold_nm)` condition. Columns report
the sample count, particle count, means of the three edge classes, the maximum
dependent-edge count, and mean or maximum widths of component-level intervals.

## `data/derived/interval_radius_case_study.json`

The JSON file records the analysis design, completion checks, included and
excluded sources, source SHA-256 values, and the names of the derived outputs.

## `data/provenance/RUN_PROVENANCE.csv`

One row per included raw HDF5 file. It records the run identifier, original
project-relative path, byte size, SHA-256 digest, completion flags, frame and
particle counts, simulation time step, and final physical time.
