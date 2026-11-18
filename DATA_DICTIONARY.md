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

## `data/derived/critical_widths.json`

One record per provenance-listed run, using its middle saved frame. The
`source_sha256` identifies the raw HDF5 file; `candidate_pairs` is the safe
tree-search output at maximum width 0.1 and surface gap 2 nm.
`possible_edge_birth_events` and `guaranteed_edge_exit_events` count pairwise
events inside the declared width interval. `critical_width_count` counts
distinct numerical event widths plus both domain endpoints. These are edge
events, not necessarily changes in connected components.
`possible_component_merger_events` counts unions among possible-edge births;
`guaranteed_component_split_events` counts reverse-sweep unions among
guaranteed-edge exits. Both count merger operations, so simultaneous events
at one width can contribute more than one count.

`endpoints` reports exact graph summaries at each width in the file's `widths`
array. The `quotient_at_0p1` fields count uncertain edges inside and between
guaranteed components, guaranteed and possible component counts, and the
number of component mergers. The 15 records are one frame per heterogeneous
simulation run; they are not 15 exchangeable biological replicates.

## `data/derived/width_sweep_benchmark.json`

Six records: one middle saved frame for the first manifest run at each of
three particle counts, each tested with 4 and 40 query widths. Both methods
use one safe KD-tree candidate search and report the same endpoint edge count,
component count, largest-component size, and isolated-node count at every
queried width. `exact_summary_equality` records this check.
`repeated_csr_seconds` and `width_sweep_seconds` contain five wall-clock
repetitions each; their corresponding median fields are in seconds.
`median_time_ratio_repeated_over_sweep` is the ratio of the two medians, with
values above one favouring the event sweep. File read and hashing are excluded
from timing. The comparison is workload- and implementation-specific.

## `data/derived/radius_baseline_comparison.json`

One middle frame per hash-verified HDF5 run, using a 0.10 radius half-width
for type-1 nodes and a 2 nm surface-gap threshold. Schema
`radius-baselines-3` records exact endpoint edge counts and largest-component
fractions. `sampling.independent_uniform` gives independent uniform radii for
each varying node; `sampling.common_uniform` uses one shared uniform factor
for all varying radii in a graph. Each law has a reproducible seed and nested
64- and 1024-draw summaries of sampled edge-count and largest-component
spans, sample standard deviations, 5th/95th percentiles, endpoint hits,
and component-count ranges. `n_varied_nodes` counts the varying radii in
each frame. The span fraction divides
the sampled range by the exact endpoint range; it is null when the exact
range is zero. `type_pair_cutoffs` records the nominal-gap shift coefficient
for each type pair and whether it is constant; the equality flags check both
endpoint edge sets. `scalar_edge_disagreement` is the symmetric difference
between the single edge-count-matched cutoff and the possible pair set.
Neither law is inferred from biological data. No Monte Carlo value is used
as a substitute for the deterministic endpoint certificate.

## `data/derived/correlated_radius_continuum.json`

Exploratory within-frame dependence experiment. Schema
`correlated-radius-continuum-1` stores 15 hash-verified archived middle frames
and nine synthetic finite disk-graph cases. Each record has 1,024 draw-level
edge counts and largest-component fractions at eight target pairwise Pearson
correlations of uniform radius factors. `rho_results` stores sample SD,
quantiles, exact endpoint span, and two SD predictions. The
`linearized_sd_prediction` interpolates the empirically observed endpoint
variances; `naive_n_var_sd_prediction` uses the raw varying-node count.
`degree_effective_count` is an ambiguous-edge-incidence proxy, not a fitted
derivative count. `mcdiarmid_sum_squares` is the sum of squared ambiguous
incidences over varying nodes and supports an independent-radius edge-count
bounded-differences bound. The supplied correlation law is a Gaussian copula
with uniform marginals; it is not calibrated to observed protein sizes.
`correlated_radius_continuum_width_0p05.json` repeats the same design at a
5% radius half-width with 512 draws per case and correlation. Its schema and
field meanings are identical; the top-level `width` and per-record `draws`
state the changed settings.

## `data/derived/bbbc039_radius_graph.json`

Independent external application to the public BBBC039v1 nuclei-mask
archive. The record stores the official source URL, SHA-256 digest, CC0
status, decoding rule, 200 image identifiers, per-image object counts,
and endpoint edge/component summaries at 0 and 2 pixel surface gaps.
Three fields have fewer than two decoded objects and therefore have no
graph summaries. Equivalent-circle radii are computed from segmented pixel
areas; the 10% interval is a declared sensitivity scenario.
