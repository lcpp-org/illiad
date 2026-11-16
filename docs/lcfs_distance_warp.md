# Warp LCFS distance tracing

`misc_scripts/trace_lcfs_distance_cpu_warp.py` traces both directions from a
regular poloidal launch grid and measures unsigned distance to a reconstructed
3D LCFS triangle mesh. It is independent of `trace_lcfs_cpu.py` and
`trace_lcfs_distance_cpu_v2.py`; neither is needed to trace or replot its results.
It reuses integration helpers from `warp_sol_prototype.py`, the installed ILLIAD
package, and the companion modules in this checkout.

## Code organization

- `trace_lcfs_distance_cpu_warp.py`: editable defaults, CLI, float64 Warp kernels,
  refillable queue, and orchestration of single runs, offset scans and phi batches.
- `lcfs_distance_utils.py`: field loading, seed filtering, mesh caching, path
  metrics, validated atomic summary writes and path cleanup.
- `lcfs_distance_plots.py`: saved-path figures and per-point figures, including
  common color scales across offsets. Plotting does not load magnetic fields.
  Distance and connection-length colormaps are editable at the top of this file.
- `lcfs_distance_geometry.py`: existing LCFS mesh construction and reference
  nearest-triangle geometry used to build the Warp distance tree.

The numerical defaults previously inherited from the CPU scripts now live at
the top of the Warp script. Existing CLI options, archive arrays, CSV columns,
figure names, and output-folder conventions are retained. Legacy `workers`,
`max_step_m`, `rtol` and `atol` settings keys remain in saved metadata for readers;
Warp uses `step_size_m` and records null LSODA tolerances.

## Running and replotting

For example, with an environment containing ILLIAD and the optional Warp dependency:

```bash
conda run --no-capture-output -n warpenv python misc_scripts/trace_lcfs_distance_cpu_warp.py \
    output/IOTA3_1000sp_atol1e-9 --phi-deg 306 --nrho 32 --ntheta 48 \
    --offset-mm 5 --length-m 30 --active-seeds 512 --samples-per-chunk 32 \
    --no-plots --delete-paths
```

The default output folder is `RUN_DIR/lcfs_distance_warp`. `--all-phi` pools all
saved launch planes into one output; adding `--phi-batches` processes sequential
`phiDDD` folders instead. Multiple `--offset-mm` values create `offset_*mm`
subfolders and an offset manifest. Existing results are protected from overwrite.

Each run saves settings, the reference mesh, `summary.npz`, and three CSVs:
`connection_lengths.csv`, `lcfs_distances.csv`, and `point_metrics.csv`. Paths are
saved under `paths/` unless `--delete-paths` can skip them. When full-path figures
need those files, cleanup occurs after summaries and requested figures succeed.

```bash
conda run --no-capture-output -n warpenv python misc_scripts/trace_lcfs_distance_cpu_warp.py \
    --output output/IOTA3_1000sp_atol1e-9/lcfs_distance_warp \
    --plot-only --no-path-plots
```

Replotting uses saved summaries and settings, including after path cleanup.
`--combined-only` redraws combined offset figures. Full trajectory figures need
retained paths; if they are unavailable, per-point figures are still generated.

## Numerical and queue controls

`--step-size-m` controls the maximum integration step; `--sample-step-m` controls
saved samples and distance queries. RK4 is the default; midpoint is optional.
Steps shorten at sample locations and at the per-direction length cap. The wall
endpoint is refined by bisection along the terminal step's chord.

`--active-seeds` limits concurrent seed slots, each with forward/backward lanes.
`--samples-per-chunk` limits samples advanced per lane before transferring results
and refilling finished slots. Larger values can reduce launch overhead but use
more memory and delay refills. These controls do not change numerical resolution.
A seed slot is reused only after both directions finish; saved seed IDs stay in
launch order even when completion order differs.

Distance minima/maxima are taken over saved samples, including launch and terminal
points. The mean uses trapezoidal arc-length integration over both directions.
The shortest one-sided wall connection remains a lower bound if its selected
direction reaches the length cap. Step/sample refinement and denser LCFS meshes
are needed to assess convergence for a particular run.

## Targeted validation

```bash
env MPLCONFIGDIR=/tmp/illiad-mplconfig WARP_CACHE_PATH=/tmp/illiad-warp-cache \
    conda run --no-capture-output -n warpenv python -m unittest discover \
    -s tests -p test_lcfs_distance_warp.py -v
```

Tests default to Warp's CPU device. Set `ILLIAD_WARP_TEST_DEVICE=cuda` for GPU
execution. They use synthetic fields and saved surfaces and block imports of the
two retired CPU tracing scripts. CPU checks establish behavior only on that device; they do
not measure GPU performance or full-scan convergence.
