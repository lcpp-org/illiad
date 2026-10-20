"""Plot wall-density contour outlines for any number of saved Boris runs.

Uses the same saved runs, full-wall normalization, bins, threshold (1e-5),
and 0--72 degree view as plot_wall_hist_overlay.py. Edit RUNS in that script;
runs are drawn in list order, with later entries on top.
Contours interpolate between histogram bin centers, without smoothing.

Run directly or use --runs DIR [DIR ...], --threshold, --output, and --show.
Default output: output/wall_hist_overlay/IOTA3_IOTA4_IOTA5_contours.png.
"""

from plot_wall_hist_overlay import main


if __name__ == "__main__":
    main(default_style="contour")
