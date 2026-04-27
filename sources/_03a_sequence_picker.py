# _03a_sequence_picker.py - INTERACTIVE SEQUENCE BOUNDARY PICKER
# Manual click-based tool to mark protocol sequence boundaries on the torque signal.
# This step requires human input and cannot be automated: the operator left-clicks on the
# torque plot to place a boundary marker before and after each protocol sequence.
#
# Inputs
#   ctx keys : RUN_ID, CACHE_DIR, master_index_grid, torque_compact_df, ts_ref
#   parameter: seq_order      — ordered list of SEQ labels (must start with "INIT", end with "END")
#              force_recompute — if True, deletes existing JSON and reopens the picker
#
# Outputs
#   return value : path to the saved JSON file (03a_sequence_picked.json)
#   cache file   : 03a_sequence_picked.json (written after every click, committed when satisfied)
#
# JSON payload structure
#   {
#     "run_id":         "<RUN_ID>",
#     "seq_order":      ["INIT", "MVC_REF", ..., "END"],
#     "picked_times_s": [t0, t1, ..., tN],   ← one boundary per gap between sequences
#     "ts_ref":         "<ISO timestamp>"
#   }
#
# Notes
#   - Requires an interactive matplotlib backend (runs in Jupyter with %matplotlib widget).
#   - Boundaries are sorted automatically after each click so order of clicking does not matter.
#   - The JSON is written after every click for safety; moving to the next cell commits the result.
#   - force_recompute=True physically deletes the existing JSON so the picker starts fresh.

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


# Downsampling factor applied to the torque signal before plotting.
# The full-resolution torque (~100 Hz) is dense enough to make interactive clicking imprecise;
# decimating by 10 keeps the shape visible while making boundary placement easier.
TORQUE_PLOT_DECIMATION = 10


def run_sequence_manual_selection(
    ctx,
    seq_order,
    force_recompute,
):
    """
    Open an interactive torque plot for manual protocol sequence boundary picking.

    The operator left-clicks on the plot to place a vertical boundary marker at each
    transition between sequences. Boundaries are saved to JSON after every click.
    The expected number of boundaries is len(seq_order) - 1 (one per gap between labels).

    Returns the path to the saved JSON file. Move to the next notebook cell when done.
    """

    # --- Resolve ctx ---
    run_id            = ctx["RUN_ID"]
    cache_dir         = ctx["CACHE_DIR"]
    master_index_grid = ctx["master_index_grid"]
    torque_compact_df = ctx["torque_compact_df"]
    ts_ref            = ctx["ts_ref"]

    output_json_path = cache_dir / "03a_sequence_picked.json"

    # --- Cache check ---
    if output_json_path.exists() and not force_recompute:
        print(f"[03a_sequence_picker] Boundaries already picked for {run_id}. Set force_recompute=True to redo.")
        return output_json_path

    # force_recompute: delete the existing JSON so the picker starts with a clean slate.
    if force_recompute and output_json_path.exists():
        output_json_path.unlink()

    # --- Build torque time series for plotting ---
    # Map torque time_index values back to continuous time using the master grid.
    time_ref_s_full   = master_index_grid["time_ref_s"].to_numpy(dtype=float)
    torque_time_index = torque_compact_df["time_index"].to_numpy(dtype=int)
    torque_values_raw = torque_compact_df["torque_raw"].to_numpy(dtype=float)
    torque_time_ref_s = time_ref_s_full[torque_time_index]

    # Decimate for display — full resolution is not needed for manual boundary picking.
    torque_time_ref_s = torque_time_ref_s[::TORQUE_PLOT_DECIMATION]
    torque_values_raw = torque_values_raw[::TORQUE_PLOT_DECIMATION]

    # --- Interactive state ---
    # picked_boundary_times_s accumulates one time value per left-click.
    # It is kept sorted so JSON output is always in chronological order.
    picked_boundary_times_s = []

    def save_picks():
        """Write current boundary picks to JSON. Called after every click and at init."""
        payload = {
            "run_id":         run_id,
            "seq_order":      list(seq_order),
            "picked_times_s": picked_boundary_times_s,
            "ts_ref":         str(ts_ref),
        }
        with open(output_json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    # --- Plot ---
    plt.close("all")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(torque_time_ref_s, torque_values_raw, alpha=0.75)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque (raw)")

    # vertical_lines holds the Line2D objects for the boundary markers so they can be
    # cleared and redrawn on each click without accumulating stale lines.
    vertical_lines = []

    def redraw():
        """Clear and redraw all boundary markers, then update the title with pick count."""
        for line in vertical_lines:
            line.remove()
        vertical_lines.clear()

        for t_s in picked_boundary_times_s:
            vertical_lines.append(ax.axvline(t_s, color="orange", lw=2))

        n_expected = len(seq_order) - 1
        ax.set_title(f"{run_id} | Boundaries picked: {len(picked_boundary_times_s)} / {n_expected}")
        fig.canvas.draw_idle()

    def on_click(event):
        """Register a left-click as a new boundary, sort, save, and redraw."""
        if event.inaxes != ax or event.xdata is None:
            return
        picked_boundary_times_s.append(float(event.xdata))
        # Sort after each click so the operator can place boundaries in any order.
        picked_boundary_times_s.sort()
        redraw()
        save_picks()

    fig.canvas.mpl_connect("button_press_event", on_click)

    # Write an empty JSON immediately so the next cell can detect that picking has started,
    # even before any clicks have been registered.
    save_picks()
    redraw()

    print(
        f"[03a_sequence_picker] Click to place {len(seq_order) - 1} boundaries on the torque plot.\n"
        f"  Title shows current count. Move to the next cell when done."
    )
    plt.show()

    return output_json_path
