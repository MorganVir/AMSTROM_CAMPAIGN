# 03a_sequence_picker.py
# Interactive SEQ boundary picker over torque plot
# Step-owned artifact: 03a_sequence_picked.json

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


def run_sequence_manual_selection(
    ctx,
    seq_order,
    force_recompute,
):
    """
    Interactive torque-based SEQ boundary picker.

    Parameters
    ----------
    ctx : dict
        Must contain:
            - RUN_ID
            - CACHE_DIR
            - master_index_grid
            - torque_compact_df
            - ts_ref

    seq_order : list[str]
        Ordered list of SEQ labels (INIT ... END)

    force_recompute : bool
        If True, delete existing JSON and relaunch picker

    Returns
    -------
    output_json_path : Path
    """

    assert isinstance(ctx, dict)
    assert "RUN_ID" in ctx
    assert "CACHE_DIR" in ctx
    assert "master_index_grid" in ctx
    assert "torque_compact_df" in ctx
    assert "ts_ref" in ctx

    run_id = ctx["RUN_ID"]
    cache_dir = ctx["CACHE_DIR"]
    master_index_grid = ctx["master_index_grid"]
    torque_compact_df = ctx["torque_compact_df"]
    ts_ref = ctx["ts_ref"]

    assert isinstance(cache_dir, Path)
    assert isinstance(seq_order, (list, tuple))
    assert seq_order[0] == "INIT"
    assert seq_order[-1] == "END"

    output_json_path = cache_dir / "03a_sequence_picked.json"

    if output_json_path.exists() and not force_recompute:
        print(f"[03a_sequence_picker] Cache file for {run_id} exists. Existing sequences have been loaded. Force recompute or delete cache (03a_sequence_picker.json) to try again.")
        return output_json_path

    if force_recompute and output_json_path.exists():
        output_json_path.unlink()



    # Build torque time series
    time_ref_s_full = master_index_grid["time_ref_s"].to_numpy(dtype=float)

    torque_time_index = torque_compact_df["time_index"].to_numpy(dtype=int)
    torque_values_raw = torque_compact_df["torque_raw"].to_numpy(dtype=float)

    torque_time_ref_s = time_ref_s_full[torque_time_index]

    decimator_factor = 10

    torque_time_ref_s = torque_time_ref_s[::decimator_factor]
    torque_values_raw = torque_values_raw[::decimator_factor]

    # Interactive state
    picked_boundary_times_s = []

    def autosave():
        payload = {
            "run_id": run_id,
            "seq_order": list(seq_order),
            "picked_times_s": picked_boundary_times_s,
            "ts_ref": str(ts_ref),
        }
        with open(output_json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print("[03a_sequence_picker] Please select the different sequences boundaries (left clic on the torque plot BEFORE and AFTER each sequences). \n Once done, move to the next cell. \n Plot title must display 14/14 boundaries picked")

    # ----------------------------
    # Plot
    # ----------------------------

    plt.close("all")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(torque_time_ref_s, torque_values_raw, alpha=0.75)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque (raw)")

    vertical_lines = []

    def redraw():
        for line in vertical_lines:
            line.remove()
        vertical_lines.clear()

        for t_s in picked_boundary_times_s:
            vertical_lines.append(ax.axvline(t_s, color="orange", lw=2))

        ax.set_title(
            f"{run_id} | Boundaries: {len(picked_boundary_times_s)} / {len(seq_order) - 1}"
        )
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return

        picked_boundary_times_s.append(float(event.xdata))
        picked_boundary_times_s.sort()

        redraw()
        autosave()

    fig.canvas.mpl_connect("button_press_event", on_click)

    # Create empty JSON immediately
    autosave()
    redraw()

    plt.show()

    return output_json_path