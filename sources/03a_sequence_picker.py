# 04a_seq_pick - interactive SEQ boundary picker over torque plot
# Output: 04a_seq_pick.json (autosaved on each click)

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


# REQUIRED INPUTS 
assert "RUN_ID" in globals(), "RUN_ID missing"
assert "CACHE_DIR" in globals(), "CACHE_DIR missing"
assert isinstance(CACHE_DIR, Path), "CACHE_DIR must be a pathlib.Path"

assert "master_index_grid" in globals(), "master_index_grid missing"
assert "torque_compact_df" in globals(), "torque_compact_df missing"
assert "SEQ_ORDER" in globals(), "SEQ_ORDER missing"
assert "ts_ref" in globals(), "ts_ref missing"


# CONFIG 

OUTPUT_JSON_PATH = CACHE_DIR / "03a_sequence_picked.json"
DECIMATION_FACTOR = 5  # keep 1 point every 5 torque samples (~20 Hz visual)



# ----------------------------
# Torque series at native torque rate (mapped to ref grid indices)
# - plot only existing torque samples (no NaN-filled full grid)
# ----------------------------
time_ref_s_full = master_index_grid["time_ref_s"].to_numpy(dtype=float)

torque_time_index = torque_compact_df["time_index"].to_numpy(dtype=int)
torque_values_raw = torque_compact_df["torque_raw"].to_numpy(dtype=float)

torque_time_ref_s = time_ref_s_full[torque_time_index]

DECIMATION_FACTOR = 5  # optional
torque_time_ref_s = torque_time_ref_s[::DECIMATION_FACTOR]
torque_values_raw = torque_values_raw[::DECIMATION_FACTOR]




# Boundary picking state + autosave

picked_boundary_times_s: list[float] = []

def autosave_picked_boundaries() -> None:
    payload = {
        "run_id": RUN_ID,
        "seq_order": list(SEQ_ORDER),
        "picked_times_s": picked_boundary_times_s,
        "ts_ref": str(ts_ref),
    }
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)
    print(f"[SAVED] {len(picked_boundary_times_s)} boundaries -> {OUTPUT_JSON_PATH.name}")


# Plot + click handler

figure, axis = plt.subplots(figsize=(14, 5))
axis.plot(torque_time_ref_s, torque_values_raw, alpha=0.75)
axis.set_xlabel("Time (s)")
axis.set_ylabel("Torque (raw)")


vertical_lines = []

def redraw_vertical_lines() -> None:
    for line in vertical_lines:
        line.remove()
    vertical_lines.clear()

    for boundary_time_s in picked_boundary_times_s:
        vertical_lines.append(axis.axvline(boundary_time_s, color="orange", lw=2))

    axis.set_title(
        f"{RUN_ID} | Boundaries: {len(picked_boundary_times_s)} / {len(SEQ_ORDER) - 1} | autosave: {OUTPUT_JSON_PATH.name}"
    )
    figure.canvas.draw_idle()

def on_mouse_click(event) -> None:
    if event.inaxes != axis or event.xdata is None:
        return

    picked_boundary_times_s.append(float(event.xdata))
    picked_boundary_times_s.sort()

    redraw_vertical_lines()
    autosave_picked_boundaries()

figure.canvas.mpl_connect("button_press_event", on_mouse_click)

# Save an initial file (empty picks) so the artifact always exists once launched
autosave_picked_boundaries()
redraw_vertical_lines()
plt.show()
