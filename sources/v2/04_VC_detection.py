# 04_vc_detection 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Required globals

assert "RUN_ID" in globals()
assert "master_index_grid" in globals()
assert "torque_compact_df" in globals()
assert "CACHE_04_VC_KNOBS_EVENTS" in globals()
assert "emg_compact_df" in globals() and isinstance(emg_compact_df, pd.DataFrame), "emg_compact_df missing/invalid"


TORQUE_COL = "torque_raw"
VC_QC_PLOT_DECIM = 500
cache_path = CACHE_04_VC_KNOBS_EVENTS


# Required columns

for c in ["time_index", "time_ref_s", "SEQ"]:
    assert c in master_index_grid.columns

for c in ["time_index", TORQUE_COL]:
    assert c in torque_compact_df.columns



# Knobs comes from the dashboard
assert "VC_KNOBS_BY_SEQ" in globals(), "04_vc_detection: VC_KNOBS_BY_SEQ missing (dashboard must provide knobs)"
VC_KNOBS_BY_SEQ = globals()["VC_KNOBS_BY_SEQ"]
assert isinstance(VC_KNOBS_BY_SEQ, dict) and len(VC_KNOBS_BY_SEQ) > 0, "04_vc_detection: VC_KNOBS_BY_SEQ must be a non-empty dict"

thr_frac_by_seq = {k: float(v["thr_frac"]) for k, v in VC_KNOBS_BY_SEQ.items()}
merge_gap_s_by_seq = {k: float(v["merge_gap_s"]) for k, v in VC_KNOBS_BY_SEQ.items()}
min_dur_s_by_seq = {k: float(v["min_dur_s"]) for k, v in VC_KNOBS_BY_SEQ.items()}




# Map torque to master time
time_ref_s_full = master_index_grid["time_ref_s"].to_numpy(dtype=float)

torque_time_index = torque_compact_df["time_index"].to_numpy(dtype=int)
torque_time_ref_s = time_ref_s_full[torque_time_index]
torque_values = torque_compact_df[TORQUE_COL].to_numpy(dtype=float)

seq_labels_full = master_index_grid["SEQ"].to_numpy(dtype=object)


# Prepare output labels
VC = np.zeros(len(master_index_grid), dtype=int)
VC_count = np.zeros(len(master_index_grid), dtype=int)

event_rows = []


# Detection per SEQ
for seq_name, thr_frac in thr_frac_by_seq.items():

    seq_mask = (seq_labels_full[torque_time_index] == seq_name)
    if not np.any(seq_mask):
        continue

    seq_indices_global = torque_time_index[seq_mask]
    seq_torque = torque_values[seq_mask]
    seq_time = torque_time_ref_s[seq_mask]

    seq_max = float(np.max(seq_torque))
    threshold = float(thr_frac) * seq_max
    above = seq_torque >= threshold

    # find segments 
    diff_mask = np.diff(above.astype(int))
    start_local = np.where(diff_mask == 1)[0] + 1
    end_local = np.where(diff_mask == -1)[0]

    if above.size > 0 and above[0]:
        start_local = np.r_[0, start_local]
    if above.size > 0 and above[-1]:
        end_local = np.r_[end_local, len(above) - 1]

    segments = list(zip(start_local.tolist(), end_local.tolist()))
    if not segments:
        continue

    # merge gaps 
    merge_gap_s = float(merge_gap_s_by_seq.get(seq_name, 0.0))
    merged = []
    cur_s, cur_e = segments[0]

    for nxt_s, nxt_e in segments[1:]:
        gap_s = seq_time[nxt_s] - seq_time[cur_e]
        if gap_s <= merge_gap_s:
            cur_e = nxt_e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = nxt_s, nxt_e

    merged.append((cur_s, cur_e))

    # drop short 
    min_dur_s = float(min_dur_s_by_seq.get(seq_name, 0.0))
    final_segments = []

    for s, e in merged:
        if (seq_time[e] - seq_time[s]) >= min_dur_s:
            final_segments.append((s, e))

    #  write labels 
    vc_number = 0
    for s, e in final_segments:
        vc_number += 1

        global_start = seq_indices_global[s]
        global_end = seq_indices_global[e]

        VC[global_start:global_end+1] = 1
        VC_count[global_start:global_end+1] = vc_number

        event_rows.append({
            "run_id": RUN_ID,
            "row_type": "event",
            "seq": seq_name,
            "vc_in_seq": vc_number,
            "start_time_index": int(global_start),
            "stop_time_index": int(global_end),
            "start_time_ref_s": float(seq_time[s]),
            "stop_time_ref_s": float(seq_time[e]),
            "dur_s": float(seq_time[e] - seq_time[s]),
            "threshold": threshold,
            "seq_max": seq_max,
        })

vc_events_df = pd.DataFrame(event_rows)


# Apply VC labels
master_index_grid_out = master_index_grid.copy()
master_index_grid_out["VC"] = VC
master_index_grid_out["VC_count"] = VC_count


# ... and propagate VC labels to compact tables (emg & torque) via time_index

master_VC = master_index_grid_out["VC"].to_numpy(dtype=int)
master_VC_count = master_index_grid_out["VC_count"].to_numpy(dtype=int)

torque_time_index = torque_compact_df["time_index"].to_numpy(dtype=int)
assert torque_time_index.size > 0, "torque_compact_df has zero rows"
assert int(np.max(torque_time_index)) < len(master_index_grid_out), "torque time_index out of master range"

torque_compact_df_out = torque_compact_df.copy()
torque_compact_df_out["VC"] = master_VC[torque_time_index]
torque_compact_df_out["VC_count"] = master_VC_count[torque_time_index]

emg_time_index = emg_compact_df["time_index"].to_numpy(dtype=int)
assert emg_time_index.size > 0, "emg_compact_df has zero rows"
assert int(np.max(emg_time_index)) < len(master_index_grid_out), "emg time_index out of master range"

emg_compact_df_out = emg_compact_df.copy()
emg_compact_df_out["VC"] = master_VC[emg_time_index]
emg_compact_df_out["VC_count"] = master_VC_count[emg_time_index]

# Commit only if requested
VC_COMMIT = bool(globals().get("VC_COMMIT", False))

if VC_COMMIT:
    # Build committed knob table from the effective values used this run
    committed_knob_rows = []
    for seq_name in thr_frac_by_seq.keys():
        committed_knob_rows.append({
            "run_id": RUN_ID,
            "row_type": "knob",
            "seq": seq_name,
            "thr_frac": float(thr_frac_by_seq[seq_name]),
            "merge_gap_s": float(merge_gap_s_by_seq[seq_name]),
            "min_dur_s": float(min_dur_s_by_seq[seq_name]),
        })
    vc_committed_knobs_df = pd.DataFrame(committed_knob_rows)

    vc_knobs_plus_events_df = pd.concat([vc_committed_knobs_df, vc_events_df], ignore_index=True, sort=False)
    vc_knobs_plus_events_df.to_parquet(cache_path, index=False)



VC_PLOT = bool(globals().get("VC_PLOT", True))

if VC_PLOT:

    # QC Plot (decimate torque only)
    plot_decim = max(1, int(VC_QC_PLOT_DECIM))

    torque_time_ref_s_plot = torque_time_ref_s[::plot_decim]
    torque_values_plot = torque_values[::plot_decim]
    torque_time_index_plot = torque_time_index[::plot_decim]

    seq_labels_plot = seq_labels_full[torque_time_index_plot]

    fig, axes = plt.subplots(len(thr_frac_by_seq), 1, figsize=(14, 2.2 * len(thr_frac_by_seq)))
    if len(thr_frac_by_seq) == 1:
        axes = [axes]

    vc_events_by_seq = {seq: df for seq, df in vc_events_df.groupby("seq")} if len(vc_events_df) > 0 else {}

    for ax, seq_name in zip(axes, thr_frac_by_seq.keys()):

        mask = (seq_labels_plot == seq_name)
        if not np.any(mask):
            ax.set_title(f"{seq_name} (missing)")
            ax.axis("off")
            continue

        ax.plot(torque_time_ref_s_plot[mask], torque_values_plot[mask], linewidth=1)

        seq_full_mask = (seq_labels_full[torque_time_index] == seq_name)
        if np.any(seq_full_mask):
            full_seq_torque = torque_values[seq_full_mask]
            thr = thr_frac_by_seq[seq_name] * np.max(full_seq_torque)
            ax.axhline(thr, linestyle="--", linewidth=1)

        seq_events = vc_events_by_seq.get(seq_name, vc_events_df.iloc[0:0])
        n_vc = int(seq_events["vc_in_seq"].max()) if len(seq_events) > 0 else 0

        for ev in seq_events.itertuples(index=False):
            ax.axvspan(ev.start_time_ref_s, ev.stop_time_ref_s, alpha=0.15)
            ax.text(
                ev.start_time_ref_s,
                0.95 * np.max(torque_values_plot[mask]),
                str(int(ev.vc_in_seq)),
                va="top",
                ha="left",
                fontsize=9,
            )

        ax.set_title(
            f"{seq_name} | "
            f"thr={thr_frac_by_seq[seq_name]:.3f} | "
            f"merge={merge_gap_s_by_seq[seq_name]:.2f}s | "
            f"min_dur={min_dur_s_by_seq[seq_name]:.2f}s | "
            f"n_VC={n_vc}"
        )

        ax.set_ylabel("Torque (raw)")

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


