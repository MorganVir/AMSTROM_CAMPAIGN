# _04_VC_detection.py - VOLUNTARY CONTRACTION DETECTION
# Detect individual contractions from the torque signal and label them in the master grid.
#
# Inputs
#   ctx keys:  RUN_ID, CACHE_DIR, master_index_grid, torque_compact_df, emg_compact_df
#
# Outputs (ctx keys updated)
#   - master_index_grid  (VC and VC_count columns added)
#   - torque_compact_df  (VC and VC_count propagated)
#   - emg_compact_df     (VC and VC_count propagated)
#
# Cache
#   - 04_vc_knobs_events.parquet  (committed after manual user validation)
#
# Notes
#   - Detection is automatic (torque threshold) with per-SEQ manual knobs for edge cases.
#   - VC_count increments within each SEQ; resets at each new sequence.
#   - Human validation is required before committing to cache.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.close('all') #prevent ghosting with previous cells figure being recomputed here.

def run_voluntary_contraction_detection(
    ctx,
    vc_knobs_by_seq,
    force_recompute,
):
    """
    Torque-based VC detection.

    Parameters
    ----------
    ctx : dict
        Must contain:
            - RUN_ID
            - CACHE_DIR
            - master_index_grid
            - torque_compact_df
            - emg_compact_df

    vc_knobs_by_seq : dict
        {
            "SEQ_NAME": {
                "thr_frac": float, (fraction of SEQ max torque to set as threshold)
                "merge_gap_s": float, (max gap to merge consecutive segments above threshold)
                "min_dur_s": float, (minimum duration of above-threshold segment to be considered a VC. Here, should be between 2-3s))
            },
            ...
        }
    force_recompute : bool
    """

    required = [
        "RUN_ID",
        "CACHE_DIR",
        "master_index_grid",
        "torque_compact_df",
        "emg_compact_df",
    ]
    for k in required:
        assert k in ctx, f"{k} missing in ctx"

    run_id = ctx["RUN_ID"]
    cache_dir = ctx["CACHE_DIR"]
    master_index_grid = ctx["master_index_grid"]
    torque_compact_df = ctx["torque_compact_df"]
    emg_compact_df = ctx["emg_compact_df"]

    cache_path = cache_dir / "04_vc_knobs_events.parquet"


    # cache hit
    if cache_path.exists() and not force_recompute:
        df = pd.read_parquet(cache_path)
        if "row_type" in df.columns and df["row_type"].eq("event").any():
            print("[04_VC_Detection] Cache exists and already contains events. Existing VC events have been loaded.\nForce recompute or delete cache (04_vc_knobs_events.parquet) to try again.\nMake sure to run the next cell so the VC events are properly applied to the existing DataFrames.")
            return master_index_grid, torque_compact_df, emg_compact_df, df


    # Checks
    assert isinstance(vc_knobs_by_seq, dict) and len(vc_knobs_by_seq) > 0
    assert "SEQ" in master_index_grid.columns

    TORQUE_COL = ctx.get("TORQUE_COL", "torque_raw")

    for c in ["time_index", "time_ref_s", "SEQ"]:
        assert c in master_index_grid.columns

    for c in ["time_index", TORQUE_COL]:
        assert c in torque_compact_df.columns


    # Prepare data
    time_ref_s_full = master_index_grid["time_ref_s"].to_numpy(dtype=float)
    seq_labels_full = master_index_grid["SEQ"].to_numpy(dtype=object)

    torque_time_index = torque_compact_df["time_index"].to_numpy(dtype=int)
    torque_time_ref_s = time_ref_s_full[torque_time_index]
    torque_values = torque_compact_df[TORQUE_COL].to_numpy(dtype=float)

    VC = np.zeros(len(master_index_grid), dtype=int)
    VC_count = np.zeros(len(master_index_grid), dtype=int)

    event_rows = []


    # Detection per SEQ
    for seq_name, knobs in vc_knobs_by_seq.items():

        thr_frac = float(knobs["thr_frac"])
        merge_gap_s = float(knobs["merge_gap_s"])
        min_dur_s = float(knobs["min_dur_s"])

        seq_mask = (seq_labels_full[torque_time_index] == seq_name)
        if not np.any(seq_mask):
            continue

        seq_indices_global = torque_time_index[seq_mask]
        seq_torque = torque_values[seq_mask]
        seq_time = torque_time_ref_s[seq_mask]

        seq_max = float(np.max(seq_torque))
        threshold = thr_frac * seq_max
        above = seq_torque >= threshold

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

        # merge
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

        # duration filter
        final_segments = []
        for s, e in merged:
            if (seq_time[e] - seq_time[s]) >= min_dur_s:
                final_segments.append((s, e))

        vc_number = 0
        for s, e in final_segments:
            vc_number += 1

            global_start = seq_indices_global[s]
            global_end = seq_indices_global[e]

            VC[global_start:global_end+1] = 1
            VC_count[global_start:global_end+1] = vc_number

            event_rows.append({
                "run_id": run_id,
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


    # Apply VC labels to master grid and propagate to torque and EMG via time_index
    master_index_grid_out = master_index_grid.copy()
    master_index_grid_out["VC"] = VC
    master_index_grid_out["VC_count"] = VC_count

    master_VC = master_index_grid_out["VC"].to_numpy(dtype=int)
    master_VC_count = master_index_grid_out["VC_count"].to_numpy(dtype=int)

    torque_compact_df_out = torque_compact_df.copy()
    torque_compact_df_out["VC"] = master_VC[torque_time_index]
    torque_compact_df_out["VC_count"] = master_VC_count[torque_time_index]

    emg_time_index = emg_compact_df["time_index"].to_numpy(dtype=int)

    emg_compact_df_out = emg_compact_df.copy()
    emg_compact_df_out["VC"] = master_VC[emg_time_index]
    emg_compact_df_out["VC_count"] = master_VC_count[emg_time_index]


    # Commit

    knob_rows = []
    for seq_name, knobs in vc_knobs_by_seq.items():
        knob_rows.append({
            "run_id": run_id,
            "row_type": "knob",
            "seq": seq_name,
            "thr_frac": float(knobs["thr_frac"]),
            "merge_gap_s": float(knobs["merge_gap_s"]),
            "min_dur_s": float(knobs["min_dur_s"]),
        })

    vc_committed_knobs_df = pd.DataFrame(knob_rows)
    vc_knobs_plus_events_df = pd.concat(
        [vc_committed_knobs_df, vc_events_df],
        ignore_index=True,
        sort=False,
    )

    vc_knobs_plus_events_df.to_parquet(cache_path, index=False)
    print("[04_VC_Detection] Please adjust the knobs and reload this cell until each contraction is properly detected :\n"
    "WU = variable, depends on the number of repetitions the subject needed to properly warm-up.\n"
    "MVC_REF = MUST contain 3 or 4 repetitions. No more, no less.\n"
    "SVC_REF = Always contain 3 repetitions, each growing in amplitude \n"
    "EX_DYN = Always contain 62 repetitions. Amplitude mostly decreasing as time goes \n"
    "MVC_RECOV = Both Recovery MVC is 1 rep.\n"
    "EX_STA = 1 sustained rep (1 VC) of 60s\n"
    "When set, move on to the next cell to commit into cache and apply the VC to the already existing DataFrame.")


    # QC Plot

    plot_decim = 10 

    torque_time_ref_s_plot = torque_time_ref_s[::plot_decim]
    torque_values_plot = torque_values[::plot_decim]
    torque_time_index_plot = torque_time_index[::plot_decim]

    seq_labels_plot = seq_labels_full[torque_time_index_plot]

    fig, axes = plt.subplots(
        len(vc_knobs_by_seq),
        1,
        figsize=(14, 2.2 * len(vc_knobs_by_seq))
    )

    if len(vc_knobs_by_seq) == 1:
        axes = [axes]

    vc_events_by_seq = (
        {seq: df for seq, df in vc_events_df.groupby("seq")}
        if len(vc_events_df) > 0 else {}
    )

    for ax, seq_name in zip(axes, vc_knobs_by_seq.keys()):

        mask = (seq_labels_plot == seq_name)
        if not np.any(mask):
            ax.set_title(f"{seq_name} (missing)")
            ax.axis("off")
            continue

        ax.plot(
            torque_time_ref_s_plot[mask],
            torque_values_plot[mask],
            linewidth=1,
        )

        seq_full_mask = (seq_labels_full[torque_time_index] == seq_name)
        if np.any(seq_full_mask):
            full_seq_torque = torque_values[seq_full_mask]
            thr = vc_knobs_by_seq[seq_name]["thr_frac"] * np.max(full_seq_torque)
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
            f"{seq_name} | thr={vc_knobs_by_seq[seq_name]['thr_frac']:.3f} | "
            f"merge={vc_knobs_by_seq[seq_name]['merge_gap_s']:.2f}s | "
            f"min_dur={vc_knobs_by_seq[seq_name]['min_dur_s']:.2f}s | "
            f"n_VC={n_vc}"
        )

        ax.set_ylabel("Torque (raw)")

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

    return master_index_grid_out, torque_compact_df_out, emg_compact_df_out, vc_events_df