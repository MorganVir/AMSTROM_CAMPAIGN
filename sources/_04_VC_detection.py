# _04_VC_detection.py - VOLUNTARY CONTRACTION DETECTION
# Detect and label individual voluntary contractions (VCs) from the torque signal.
# Each VC is a sustained peak above a per-sequence threshold; the operator reviews
# the QC plot and tunes knobs until every contraction is correctly isolated.
#
# Inputs
#   ctx keys  : RUN_ID, CACHE_DIR, master_index_grid, torque_compact_df, emg_compact_df
#               TORQUE_COL (optional, default "torque_raw")
#   parameter : vc_knobs_by_seq — per-SEQ threshold and timing knobs (see docstring)
#               force_recompute — if True, ignores the cache and re-runs detection
#
# Outputs (the next notebook cell writes these to ctx after the operator validates)
#   - master_index_grid  : VC (0/1 flag) and VC_count (contraction index within SEQ)
#   - torque_compact_df  : same columns propagated via time_index
#   - emg_compact_df     : same columns propagated via time_index
#
# Cache
#   - 04_vc_knobs_events.parquet — two row types stored in the same file:
#       "knob"  rows: the per-SEQ parameter values used for this detection run
#       "event" rows: one row per detected VC with timing, duration, and threshold
#     Storing both together keeps the detection parameters permanently auditable
#     alongside the detected events.
#
# Algorithm (per SEQ)
#   1. Compute threshold = thr_frac × max(torque in SEQ)
#      e.g. thr_frac=0.10, peak torque=450 Nm → threshold = 45 Nm
#   2. Find all above-threshold segments (rising/falling edge detection via np.diff)
#   3. Merge consecutive segments whose gap is smaller than merge_gap_s
#      (brief dips below threshold during a sustained contraction are artefacts)
#   4. Discard merged segments shorter than min_dur_s (isolated noise spikes)
#   5. Stamp remaining segments as VCs: VC=1, VC_count = 1-based index within SEQ
#
# Notes
#   - The operator must inspect the QC plot and tune the knobs before the next
#     notebook cell commits the result to the master grid.
#   - VC_count resets to 1 at the start of each SEQ; it counts contractions within
#     that sequence only, not across the whole session.
#   - The next cell writes the *_out DataFrames back to ctx after the operator
#     confirms the result, so a rejected detection run leaves ctx unchanged.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Downsampling factor applied to the torque signal when drawing the QC plot.
# Full-resolution torque (~100 Hz) renders slowly and makes individual contractions
# hard to read; dividing by 10 keeps the shape clearly visible at interactive speed.
QC_PLOT_DECIMATION = 10


def run_voluntary_contraction_detection(
    ctx,
    vc_knobs_by_seq,
    force_recompute,
):
    """
    Detect voluntary contractions (VCs) from the torque signal and display a QC plot.

    Detection is automatic (threshold + gap merge + duration filter) but requires
    human sign-off: the operator inspects the QC plot, tunes vc_knobs_by_seq, and
    re-runs the cell until every contraction is correctly isolated. The next notebook
    cell applies the result to the master grid and commits it to ctx.

    vc_knobs_by_seq format — one entry per sequence that contains contractions:
        {
            "SEQ_NAME": {
                "thr_frac":    float,  # fraction of SEQ max torque used as threshold
                                       # e.g. 0.10 → threshold = 10 % of peak torque
                "merge_gap_s": float,  # segments closer than this (in seconds) are merged
                "min_dur_s":   float,  # segments shorter than this (in seconds) are dropped
            },
            ...
        }

    Expected contraction counts by SEQ (use these to judge whether knobs need tuning):
        WU        : variable — depends on warm-up repetitions needed by the subject
        MVC_REF   : exactly 3 or 4
        SVC_REF   : 3 (increasing amplitude)
        EX_DYN    : 62 (amplitude generally decreasing over time)
        MVC_RECOV : 1 per recovery block
        EX_STA    : 1 sustained contraction (~60 s)

    Returns (master_index_grid_out, torque_compact_df_out, emg_compact_df_out, vc_events_df).
    The *_out DataFrames carry VC and VC_count columns but are NOT written to ctx here;
    the next notebook cell does that after the operator confirms the detection looks correct.
    """

    # --- Resolve ctx ---
    run_id            = ctx["RUN_ID"]
    cache_dir         = ctx["CACHE_DIR"]
    master_index_grid = ctx["master_index_grid"]
    torque_compact_df = ctx["torque_compact_df"]
    emg_compact_df    = ctx["emg_compact_df"]
    # Optional override: allows swapping torque_raw for a pre-filtered column if needed.
    TORQUE_COL = ctx.get("TORQUE_COL", "torque_raw")

    cache_path = cache_dir / "04_vc_knobs_events.parquet"

    # --- Cache check ---
    # The cache file can exist with only "knob" rows if a previous run was interrupted
    # before detection completed. A valid cache hit requires "event" rows to be present.
    if cache_path.exists() and not force_recompute:
        df = pd.read_parquet(cache_path)
        if "row_type" in df.columns and df["row_type"].eq("event").any():
            print(
                "[04_VC_detection] Cache found with existing VC events — loaded from cache.\n"
                "  Set force_recompute=True or delete 04_vc_knobs_events.parquet to re-detect.\n"
                "  Run the next cell to apply the cached events to the master grid."
            )
            return master_index_grid, torque_compact_df, emg_compact_df, df

    # --- Prepare data ---
    # Expand the master grid arrays once so the per-SEQ loop can index into them cheaply.
    time_ref_s_full = master_index_grid["time_ref_s"].to_numpy(dtype=float)
    seq_labels_full = master_index_grid["SEQ"].to_numpy(dtype=object)

    torque_time_index = torque_compact_df["time_index"].to_numpy(dtype=int)
    torque_time_ref_s = time_ref_s_full[torque_time_index]
    torque_values     = torque_compact_df[TORQUE_COL].to_numpy(dtype=float)

    # VC and VC_count are built as full master-grid arrays (one slot per grid point).
    # Initialised to 0: grid points outside any detected contraction remain 0.
    VC       = np.zeros(len(master_index_grid), dtype=int)
    VC_count = np.zeros(len(master_index_grid), dtype=int)

    event_rows = []

    # --- Detection per SEQ ---
    for seq_name, knobs in vc_knobs_by_seq.items():

        thr_frac    = float(knobs["thr_frac"])
        merge_gap_s = float(knobs["merge_gap_s"])
        min_dur_s   = float(knobs["min_dur_s"])

        # Restrict to torque samples that belong to this sequence.
        seq_mask = (seq_labels_full[torque_time_index] == seq_name)
        if not np.any(seq_mask):
            continue

        seq_indices_global = torque_time_index[seq_mask]
        seq_torque         = torque_values[seq_mask]
        seq_time           = torque_time_ref_s[seq_mask]

        seq_max   = float(np.max(seq_torque))
        threshold = thr_frac * seq_max
        above     = seq_torque >= threshold

        # Rising edges (+1 in diff) mark segment starts; falling edges (-1) mark ends.
        # +1 on rising-edge positions corrects the one-sample offset from np.diff:
        # diff[i] = above[i+1] − above[i], so a rise at diff[i] means the segment
        # actually starts at index i+1 in the original array.
        diff_mask   = np.diff(above.astype(int))
        start_local = np.where(diff_mask ==  1)[0] + 1
        end_local   = np.where(diff_mask == -1)[0]

        # Edge case: signal is already above threshold at sample 0 — prepend index 0.
        if above.size > 0 and above[0]:
            start_local = np.r_[0, start_local]
        # Edge case: signal is still above threshold at the last sample — append last index.
        if above.size > 0 and above[-1]:
            end_local = np.r_[end_local, len(above) - 1]

        segments = list(zip(start_local.tolist(), end_local.tolist()))
        if not segments:
            continue

        # Merge consecutive segments whose gap is smaller than merge_gap_s.
        # Brief dips below threshold during a sustained contraction are measurement
        # artefacts; merging prevents them from splitting one contraction into two.
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

        # Discard merged segments shorter than min_dur_s — isolated noise spikes.
        final_segments = [
            (s, e) for s, e in merged
            if (seq_time[e] - seq_time[s]) >= min_dur_s
        ]

        # Stamp each surviving segment onto the VC and VC_count arrays.
        # vc_number is 1-based and resets to 1 for the first contraction of each SEQ.
        for vc_number, (s, e) in enumerate(final_segments, 1):
            global_start = seq_indices_global[s]
            global_end   = seq_indices_global[e]

            VC[global_start : global_end + 1]       = 1
            VC_count[global_start : global_end + 1] = vc_number

            event_rows.append({
                "run_id":           run_id,
                "row_type":         "event",
                "seq":              seq_name,
                "vc_in_seq":        vc_number,
                "start_time_index": int(global_start),
                "stop_time_index":  int(global_end),
                "start_time_ref_s": float(seq_time[s]),
                "stop_time_ref_s":  float(seq_time[e]),
                "dur_s":            float(seq_time[e] - seq_time[s]),
                "threshold":        threshold,
                "seq_max":          seq_max,
            })

    vc_events_df = pd.DataFrame(event_rows)

    # --- Apply VC labels ---
    # Build new DataFrames with VC columns added; the originals in ctx stay unchanged.
    # The next notebook cell writes the results to ctx after the operator confirms them.
    master_index_grid_out = master_index_grid.copy()
    master_index_grid_out["VC"]       = VC
    master_index_grid_out["VC_count"] = VC_count

    master_VC       = master_index_grid_out["VC"].to_numpy(dtype=int)
    master_VC_count = master_index_grid_out["VC_count"].to_numpy(dtype=int)

    torque_compact_df_out             = torque_compact_df.copy()
    torque_compact_df_out["VC"]       = master_VC[torque_time_index]
    torque_compact_df_out["VC_count"] = master_VC_count[torque_time_index]

    emg_time_index                = emg_compact_df["time_index"].to_numpy(dtype=int)
    emg_compact_df_out            = emg_compact_df.copy()
    emg_compact_df_out["VC"]      = master_VC[emg_time_index]
    emg_compact_df_out["VC_count"] = master_VC_count[emg_time_index]

    # --- Commit to cache ---
    # Knob rows and event rows are written together so the parameters used for this
    # detection run remain permanently auditable alongside the detected contractions.
    knob_rows = [
        {
            "run_id":      run_id,
            "row_type":    "knob",
            "seq":         seq_name,
            "thr_frac":    float(knobs["thr_frac"]),
            "merge_gap_s": float(knobs["merge_gap_s"]),
            "min_dur_s":   float(knobs["min_dur_s"]),
        }
        for seq_name, knobs in vc_knobs_by_seq.items()
    ]

    vc_knobs_plus_events_df = pd.concat(
        [pd.DataFrame(knob_rows), vc_events_df],
        ignore_index=True,
        sort=False,
    )
    vc_knobs_plus_events_df.to_parquet(cache_path, index=False)

    print(
        "[04_VC_detection] Adjust knobs and re-run until each contraction is correctly detected.\n"
        "  WU        : variable — depends on warm-up repetitions needed by the subject\n"
        "  MVC_REF   : must contain exactly 3 or 4 repetitions\n"
        "  SVC_REF   : always 3 repetitions, increasing in amplitude\n"
        "  EX_DYN    : always 62 repetitions, amplitude generally decreasing over time\n"
        "  MVC_RECOV : 1 repetition per recovery block\n"
        "  EX_STA    : 1 sustained contraction (~60 s)\n"
        "  When satisfied, move to the next cell to apply VC labels to the master grid."
    )

    # --- QC plot ---
    # One subplot per SEQ; each shows the decimated torque trace, the detection threshold
    # (dashed line), and shaded spans with contraction numbers for every detected VC.
    plt.close("all")

    torque_time_ref_s_plot = torque_time_ref_s[::QC_PLOT_DECIMATION]
    torque_values_plot     = torque_values[::QC_PLOT_DECIMATION]
    torque_time_index_plot = torque_time_index[::QC_PLOT_DECIMATION]
    seq_labels_plot        = seq_labels_full[torque_time_index_plot]

    fig, axes = plt.subplots(
        len(vc_knobs_by_seq),
        1,
        figsize=(14, 2.2 * len(vc_knobs_by_seq)),
    )
    if len(vc_knobs_by_seq) == 1:
        axes = [axes]

    # Pre-group events by sequence for O(1) lookup inside the per-axes loop.
    vc_events_by_seq = (
        {seq: grp for seq, grp in vc_events_df.groupby("seq")}
        if len(vc_events_df) > 0
        else {}
    )

    for ax, seq_name in zip(axes, vc_knobs_by_seq.keys()):

        mask = (seq_labels_plot == seq_name)
        if not np.any(mask):
            ax.set_title(f"{seq_name} (no torque data)")
            ax.axis("off")
            continue

        ax.plot(torque_time_ref_s_plot[mask], torque_values_plot[mask], linewidth=1)

        # Recompute the threshold from full-resolution data to match what the algorithm
        # actually used — the decimated plot array would give a slightly different max.
        seq_full_mask = (seq_labels_full[torque_time_index] == seq_name)
        if np.any(seq_full_mask):
            thr = vc_knobs_by_seq[seq_name]["thr_frac"] * np.max(torque_values[seq_full_mask])
            ax.axhline(thr, linestyle="--", linewidth=1)

        # vc_events_df.iloc[0:0] produces an empty DataFrame with the correct columns,
        # so the loop below works without any special-casing when no VCs were detected.
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
