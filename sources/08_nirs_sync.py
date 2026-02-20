# ============================================================
# 08_nirs_sync — estimate lag in TARGET SEQ using real NIRS samples only
# - Objective (Option B, robust): maximize |mean(z in VC) - mean(z out VC)| within anchor SEQ
# - NO padding / NO sentinel / NO clipping: out-of-range after shift => DROP (reported)
# - Apply best shift by shifting time_index, then relabel from master
# - QC plot BEFORE vs AFTER (optional)
# Cache (COMMIT only): nirs_compact_aligned.parquet + nirs_sync_report.parquet under CACHE_08_NIRS_SYNC.parent
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("### RUNNING 08_nirs_sync: BIA-STYLE VC-DISCRIM VERSION ###")

# ----------------------------
# REQUIRED INPUTS (fail loud)
# ----------------------------
assert "RUN_ID" in globals() and isinstance(RUN_ID, str) and RUN_ID, "RUN_ID missing/invalid"

assert "master_index_grid" in globals() and isinstance(master_index_grid, pd.DataFrame), "master_index_grid missing/invalid"
for c in ["time_ref_s", "time_index", "SEQ", "SEQ_index", "VC", "VC_count"]:
    assert c in master_index_grid.columns, f"master_index_grid missing required column: {c}"

assert "nirs_compact_df" in globals() and isinstance(nirs_compact_df, pd.DataFrame), "nirs_compact_df missing/invalid"
for c in ["time_index", "SEQ_index", "VC", "VC_count"]:
    assert c in nirs_compact_df.columns, f"nirs_compact_df missing required column: {c}"

# knobs (dashboard-owned)
assert "NIRS_SYNC_TARGET_SEQ" in globals(), "NIRS_SYNC_TARGET_SEQ missing"
assert "NIRS_SYNC_SIGNAL_COL" in globals(), "NIRS_SYNC_SIGNAL_COL missing (which NIRS column to score)"
assert "NIRS_SYNC_LAG_MIN_S" in globals(), "NIRS_SYNC_LAG_MIN_S missing"
assert "NIRS_SYNC_LAG_MAX_S" in globals(), "NIRS_SYNC_LAG_MAX_S missing"
assert "NIRS_SYNC_STEP_S" in globals(), "NIRS_SYNC_STEP_S missing"
assert "NIRS_MANUAL_NUDGE_S" in globals(), "NIRS_MANUAL_NUDGE_S missing"
assert "NIRS_SYNC_PLOT" in globals(), "NIRS_SYNC_PLOT missing"
assert "NIRS_SYNC_COMMIT" in globals(), "NIRS_SYNC_COMMIT missing"
# require torque col for QC 
assert "torque_compact_df" in globals() and isinstance(torque_compact_df, pd.DataFrame), "torque_compact_df missing/invalid (needed for QC)"
assert "TORQUE_COL" in globals() and isinstance(TORQUE_COL, str), "TORQUE_COL missing/invalid"
assert TORQUE_COL in torque_compact_df.columns, f"TORQUE_COL '{TORQUE_COL}' not in torque_compact_df"
assert "time_index" in torque_compact_df.columns, "torque_compact_df missing time_index"


assert isinstance(NIRS_SYNC_SIGNAL_COL, str) and NIRS_SYNC_SIGNAL_COL in nirs_compact_df.columns, \
    f"NIRS_SYNC_SIGNAL_COL '{NIRS_SYNC_SIGNAL_COL}' not in nirs_compact_df"

# optional torque for QC overlay (recommended but not required for scoring)
TORQUE_FOR_QC = False
if "torque_compact_df" in globals() and isinstance(torque_compact_df, pd.DataFrame):
    if "time_index" in torque_compact_df.columns:
        TORQUE_FOR_QC = True

assert "CACHE_08_NIRS_SYNC" in globals() and isinstance(CACHE_08_NIRS_SYNC, Path), "CACHE_08_NIRS_SYNC missing/invalid"
CACHE_08_NIRS_SYNC.parent.mkdir(parents=True, exist_ok=True)

cache_aligned_path = CACHE_08_NIRS_SYNC.parent / "nirs_compact_aligned.parquet"
cache_report_path  = CACHE_08_NIRS_SYNC.parent / "nirs_sync_report.parquet"

# ----------------------------
# Cache hit (only when COMMIT expected outputs exist)
# ----------------------------
if cache_aligned_path.exists() and cache_report_path.exists():
    nirs_compact_aligned_df_out = pd.read_parquet(cache_aligned_path)
    nirs_sync_report_df_out = pd.read_parquet(cache_report_path)
    print("[08_nirs_sync] cache hit:", cache_aligned_path.name)
else:
    master_time_ref_s = master_index_grid["time_ref_s"].to_numpy(dtype=float)
    median_dt = float(np.median(np.diff(master_time_ref_s)))
    assert median_dt > 0, "master_index_grid.time_ref_s median dt must be > 0"
    master_fs_hz = 1.0 / median_dt

    # ----------------------------
    # Build anchor masks on master
    # ----------------------------
    anchor_seq = str(NIRS_SYNC_TARGET_SEQ)
    master_anchor_mask = (master_index_grid["SEQ"].astype(str).to_numpy() == anchor_seq)
    assert np.any(master_anchor_mask), f"No master samples found for anchor SEQ='{anchor_seq}'"

    # ----------------------------
    # Candidate lags (seconds -> samples)
    # ----------------------------
    lag_min_s = float(NIRS_SYNC_LAG_MIN_S)
    lag_max_s = float(NIRS_SYNC_LAG_MAX_S)
    lag_step_s = float(NIRS_SYNC_STEP_S)
    assert lag_step_s > 0, "NIRS_SYNC_STEP_S must be > 0"
    assert lag_max_s >= lag_min_s, "NIRS_SYNC_LAG_MAX_S must be >= NIRS_SYNC_LAG_MIN_S"

    lag_candidates_s = np.arange(lag_min_s, lag_max_s + 1e-12, lag_step_s, dtype=float)
    shift_candidates_samples = np.rint(lag_candidates_s * master_fs_hz).astype(int)

    # ----------------------------
    # Scoring function (Option B robust)
    # ----------------------------
    nirs_ti = nirs_compact_df["time_index"].to_numpy(dtype=int)
    nirs_sig = nirs_compact_df[NIRS_SYNC_SIGNAL_COL].to_numpy(dtype=float)

    def score_for_shift(shift_samples: int) -> float:
        ti_shifted = nirs_ti + int(shift_samples)

        # in master range only
        in_range = (ti_shifted >= 0) & (ti_shifted < len(master_index_grid))
        if not np.any(in_range):
            return -np.inf

        ti_ok = ti_shifted[in_range]
        sig_ok = nirs_sig[in_range]

        # anchor window selection using master SEQ labels
        anchor_ok = master_anchor_mask[ti_ok]
        if np.sum(anchor_ok) < 10:
            return -np.inf

        ti_a = ti_ok[anchor_ok]
        sig_a = sig_ok[anchor_ok]

        # VC split using master VC_count (robust to labeling)
        vc_count = master_index_grid.loc[ti_a, "VC_count"].to_numpy()
        in_vc = vc_count > 0
        out_vc = ~in_vc
        if np.sum(in_vc) < 5 or np.sum(out_vc) < 5:
            return -np.inf

        # z-score within anchor window (scale-robust)
        mu = float(np.nanmean(sig_a))
        sd = float(np.nanstd(sig_a))
        if not np.isfinite(sd) or sd == 0.0:
            return -np.inf
        z = (sig_a - mu) / sd

        # robust discrimination (sign-robust)
        return float(abs(np.nanmean(z[in_vc]) - np.nanmean(z[out_vc])))

    scores = np.array([score_for_shift(s) for s in shift_candidates_samples], dtype=float)
    best_idx = int(np.nanargmax(scores))
    best_shift_samples_auto = int(shift_candidates_samples[best_idx])
    best_lag_s_auto = float(lag_candidates_s[best_idx])
    score_peak = float(scores[best_idx])

    # ----------------------------
    # Apply manual nudge
    # ----------------------------
    manual_nudge_s = float(NIRS_MANUAL_NUDGE_S)
    total_lag_s = float(best_lag_s_auto + manual_nudge_s)
    total_shift_samples = int(np.rint(total_lag_s * master_fs_hz))

    df_aligned = nirs_compact_df.copy()
    df_aligned["time_index"] = df_aligned["time_index"].to_numpy(dtype=int) + total_shift_samples

    # drop out-of-range
    in_range = (df_aligned["time_index"] >= 0) & (df_aligned["time_index"] < len(master_index_grid))
    drop_low = int(np.sum(df_aligned["time_index"] < 0))
    drop_high = int(np.sum(df_aligned["time_index"] >= len(master_index_grid)))
    df_aligned = df_aligned.loc[in_range].copy()

    # relabel from master
    ti = df_aligned["time_index"].to_numpy(dtype=int)
    df_aligned["SEQ_index"] = master_index_grid.loc[ti, "SEQ_index"].to_numpy()
    df_aligned["VC"] = master_index_grid.loc[ti, "VC"].to_numpy()
    df_aligned["VC_count"] = master_index_grid.loc[ti, "VC_count"].to_numpy()

    # ----------------------------
    # Report (explicit schema)
    # ----------------------------
    nirs_sync_report_df_out = pd.DataFrame([{
        "RUN_ID": RUN_ID,
        "anchor_seq": anchor_seq,
        "signal_col": NIRS_SYNC_SIGNAL_COL,
        "fs_ref_hz": float(master_fs_hz),

        "lag_min_s": float(lag_min_s),
        "lag_max_s": float(lag_max_s),
        "lag_step_s": float(lag_step_s),

        "best_lag_s_auto": float(best_lag_s_auto),
        "best_shift_samples_auto": int(best_shift_samples_auto),

        "manual_nudge_s": float(manual_nudge_s),
        "total_lag_s": float(total_lag_s),
        "total_shift_samples": int(total_shift_samples),

        "drop_low": int(drop_low),
        "drop_high": int(drop_high),
        "score_peak": float(score_peak),
        "n_in_aligned": int(len(df_aligned)),
    }])

    nirs_compact_aligned_df_out = df_aligned

# ----------------------------
# QC plot (before vs after)
# ----------------------------
if bool(NIRS_SYNC_PLOT):
    fig = plt.figure(figsize=(14, 10))

    # Marker tuning (small + clean)
    marker_size_overview = 1.6
    markevery_overview = 8

    # NIRS colors (keep as you requested)
    nirs_before_color = "blue"
    nirs_after_color  = "red"

    # VC box style: fill has alpha, border is fully opaque
    vc_fill_alpha = 0.18
    vc_edge_lw = 2.0
    vc_face_rgba = (1.0, 0.5, 0.0, vc_fill_alpha)  # orange with alpha
    vc_edge_rgba = (1.0, 0.5, 0.0, 1.0)            # orange fully opaque

    # ----------------------------
    # Anchor window on master
    # ----------------------------
    anchor_time_indices = np.where(master_anchor_mask)[0]
    assert len(anchor_time_indices) > 10, f"Anchor SEQ '{anchor_seq}' too short or missing"

    anchor_start_index = int(anchor_time_indices[0])
    anchor_end_index   = int(anchor_time_indices[-1])

    anchor_start_time_s = float(master_time_ref_s[anchor_start_index])
    anchor_end_time_s   = float(master_time_ref_s[anchor_end_index])

    # ----------------------------
    # VC spans inside anchor window
    # ----------------------------
    master_vc_mask = master_index_grid["VC_count"].to_numpy() > 0

    vc_spans_in_anchor = []
    in_vc_segment = False
    vc_segment_start_index = None

    for master_idx in range(anchor_start_index, anchor_end_index + 1):
        if master_vc_mask[master_idx] and not in_vc_segment:
            in_vc_segment = True
            vc_segment_start_index = master_idx

        if (not master_vc_mask[master_idx]) and in_vc_segment:
            vc_spans_in_anchor.append((vc_segment_start_index, master_idx - 1))
            in_vc_segment = False
            vc_segment_start_index = None

    if in_vc_segment:
        vc_spans_in_anchor.append((vc_segment_start_index, anchor_end_index))

    assert len(vc_spans_in_anchor) > 0, "No VC spans detected inside anchor window."

    # ----------------------------
    # Prepare signals (overview)
    # ----------------------------
    # Torque
    torque_time_indices = torque_compact_df["time_index"].to_numpy(dtype=int)
    torque_time_s = master_time_ref_s[torque_time_indices]
    torque_values = torque_compact_df[TORQUE_COL].to_numpy(dtype=float)

    # NIRS BEFORE
    nirs_before_time_indices = nirs_compact_df["time_index"].to_numpy(dtype=int)
    nirs_before_time_s = master_time_ref_s[nirs_before_time_indices]
    nirs_before_values = nirs_compact_df[NIRS_SYNC_SIGNAL_COL].to_numpy(dtype=float)
    sort_order_before = np.argsort(nirs_before_time_s)
    nirs_before_time_s = nirs_before_time_s[sort_order_before]
    nirs_before_values = nirs_before_values[sort_order_before]

    # NIRS AFTER
    nirs_after_time_indices = nirs_compact_aligned_df_out["time_index"].to_numpy(dtype=int)
    nirs_after_time_s = master_time_ref_s[nirs_after_time_indices]
    nirs_after_values = nirs_compact_aligned_df_out[NIRS_SYNC_SIGNAL_COL].to_numpy(dtype=float)
    sort_order_after = np.argsort(nirs_after_time_s)
    nirs_after_time_s = nirs_after_time_s[sort_order_after]
    nirs_after_values = nirs_after_values[sort_order_after]

    # Robust NIRS y-lims + padding (avoid clipping at plot edges)
    nirs_all_values = np.concatenate([nirs_before_values, nirs_after_values])
    nirs_p01 = float(np.nanpercentile(nirs_all_values, 1))
    nirs_p99 = float(np.nanpercentile(nirs_all_values, 99))
    use_nirs_ylim = np.isfinite(nirs_p01) and np.isfinite(nirs_p99) and (nirs_p99 > nirs_p01)

    if use_nirs_ylim:
        nirs_span = nirs_p99 - nirs_p01
        nirs_pad = 0.05 * nirs_span
        nirs_ylim_lo = nirs_p01 - nirs_pad
        nirs_ylim_hi = nirs_p99 + nirs_pad

    # ----------------------------
    # SUBPLOT 1 — Overview OVERLAY (full session)
    # ----------------------------
    ax1_torque = fig.add_subplot(3, 1, 1)
    ax1_torque.set_title(
        f"{RUN_ID} - NIRS sync overview (anchor={anchor_seq}) - OVERLAY "
        f"| auto={best_lag_s_auto:.3f}s | manual={manual_nudge_s:.3f}s | total={total_lag_s:.3f}s"
    )

    # subtle anchor shading
    ax1_torque.axvspan(anchor_start_time_s, anchor_end_time_s, facecolor=(0, 0, 0, 0.05), edgecolor="none")

    # VC boxes (fill alpha on face only, border fully opaque)
    first_vc_label = True
    for span_start_idx, span_end_idx in vc_spans_in_anchor:
        span_start_time = master_time_ref_s[span_start_idx]
        span_end_time   = master_time_ref_s[span_end_idx]
        if first_vc_label:
            ax1_torque.axvspan(
                span_start_time, span_end_time,
                facecolor=vc_face_rgba,
                edgecolor=vc_edge_rgba,
                linewidth=vc_edge_lw,
                label="VC",
            )
            first_vc_label = False
        else:
            ax1_torque.axvspan(
                span_start_time, span_end_time,
                facecolor=vc_face_rgba,
                edgecolor=vc_edge_rgba,
                linewidth=vc_edge_lw,
            )

    # torque left axis (keep default style like BIA)
    ax1_torque.plot(
        torque_time_s,
        torque_values,
        linewidth=0.9,
        label=f"{TORQUE_COL} (compact)",
    )
    ax1_torque.set_ylabel("torque")
    ax1_torque.set_xlabel("time_ref_s (s)")

    # NIRS right axis
    ax1_nirs = ax1_torque.twinx()
    ax1_nirs.plot(
        nirs_before_time_s,
        nirs_before_values,
        linewidth=1.0,
        color=nirs_before_color,
        marker="o",
        markersize=marker_size_overview,
        markeredgewidth=0.0,
        markevery=markevery_overview,
        label=f"NIRS BEFORE ({NIRS_SYNC_SIGNAL_COL})",
    )
    ax1_nirs.plot(
        nirs_after_time_s,
        nirs_after_values,
        linewidth=1.0,
        color=nirs_after_color,
        marker="o",
        markersize=marker_size_overview,
        markeredgewidth=0.0,
        markevery=markevery_overview,
        label="NIRS AFTER",
    )
    ax1_nirs.set_ylabel("NIRS")
    if use_nirs_ylim:
        ax1_nirs.set_ylim(nirs_ylim_lo, nirs_ylim_hi)

    # combined legend (TOP LEFT)
    h1, l1 = ax1_torque.get_legend_handles_labels()
    h2, l2 = ax1_nirs.get_legend_handles_labels()
    ax1_torque.legend(h1 + h2, l1 + l2, loc="upper left")

    # ----------------------------
    # SUBPLOT 2 — BEFORE only (ZOOM on anchor) — NO LEGEND
    # ----------------------------
    ax2_torque = fig.add_subplot(3, 1, 2)
    ax2_torque.set_title("BEFORE (unsynced) — Zoom on anchor")

    # VC boxes again
    for span_start_idx, span_end_idx in vc_spans_in_anchor:
        span_start_time = master_time_ref_s[span_start_idx]
        span_end_time   = master_time_ref_s[span_end_idx]
        ax2_torque.axvspan(
            span_start_time, span_end_time,
            facecolor=vc_face_rgba,
            edgecolor=vc_edge_rgba,
            linewidth=vc_edge_lw,
        )

    # torque zoom
    torque_in_anchor_mask = (torque_time_indices >= anchor_start_index) & (torque_time_indices <= anchor_end_index)
    ax2_torque.plot(
        master_time_ref_s[torque_time_indices[torque_in_anchor_mask]],
        torque_values[torque_in_anchor_mask],
        linewidth=0.9,
    )
    ax2_torque.set_xlim(anchor_start_time_s, anchor_end_time_s)
    ax2_torque.set_ylabel("torque")
    ax2_torque.set_xlabel("time_ref_s (s)")

    # NIRS BEFORE zoom on right axis
    ax2_nirs = ax2_torque.twinx()
    nirs_before_in_anchor_mask = (nirs_before_time_indices >= anchor_start_index) & (nirs_before_time_indices <= anchor_end_index)
    zoom_before_time_s = master_time_ref_s[nirs_before_time_indices[nirs_before_in_anchor_mask]]
    zoom_before_values = nirs_compact_df.loc[nirs_before_in_anchor_mask, NIRS_SYNC_SIGNAL_COL].to_numpy(dtype=float)
    sort_order_zoom_before = np.argsort(zoom_before_time_s)

    ax2_nirs.plot(
        zoom_before_time_s[sort_order_zoom_before],
        zoom_before_values[sort_order_zoom_before],
        linewidth=1.1,
        color=nirs_before_color,
        marker="o",
        markersize=marker_size_overview,
        markeredgewidth=0.0,
        markevery=max(1, markevery_overview // 2),
    )
    ax2_nirs.set_ylabel("NIRS")

    z01 = float(np.nanpercentile(zoom_before_values, 1))
    z99 = float(np.nanpercentile(zoom_before_values, 99))
    if np.isfinite(z01) and np.isfinite(z99) and (z99 > z01):
        zspan = z99 - z01
        zpad = 0.05 * zspan
        ax2_nirs.set_ylim(z01 - zpad, z99 + zpad)

    # ----------------------------
    # SUBPLOT 3 — AFTER only (ZOOM on anchor) — NO LEGEND
    # ----------------------------
    ax3_torque = fig.add_subplot(3, 1, 3)
    ax3_torque.set_title("AFTER (synced) — Zoom on anchor")

    # VC boxes again
    for span_start_idx, span_end_idx in vc_spans_in_anchor:
        span_start_time = master_time_ref_s[span_start_idx]
        span_end_time   = master_time_ref_s[span_end_idx]
        ax3_torque.axvspan(
            span_start_time, span_end_time,
            facecolor=vc_face_rgba,
            edgecolor=vc_edge_rgba,
            linewidth=vc_edge_lw,
        )

    # torque zoom
    ax3_torque.plot(
        master_time_ref_s[torque_time_indices[torque_in_anchor_mask]],
        torque_values[torque_in_anchor_mask],
        linewidth=0.9,
    )
    ax3_torque.set_xlim(anchor_start_time_s, anchor_end_time_s)
    ax3_torque.set_ylabel("torque")
    ax3_torque.set_xlabel("time_ref_s (s)")

    # NIRS AFTER zoom on right axis
    ax3_nirs = ax3_torque.twinx()
    nirs_after_in_anchor_mask = (nirs_after_time_indices >= anchor_start_index) & (nirs_after_time_indices <= anchor_end_index)
    zoom_after_time_s = master_time_ref_s[nirs_after_time_indices[nirs_after_in_anchor_mask]]
    zoom_after_values = nirs_compact_aligned_df_out.loc[nirs_after_in_anchor_mask, NIRS_SYNC_SIGNAL_COL].to_numpy(dtype=float)
    sort_order_zoom_after = np.argsort(zoom_after_time_s)

    ax3_nirs.plot(
        zoom_after_time_s[sort_order_zoom_after],
        zoom_after_values[sort_order_zoom_after],
        linewidth=1.1,
        color=nirs_after_color,
        marker="o",
        markersize=marker_size_overview,
        markeredgewidth=0.0,
        markevery=max(1, markevery_overview // 2),
    )
    ax3_nirs.set_ylabel("NIRS")

    z01 = float(np.nanpercentile(zoom_after_values, 1))
    z99 = float(np.nanpercentile(zoom_after_values, 99))
    if np.isfinite(z01) and np.isfinite(z99) and (z99 > z01):
        zspan = z99 - z01
        zpad = 0.05 * zspan
        ax3_nirs.set_ylim(z01 - zpad, z99 + zpad)

    plt.tight_layout()
    plt.show()

# ----------------------------
# Cache write (COMMIT only)
# ----------------------------
if bool(NIRS_SYNC_COMMIT):
    nirs_compact_aligned_df_out.to_parquet(cache_aligned_path, index=False)
    nirs_sync_report_df_out.to_parquet(cache_report_path, index=False)
    print("[08_nirs_sync] wrote:")
    print("  -", cache_aligned_path.name)
    print("  -", cache_report_path.name)