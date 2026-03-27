# sources/_06b_nirs_sync.py

# from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import numpy as np

def run_nirs_sync(
    *,
    ctx: dict,
    nirs_signal_col: str,
    lag_min_s: float,
    lag_max_s: float,
    manual_nudge_s: float = 0.0,
    torque_col: str = "torque_raw",
    force_recompute: bool = False,
) -> Tuple[pd.DataFrame, Any]:
    """
    06b - NIRS sync (torque <-> NIRS cross-correlation).

    Requires ctx keys:
      RUN_ID, CACHE_DIR, master_index_grid, nirs_compact_df, torque_compact_df

    Returns:
      nirs_compact_aligned_df, fig_or_none
    """

    # asserts
    required = ["RUN_ID", "CACHE_DIR", "master_index_grid", "nirs_compact_df", "torque_compact_df"]
    for k in required:
        if k not in ctx:
            raise KeyError(f"[06b_nirs_sync] missing ctx['{k}']")
        if ctx[k] is None:
            raise ValueError(f"[06b_nirs_sync] ctx['{k}'] is None")

    run_id: str = ctx["RUN_ID"]
    cache_dir: Path = ctx["CACHE_DIR"]

    if not isinstance(cache_dir, Path):
        raise TypeError(f"[06b_nirs_sync] ctx['CACHE_DIR'] must be pathlib.Path, got {type(cache_dir)}")

    master_index_grid = ctx["master_index_grid"]
    nirs_compact_df = ctx["nirs_compact_df"]
    torque_compact_df = ctx["torque_compact_df"]


    # Cache path (owned here)
    cache_path = cache_dir / "06b_nirs_sync.parquet"


    # cache hit
    if cache_path.exists() and (not force_recompute):
        nirs_compact_aligned_df = pd.read_parquet(cache_path)
        fig = None

        # update ctx
        ctx["nirs_compact_aligned_df"] = nirs_compact_aligned_df
        return nirs_compact_aligned_df, fig


    # constants
    RUN_ID = run_id
    master_index_grid = ctx["master_index_grid"]
    nirs_compact_df = ctx["nirs_compact_df"]
    torque_compact_df = ctx["torque_compact_df"]

    target_seq = "MVC_REF"
    NIRS_SYNC_SIGNAL_COL = str(nirs_signal_col)
    NIRS_SYNC_LAG_MIN_S = float(lag_min_s)
    NIRS_SYNC_LAG_MAX_S = float(lag_max_s)
    step_s = 0.05  
    NIRS_MANUAL_NUDGE_S = float(manual_nudge_s)

    TORQUE_COL = str(torque_col)




    # validate required columns 
    for c in ["time_ref_s", "time_index", "SEQ", "SEQ_index", "VC", "VC_count"]:
        if c not in master_index_grid.columns:
            raise AssertionError(f"master_index_grid missing required column: {c}")

    for c in ["time_index", "SEQ_index", "VC", "VC_count"]:
        if c not in nirs_compact_df.columns:
            raise AssertionError(f"nirs_compact_df missing required column: {c}")

    if "time_index" not in torque_compact_df.columns:
        raise AssertionError("torque_compact_df missing time_index")
    if TORQUE_COL not in torque_compact_df.columns:
        raise AssertionError(f"TORQUE_COL '{TORQUE_COL}' not in torque_compact_df")
    if NIRS_SYNC_SIGNAL_COL not in nirs_compact_df.columns:
        raise AssertionError(f"NIRS_SYNC_SIGNAL_COL '{NIRS_SYNC_SIGNAL_COL}' not in nirs_compact_df")

    # cache path (preserve your existing cached artifact name)
    CACHE_NIRS_ALIGNED = cache_dir / "06b_nirs_compact_aligned.parquet"

    # register specific cache paths (so dashboard commit can reuse them without hardcoding )
    ctx.setdefault("parquet_path", {})
    ctx["parquet_path"]["CACHE_NIRS_ALIGNED"] = CACHE_NIRS_ALIGNED
    
    # Cache hit
    if CACHE_NIRS_ALIGNED.exists() and (not force_recompute):
        nirs_compact_aligned_df_out = pd.read_parquet(CACHE_NIRS_ALIGNED)
        fig = None
        ctx["nirs_compact_aligned_df"] = nirs_compact_aligned_df_out
        return nirs_compact_aligned_df_out, fig

    # Compute lag
 
    master_time_ref_s = master_index_grid["time_ref_s"].to_numpy(dtype=float)
    median_dt = float(np.median(np.diff(master_time_ref_s)))
    assert median_dt > 0, "master_index_grid.time_ref_s median dt must be > 0"
    master_fs_hz = 1.0 / median_dt

    # Build anchor masks on master
    anchor_seq = "MVC_REF"
    master_anchor_mask = (master_index_grid["SEQ"].astype(str).to_numpy() == anchor_seq)
    assert np.any(master_anchor_mask), f"No master samples found for anchor SEQ='{anchor_seq}'"

    # Candidate lags (seconds -> samples)
    lag_min_s_ = float(NIRS_SYNC_LAG_MIN_S)
    lag_max_s_ = float(NIRS_SYNC_LAG_MAX_S)
    lag_step_s_ = float(step_s)
    assert lag_step_s_ > 0, "NIRS_SYNC_STEP_S must be > 0"
    assert lag_max_s_ >= lag_min_s_, "NIRS_SYNC_LAG_MAX_S must be >= NIRS_SYNC_LAG_MIN_S"

    lag_candidates_s = np.arange(lag_min_s_, lag_max_s_ + 1e-12, lag_step_s_, dtype=float)
    shift_candidates_samples = np.rint(lag_candidates_s * master_fs_hz).astype(int)

    # Scoring function
    nirs_time_index = nirs_compact_df["time_index"].to_numpy(dtype=int)
    nirs_signal = nirs_compact_df[NIRS_SYNC_SIGNAL_COL].to_numpy(dtype=float)

    master_length = int(len(master_index_grid))
    master_vc_count = master_index_grid["VC_count"].to_numpy(dtype=int)

    master_dt_s = float(np.nanmedian(np.diff(master_time_ref_s)))
    assert np.isfinite(master_dt_s) and master_dt_s > 0.0, "master time_ref_s invalid / non-increasing"
    master_samples_per_second = int(round(1.0 / master_dt_s))
    assert master_samples_per_second > 0, "invalid master_samples_per_second from master dt"

    nirs_slope_window_s = 1.00
    nirs_slope_window_samples = int(round(nirs_slope_window_s * master_samples_per_second))
    assert nirs_slope_window_samples >= 3, "nirs_slope_window_samples too small"

    min_window_samples = 12  # => at least 11 diffs

    def score_for_shift(shift_samples: int) -> float:
        shifted_time_index = nirs_time_index + int(shift_samples)

        in_master = (shifted_time_index >= 0) & (shifted_time_index < master_length)
        if not np.any(in_master):
            return -np.inf

        time_index_ok = shifted_time_index[in_master]
        signal_ok = nirs_signal[in_master]

        anchor_ok = master_anchor_mask[time_index_ok]
        if int(np.sum(anchor_ok)) < 10:
            return -np.inf

        time_index_anchor = time_index_ok[anchor_ok]
        signal_anchor = signal_ok[anchor_ok]

        vc_count_anchor = master_vc_count[time_index_anchor]
        if vc_count_anchor.size < 3:
            return -np.inf

        in_vc = vc_count_anchor > 0
        onset_mask = in_vc & (~np.r_[False, in_vc[:-1]])
        onset_positions = np.flatnonzero(onset_mask)
        if onset_positions.size < 2:
            return -np.inf

        mu = float(np.nanmean(signal_anchor))
        sd = float(np.nanstd(signal_anchor))
        if (not np.isfinite(sd)) or sd == 0.0:
            return -np.inf
        z_anchor = (signal_anchor - mu) / sd

        slope_feature_list = []
        n = int(z_anchor.size)

        for p in onset_positions:
            right = min(n, int(p) + nirs_slope_window_samples)
            if (right - int(p)) < min_window_samples:
                continue

            dz = np.diff(z_anchor[int(p):right])
            if dz.size == 0:
                continue

            weight = np.linspace(1.0, 0.0, dz.size, endpoint=True)

            neg = dz < 0
            if np.any(neg):
                slope_feature = float(np.nanmax((-dz[neg]) * weight[neg]))
            else:
                slope_feature = float(np.nanmax(np.abs(dz) * weight))

            if np.isfinite(slope_feature):
                slope_feature_list.append(slope_feature)

        if len(slope_feature_list) < 2:
            return -np.inf

        return float(np.nanmedian(np.array(slope_feature_list, dtype=float)))

    scores = np.array([score_for_shift(int(s)) for s in shift_candidates_samples], dtype=float)
    n_finite = int(np.isfinite(scores).sum())
    assert n_finite > 0, f"No finite scores across {scores.size} shift candidates (all -inf)."

    best_idx = int(np.nanargmax(scores))
    best_shift_samples_auto = int(shift_candidates_samples[best_idx])
    best_lag_s_auto = float(lag_candidates_s[best_idx])
    score_peak = float(scores[best_idx])

    # Apply manual nudge
    manual_nudge_s_ = float(NIRS_MANUAL_NUDGE_S)
    total_lag_s = float(best_lag_s_auto + manual_nudge_s_)
    total_shift_samples = int(np.rint(total_lag_s * master_fs_hz))

    df_aligned = nirs_compact_df.copy()
    df_aligned["time_index"] = df_aligned["time_index"].to_numpy(dtype=int) + total_shift_samples

    in_range = (df_aligned["time_index"] >= 0) & (df_aligned["time_index"] < len(master_index_grid))
    drop_low = int(np.sum(df_aligned["time_index"] < 0))
    drop_high = int(np.sum(df_aligned["time_index"] >= len(master_index_grid)))
    df_aligned = df_aligned.loc[in_range].copy()

    # relabel from master
    ti = df_aligned["time_index"].to_numpy(dtype=int)
    df_aligned["SEQ_index"] = master_index_grid.loc[ti, "SEQ_index"].to_numpy()
    df_aligned["SEQ"] = master_index_grid.loc[ti, "SEQ"].to_numpy()
    df_aligned["VC"] = master_index_grid.loc[ti, "VC"].to_numpy()
    df_aligned["VC_count"] = master_index_grid.loc[ti, "VC_count"].to_numpy()

    nirs_compact_aligned_df_out = df_aligned


    # QC plot 

    fig = None
    import matplotlib.pyplot as plt
    plt.close("all")
    fig = plt.figure(figsize=(14, 10))

    marker_size_overview = 1.6
    markevery_overview = 8

    nirs_before_color = "blue"
    nirs_after_color = "red"

    vc_fill_alpha = 0.18
    vc_edge_lw = 2.0
    vc_face_rgba = (1.0, 0.5, 0.0, vc_fill_alpha)
    vc_edge_rgba = (1.0, 0.5, 0.0, 1.0)

    anchor_time_indices = np.where(master_anchor_mask)[0]
    assert len(anchor_time_indices) > 10, f"Anchor SEQ '{anchor_seq}' too short or missing"

    anchor_start_index = int(anchor_time_indices[0])
    anchor_end_index = int(anchor_time_indices[-1])

    anchor_start_time_s = float(master_time_ref_s[anchor_start_index])
    anchor_end_time_s = float(master_time_ref_s[anchor_end_index])

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

    torque_time_indices = torque_compact_df["time_index"].to_numpy(dtype=int)
    torque_time_s = master_time_ref_s[torque_time_indices]
    torque_values = torque_compact_df[TORQUE_COL].to_numpy(dtype=float)

    nirs_before_time_indices = nirs_compact_df["time_index"].to_numpy(dtype=int)
    nirs_before_time_s = master_time_ref_s[nirs_before_time_indices]
    nirs_before_values = nirs_compact_df[NIRS_SYNC_SIGNAL_COL].to_numpy(dtype=float)
    sort_order_before = np.argsort(nirs_before_time_s)
    nirs_before_time_s = nirs_before_time_s[sort_order_before]
    nirs_before_values = nirs_before_values[sort_order_before]

    nirs_after_time_indices = nirs_compact_aligned_df_out["time_index"].to_numpy(dtype=int)
    nirs_after_time_s = master_time_ref_s[nirs_after_time_indices]
    nirs_after_values = nirs_compact_aligned_df_out[NIRS_SYNC_SIGNAL_COL].to_numpy(dtype=float)
    sort_order_after = np.argsort(nirs_after_time_s)
    nirs_after_time_s = nirs_after_time_s[sort_order_after]
    nirs_after_values = nirs_after_values[sort_order_after]

    nirs_all_values = np.concatenate([nirs_before_values, nirs_after_values])
    nirs_p01 = float(np.nanpercentile(nirs_all_values, 1))
    nirs_p99 = float(np.nanpercentile(nirs_all_values, 99))
    use_nirs_ylim = np.isfinite(nirs_p01) and np.isfinite(nirs_p99) and (nirs_p99 > nirs_p01)

    if use_nirs_ylim:
        nirs_span = nirs_p99 - nirs_p01
        nirs_pad = 0.05 * nirs_span
        nirs_ylim_lo = nirs_p01 - nirs_pad
        nirs_ylim_hi = nirs_p99 + nirs_pad

    ax1_torque = fig.add_subplot(3, 1, 1)
    ax1_torque.set_title(
        f"{RUN_ID} - NIRS sync overview (anchor={anchor_seq}) - OVERLAY "
        f"| auto={best_lag_s_auto:.3f}s | manual={manual_nudge_s_:.3f}s | total={total_lag_s:.3f}s"
    )

    ax1_torque.axvspan(anchor_start_time_s, anchor_end_time_s, facecolor=(0, 0, 0, 0.05), edgecolor="none")

    first_vc_label = True
    for span_start_idx, span_end_idx in vc_spans_in_anchor:
        span_start_time = master_time_ref_s[span_start_idx]
        span_end_time = master_time_ref_s[span_end_idx]
        ax1_torque.axvspan(
            span_start_time,
            span_end_time,
            facecolor=vc_face_rgba,
            edgecolor=vc_edge_rgba,
            linewidth=vc_edge_lw,
            label="VC" if first_vc_label else None,
        )
        first_vc_label = False

    ax1_torque.plot(torque_time_s, torque_values, linewidth=0.9, label=f"{TORQUE_COL} (compact)")
    ax1_torque.set_ylabel("torque")
    ax1_torque.set_xlabel("time_ref_s (s)")

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

    h1, l1 = ax1_torque.get_legend_handles_labels()
    h2, l2 = ax1_nirs.get_legend_handles_labels()
    ax1_torque.legend(h1 + h2, l1 + l2, loc="upper left")

    ax2_torque = fig.add_subplot(3, 1, 2)
    ax2_torque.set_title("BEFORE (unsynced) — Zoom on anchor")
    for span_start_idx, span_end_idx in vc_spans_in_anchor:
        span_start_time = master_time_ref_s[span_start_idx]
        span_end_time = master_time_ref_s[span_end_idx]
        ax2_torque.axvspan(span_start_time, span_end_time, facecolor=vc_face_rgba, edgecolor=vc_edge_rgba, linewidth=vc_edge_lw)

    torque_in_anchor_mask = (torque_time_indices >= anchor_start_index) & (torque_time_indices <= anchor_end_index)
    ax2_torque.plot(master_time_ref_s[torque_time_indices[torque_in_anchor_mask]], torque_values[torque_in_anchor_mask], linewidth=0.9)
    ax2_torque.set_xlim(anchor_start_time_s, anchor_end_time_s)
    ax2_torque.set_ylabel("torque")
    ax2_torque.set_xlabel("time_ref_s (s)")

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

    ax3_torque = fig.add_subplot(3, 1, 3)
    ax3_torque.set_title("AFTER (synced) - Zoom on anchor (MVC)")
    for span_start_idx, span_end_idx in vc_spans_in_anchor:
        span_start_time = master_time_ref_s[span_start_idx]
        span_end_time = master_time_ref_s[span_end_idx]
        ax3_torque.axvspan(span_start_time, span_end_time, facecolor=vc_face_rgba, edgecolor=vc_edge_rgba, linewidth=vc_edge_lw)

    ax3_torque.plot(master_time_ref_s[torque_time_indices[torque_in_anchor_mask]], torque_values[torque_in_anchor_mask], linewidth=0.9)
    ax3_torque.set_xlim(anchor_start_time_s, anchor_end_time_s)
    ax3_torque.set_ylabel("torque")
    ax3_torque.set_xlabel("time_ref_s (s)")

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

    plt.tight_layout()
    plt.show()

    ctx["nirs_compact_aligned_df"] = nirs_compact_aligned_df_out

    return nirs_compact_aligned_df_out, fig
