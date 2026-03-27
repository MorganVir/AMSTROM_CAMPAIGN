from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#parameters

lag_step_s = 0.05 # seconds. Finer step = more compute, but smoother scores and less risk of missing the true optimum.
anchor_seq = "MVC_REF" # experimental remnant. don't swap to something else.
target_hz = 9760 #frequency bin used for the sync (must be present in bia4 freqs)


def run_bia_sync(
    *,
    ctx: dict,
    lag_min_s: float,
    lag_max_s: float,
    manual_nudge_s: float = 0.0,
    torque_col: str = "torque_raw",
    force_recompute: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Any, Dict[str, Any]]:
    """
    05b — BIA sync
    - score lag using bia4 |Z| within anchor_seq using VC_count separation
    - apply same lag to bia2 and bia4
    - drop out-of-range
    - relabel from master

    Requires ctx keys:
      RUN_ID, CACHE_DIR, master_index_grid, bia2_compact_df, bia4_compact_df, bia4_freqs_hz
    Optional (required if plotting w/o cache):
      torque_compact_df
    """


    # asserts
    required = ["RUN_ID", "CACHE_DIR", "master_index_grid", "bia2_compact_df", "bia4_compact_df", "bia4_freqs_hz"]
    for k in required:
        if k not in ctx:
            raise KeyError(f"[05b_bia_sync] missing ctx['{k}']")
        if ctx[k] is None:
            raise ValueError(f"[05b_bia_sync] ctx['{k}'] is None")

    RUN_ID: str = ctx["RUN_ID"]
    cache_dir: Path = ctx["CACHE_DIR"]
    if not isinstance(cache_dir, Path):
        raise TypeError(f"[05b_bia_sync] ctx['CACHE_DIR'] must be pathlib.Path, got {type(cache_dir)}")

    master_index_grid: pd.DataFrame = ctx["master_index_grid"]
    bia2_compact_df: pd.DataFrame = ctx["bia2_compact_df"]
    bia4_compact_df: pd.DataFrame = ctx["bia4_compact_df"]
    bia4_freqs_hz = np.asarray(ctx["bia4_freqs_hz"], dtype=float)

    if ("torque_compact_df" not in ctx or ctx["torque_compact_df"] is None):
        raise KeyError("[05b_bia_sync] Requires ctx['torque_compact_df' for plotting the torque backdrop.")

    torque_compact_df = ctx.get("torque_compact_df", None)


    # Cache paths 
    cache_dir.mkdir(parents=True, exist_ok=True)
    CACHE_BIA2_ALIGNED = cache_dir / "05b_bia2_compact_aligned.parquet"
    CACHE_BIA4_ALIGNED = cache_dir / "05b_bia4_compact_aligned.parquet"

    # register specific cache paths (so dashboard commit can reuse them without hardcoding )
    ctx.setdefault("parquet_path", {})
    ctx["parquet_path"]["CACHE_BIA_2_ALIGNED"] = CACHE_BIA2_ALIGNED
    ctx["parquet_path"]["CACHE_BIA_4_ALIGNED"] = CACHE_BIA4_ALIGNED

    # Cache hit
    if (CACHE_BIA2_ALIGNED.exists() and CACHE_BIA4_ALIGNED.exists()) and (not force_recompute):
        bia2_compact_aligned_df_out = pd.read_parquet(CACHE_BIA2_ALIGNED)
        bia4_compact_aligned_df_out = pd.read_parquet(CACHE_BIA4_ALIGNED)

        print(f"[05b_bia_sync] Cache exists. Loaded aligned BIA2/BIA4 from : \n{CACHE_BIA2_ALIGNED} \n{CACHE_BIA4_ALIGNED}.\nSet force_recompute=True or delete cache to re-run sync (dont forget to commit afterwards!)")

        fig = None
        ctx["bia2_compact_aligned_df"] = bia2_compact_aligned_df_out
        ctx["bia4_compact_aligned_df"] = bia4_compact_aligned_df_out
        return bia2_compact_aligned_df_out, bia4_compact_aligned_df_out, fig


    # Master grid 
    for col in ["time_index", "time_ref_s", "SEQ", "SEQ_index", "VC", "VC_count"]:
        if col not in master_index_grid.columns:
            raise AssertionError(f"master_index_grid missing required column: {col}")

    master_time_ref_s = master_index_grid["time_ref_s"].to_numpy(dtype=float)
    if master_time_ref_s.size < 3:
        raise ValueError("master_index_grid too short")

    master_time_deltas = np.diff(master_time_ref_s)
    if not np.all(np.isfinite(master_time_deltas)):
        raise ValueError("master time_ref_s contains non-finite deltas")
    median_dt = float(np.median(master_time_deltas))
    if median_dt <= 0:
        raise ValueError("master time_ref_s not strictly increasing (median dt <= 0)")

    master_fs_hz = 1.0 / median_dt
    master_n = int(len(master_index_grid))

    master_seq = master_index_grid["SEQ"].to_numpy()
    master_seq_index = master_index_grid["SEQ_index"].to_numpy(dtype=int)
    master_vc = master_index_grid["VC"].to_numpy(dtype=int)
    master_vc_count = master_index_grid["VC_count"].to_numpy(dtype=int)


    # Anchor window mask
    anchor_seq_name = str(anchor_seq)
    mvc_mask_master = (master_seq == anchor_seq_name)
    if int(np.sum(mvc_mask_master)) == 0:
        raise ValueError(f"Anchor SEQ '{anchor_seq_name}' not found in master_index_grid['SEQ'].")


    # Frequency must exist exactly in bia4 freqs 
    freq_match_idx = np.where(bia4_freqs_hz == target_hz)[0]
    if freq_match_idx.size == 0:
        raise ValueError(f"BIA target_hz={target_hz} not in bia4_freqs_hz bins: {bia4_freqs_hz.tolist()}")
    freq_used_hz = float(target_hz)
    freq_tag = str(int(freq_used_hz))  # your freqs are integer-valued

    bia4_R_col = f"bia4_R_ohm__f_{freq_tag}Hz"
    bia4_Xc_col = f"bia4_Xc_ohm__f_{freq_tag}Hz"
    bia2_R_col = f"bia2_R_ohm__f_{freq_tag}Hz"
    bia2_Xc_col = f"bia2_Xc_ohm__f_{freq_tag}Hz"

    for df_name, df, cols in [
        ("bia4_compact_df", bia4_compact_df, [bia4_R_col, bia4_Xc_col, "time_index"]),
        ("bia2_compact_df", bia2_compact_df, [bia2_R_col, bia2_Xc_col, "time_index"]),
    ]:
        for col in cols:
            if col not in df.columns:
                raise AssertionError(f"{df_name}: missing column {col}")


    # Build |Z| from real samples only (bia4 drives scoring)
    bia4_time_index_raw = bia4_compact_df["time_index"].to_numpy(dtype=int)
    bia4_R = bia4_compact_df[bia4_R_col].to_numpy(dtype=float)
    bia4_Xc = bia4_compact_df[bia4_Xc_col].to_numpy(dtype=float)
    bia4_absZ = np.sqrt(bia4_R * bia4_R + bia4_Xc * bia4_Xc)

    if bia4_time_index_raw.size == 0:
        raise ValueError("bia4_compact_df has zero rows.")
    if int(np.min(bia4_time_index_raw)) < 0 or int(np.max(bia4_time_index_raw)) >= master_n:
        raise ValueError("BIA time_index out of master range before sync (mapping broken).")


    # Lag candidate grid (samples)
    lag_candidates_s = np.arange(float(lag_min_s), float(lag_max_s) + 1e-12, float(lag_step_s))
    if lag_candidates_s.size < 3:
        raise ValueError("Lag grid too small. Check min/max/step.")
    shift_candidates_samples = np.rint(lag_candidates_s * master_fs_hz).astype(int)

   
    # Score lag using ONLY shifted BIA samples inside anchor_seq
    # score = abs(mean(|Z| in VC) - mean(|Z| out VC))
    scores = np.full(lag_candidates_s.shape, np.nan, dtype=float)

    for lag_idx, shift_samples in enumerate(shift_candidates_samples):
        bia4_time_index_shifted = bia4_time_index_raw + int(shift_samples)

        valid_shifted = (bia4_time_index_shifted >= 0) & (bia4_time_index_shifted < master_n)
        if int(np.sum(valid_shifted)) < 10:
            continue

        shifted_indices_valid = bia4_time_index_shifted[valid_shifted].astype(int, copy=False)
        absZ_valid = bia4_absZ[valid_shifted]

        in_anchor = mvc_mask_master[shifted_indices_valid]
        if int(np.sum(in_anchor)) < 10:
            continue

        anchor_indices = shifted_indices_valid[in_anchor]
        anchor_absZ = absZ_valid[in_anchor]

        anchor_vc_count = master_vc_count[anchor_indices]
        in_vc = (anchor_vc_count > 0)
        out_vc = ~in_vc

        if int(np.sum(in_vc)) < 3 or int(np.sum(out_vc)) < 3:
            continue

        mean_in_vc = float(np.nanmean(anchor_absZ[in_vc]))
        mean_out_vc = float(np.nanmean(anchor_absZ[out_vc]))
        scores[lag_idx] = abs(mean_in_vc - mean_out_vc)

    if not np.any(np.isfinite(scores)):
        raise ValueError("All lag scores are NaN (check anchor labeling, VC detection, lag window, BIA mapping).")

    best_score_idx = int(np.nanargmax(scores))
    best_lag_s_auto = float(lag_candidates_s[best_score_idx])
    best_shift_samples_auto = int(shift_candidates_samples[best_score_idx])

    manual_nudge_s = float(manual_nudge_s)
    total_lag_s = best_lag_s_auto + manual_nudge_s
    total_shift_samples = int(np.rint(total_lag_s * master_fs_hz))


    # Apply shift to bia2/bia4, DROP out-of-range, relabel from master

    def _shift_drop_and_relabel(compact_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        time_index_raw = compact_df["time_index"].to_numpy(dtype=int)
        time_index_shifted = time_index_raw + int(total_shift_samples)

        valid = (time_index_shifted >= 0) & (time_index_shifted < master_n)

        n_total = int(time_index_shifted.size)
        n_valid = int(np.sum(valid))
        n_drop_low = int(np.sum(time_index_shifted < 0))
        n_drop_high = int(np.sum(time_index_shifted >= master_n))

        if n_valid == 0:
            raise ValueError("Shift produced zero valid samples (check sign, lag window, manual nudge).")

        df_out = compact_df.loc[valid].copy().reset_index(drop=True)
        time_index_valid = time_index_shifted[valid].astype(int, copy=False)

        df_out["time_index"] = time_index_valid
        df_out["SEQ_index"] = master_seq_index[time_index_valid]
        df_out["SEQ"] = master_seq[time_index_valid]
        df_out["VC"] = master_vc[time_index_valid]
        df_out["VC_count"] = master_vc_count[time_index_valid]

        df_out = df_out.sort_values("time_index", kind="mergesort").reset_index(drop=True)

        stats = {
            "n_total": n_total,
            "n_valid": n_valid,
            "n_drop_low": n_drop_low,
            "n_drop_high": n_drop_high,
        }
        return df_out, stats

    bia2_compact_aligned_df_out, bia2_stats = _shift_drop_and_relabel(bia2_compact_df)
    bia4_compact_aligned_df_out, bia4_stats = _shift_drop_and_relabel(bia4_compact_df)


    # plot
    fig = None
    plt.close("all")

    torque_time_index = torque_compact_df["time_index"].to_numpy(dtype=int)
    torque_values = torque_compact_df[str(torque_col)].to_numpy(dtype=float)

    mvc_mask = (master_index_grid["SEQ"].to_numpy() == anchor_seq_name)
    mvc_idx = np.where(mvc_mask)[0]
    if mvc_idx.size == 0:
        raise ValueError(f"{anchor_seq_name} not found in master_index_grid['SEQ']")
    mvc_i0 = int(mvc_idx[0])
    mvc_i1 = int(mvc_idx[-1])

    vc_mask = (master_index_grid["VC_count"].to_numpy(dtype=int) > 0)
    mask_vc_anchor = mvc_mask & vc_mask
    d = np.diff(mask_vc_anchor.astype(int))
    starts = np.where(d == 1)[0] + 1
    stops = np.where(d == -1)[0]
    if mask_vc_anchor[0]:
        starts = np.r_[0, starts]
    if mask_vc_anchor[-1]:
        stops = np.r_[stops, len(mask_vc_anchor) - 1]

    torque_in_anchor = (torque_time_index >= mvc_i0) & (torque_time_index <= mvc_i1)

    # |Z| before/after from bia4 at freq_used_hz
    bia4_raw_time_index = bia4_compact_df["time_index"].to_numpy(dtype=int)
    bia4_raw_absZ = np.sqrt(
        bia4_compact_df[bia4_R_col].to_numpy(dtype=float) ** 2
        + bia4_compact_df[bia4_Xc_col].to_numpy(dtype=float) ** 2
    )

    raw_in_anchor = (bia4_raw_time_index >= mvc_i0) & (bia4_raw_time_index <= mvc_i1)
    x_raw = bia4_raw_time_index[raw_in_anchor]
    y_raw = bia4_raw_absZ[raw_in_anchor]
    o = np.argsort(x_raw)
    x_raw, y_raw = x_raw[o], y_raw[o]

    bia4_aln_time_index = bia4_compact_aligned_df_out["time_index"].to_numpy(dtype=int)
    bia4_aln_absZ = np.sqrt(
        bia4_compact_aligned_df_out[bia4_R_col].to_numpy(dtype=float) ** 2
        + bia4_compact_aligned_df_out[bia4_Xc_col].to_numpy(dtype=float) ** 2
    )
    aln_in_anchor = (bia4_aln_time_index >= mvc_i0) & (bia4_aln_time_index <= mvc_i1)
    x_aln = bia4_aln_time_index[aln_in_anchor]
    y_aln = bia4_aln_absZ[aln_in_anchor]
    o = np.argsort(x_aln)
    x_aln, y_aln = x_aln[o], y_aln[o]

    lag_txt = f" (lag {total_lag_s:+0.2f}s | auto {best_lag_s_auto:+0.2f}s + nudge {manual_nudge_s:+0.2f}s)"

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex="col")

    ax = axes[0, 0]
    ax.plot(torque_time_index[torque_in_anchor], torque_values[torque_in_anchor], alpha=0.85)
    for s, e in zip(starts, stops):
        if s >= mvc_i0 and e <= mvc_i1:
            ax.axvline(s, color="orange", linewidth=2)
            ax.axvline(e, color="orange", linewidth=2)
    ax.set_title(f"BEFORE (raw) - {anchor_seq_name} torque + VC borders")
    ax.set_ylabel("Torque")

    ax = axes[1, 0]
    ax.plot(x_raw, y_raw, linewidth=1.2, alpha=0.85)
    ax.scatter(x_raw, y_raw, s=14, alpha=0.85)
    for s, e in zip(starts, stops):
        if s >= mvc_i0 and e <= mvc_i1:
            ax.axvline(s, color="orange", linewidth=2)
            ax.axvline(e, color="orange", linewidth=2)
    ax.set_title(f"|Z| bia4 @ {freq_tag} Hz - BEFORE (raw)")
    ax.set_ylabel("|Z| (ohm)")
    ax.set_xlabel("Master time_index")

    ax = axes[0, 1]
    ax.plot(torque_time_index[torque_in_anchor], torque_values[torque_in_anchor], alpha=0.85)
    for s, e in zip(starts, stops):
        if s >= mvc_i0 and e <= mvc_i1:
            ax.axvline(s, color="orange", linewidth=2)
            ax.axvline(e, color="orange", linewidth=2)
    ax.set_title(f"AFTER (aligned) - {anchor_seq_name} torque + VC borders (should not move)")
    ax.set_ylabel("Torque")

    ax = axes[1, 1]
    ax.plot(x_aln, y_aln, linewidth=1.2, alpha=0.85)
    ax.scatter(x_aln, y_aln, s=14, alpha=0.85)
    for s, e in zip(starts, stops):
        if s >= mvc_i0 and e <= mvc_i1:
            ax.axvline(s, color="orange", linewidth=2)
            ax.axvline(e, color="orange", linewidth=2)
    ax.set_title(f"|Z| bia4 @ {freq_tag} Hz - AFTER{lag_txt}")
    ax.set_ylabel("|Z| (ohm)")
    ax.set_xlabel("Master time_index")

    plt.tight_layout()
    plt.show()

  
    # ctx updates

    ctx["bia2_compact_aligned_df"] = bia2_compact_aligned_df_out
    ctx["bia4_compact_aligned_df"] = bia4_compact_aligned_df_out

    return bia2_compact_aligned_df_out, bia4_compact_aligned_df_out, fig
