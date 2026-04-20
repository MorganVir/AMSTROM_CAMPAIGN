# _07_myoton_import_sync.py - MYOTON IMPORT AND SYNC
# Load discrete Myoton tap data from CSV and map tap events to the master timeline.
#
# Inputs
#   ctx keys:  RUN_ID, CACHE_DIR, master_index_grid
#   files:     data/raw_signal/myoton/<RUN_ID>*.csv
#
# Outputs (ctx keys set)
#   - myoton_df                       (tap rows and per-test mean rows, mapped to time_index)
#   - ctx['parquet_path']['CACHE_MYOTON'] updated
#
# Cache
#   - 07_myoton_compact.parquet

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run_myoton_load_sync(
    ctx,
    raw_myoton_dir,
    ts_ref,
    force_recompute,
):
    """
    Import MYOTON csv, map taps to master time_index, compute per-test mean rows,
    optionally plot MYOTON metrics over torque background.

    Parameters
    ----------
    ctx : dict
        Must contain:
            - RUN_ID
            - CACHE_DIR
            - master_index_grid
            - torque_compact_df

    raw_myoton_dir : Path
    ts_ref : pd.Timestamp
    force_recompute : bool

    Returns
    -------
    myoton_compact_df : pd.DataFrame
    fig : matplotlib Figure or None
    """

    for k in ["RUN_ID", "CACHE_DIR", "master_index_grid"]:
        assert k in ctx, f"{k} missing from ctx"
    
    # constants 
    myo_taps_per_test = 5
    csv_sep = ";"
    plot_master_decim = 50
    tap_size = 18
    test_size = 60

    run_id = ctx["RUN_ID"]
    cache_dir = ctx["CACHE_DIR"]
    master_index_grid = ctx["master_index_grid"]

    assert isinstance(raw_myoton_dir, Path) and raw_myoton_dir.exists()
    assert isinstance(ts_ref, pd.Timestamp)
    assert int(myo_taps_per_test) > 0

    for col in ["time_index", "time_ref_s", "SEQ_index", "VC", "VC_count"]:
        assert col in master_index_grid.columns, f"master_index_grid missing column: {col}"

    CACHE_MYOTON = cache_dir / "07_myoton_compact.parquet"
    if CACHE_MYOTON.exists() and not force_recompute:
        print("[07_myoton_import] Cache exist and has been loaded. Set force_recompute=True or delete cache to re-import and sync.")
        # hotfix : still print the cache path so it can be registered in the dashboard for the final qc plot. Otherwise it fails on cache load.
        ctx.setdefault("parquet_path", {})
        ctx["parquet_path"]["CACHE_MYOTON"] = CACHE_MYOTON
        df = pd.read_parquet(CACHE_MYOTON)
        return df, None

    # Cache paths 

    # register specific cache paths (so dashboard commit can reuse them without hardcoding )
    ctx.setdefault("parquet_path", {})
    ctx["parquet_path"]["CACHE_MYOTON"] = CACHE_MYOTON






    myo_raw_path = raw_myoton_dir / f"{run_id}_Myoton.csv"
    assert myo_raw_path.exists(), f"Missing MYOTON file: {myo_raw_path}"

    myo_metric_base_names = ["Frequency", "Stiffness", "Decrement", "Relaxation", "Creep"]
    myo_metric_cols = [f"myo_{name}" for name in myo_metric_base_names]

    print(f"[07_myoton_import] loading: {myo_raw_path.name}")
    myo_raw_df = pd.read_csv(myo_raw_path, sep=csv_sep, engine="python")

    assert "Measurement time" in myo_raw_df.columns, "MYOTON csv missing 'Measurement time' column"

    myo_raw_df["myo_ts"] = pd.to_datetime(
        myo_raw_df["Measurement time"],
        format="mixed",
        dayfirst=True,
    )
    assert myo_raw_df["myo_ts"].notna().all(), "Myoton timestamp parsing failed (NaT present)"

    for base_name, prefixed_name in zip(myo_metric_base_names, myo_metric_cols):
        assert base_name in myo_raw_df.columns, f"MYOTON csv missing metric column: {base_name}"
        myo_raw_df[prefixed_name] = pd.to_numeric(myo_raw_df[base_name], errors="coerce")

    # One row per tap
    myo_taps_df = myo_raw_df[["myo_ts"] + myo_metric_cols].copy().reset_index(drop=True)

    # Tap/test structure
    myo_taps_df["myo_test_id"] = (myo_taps_df.index // int(myo_taps_per_test)) + 1
    myo_taps_df["myo_tap_in_test"] = myo_taps_df.groupby("myo_test_id").cumcount() + 1

    # Timestamp -> seconds from session anchor
    myo_taps_df["myo_time_s"] = (myo_taps_df["myo_ts"] - ts_ref).dt.total_seconds()


    # Nearest mapping to master 
    master_time_ref_s = master_index_grid["time_ref_s"].to_numpy(dtype=float)
    assert master_time_ref_s.size >= 2
    assert np.all(np.diff(master_time_ref_s) >= 0)

    myo_time_s = myo_taps_df["myo_time_s"].to_numpy(dtype=float)

    t_min = float(master_time_ref_s[0])
    t_max = float(master_time_ref_s[-1])

    in_range = (myo_time_s >= t_min) & (myo_time_s <= t_max)
    dropped = int((~in_range).sum())

    if dropped > 0:
        myo_taps_df = myo_taps_df.loc[in_range].copy().reset_index(drop=True)
        myo_time_s = myo_time_s[in_range]

    idx_right = np.searchsorted(master_time_ref_s, myo_time_s, side="left")
    n = int(master_time_ref_s.size)

    assert (idx_right >= 0).all() and (idx_right < n).all()

    idx_left = np.maximum(idx_right - 1, 0)

    is_first = idx_right == 0
    choose_right = np.zeros_like(idx_right, dtype=bool)

    not_first = ~is_first
    dr = np.abs(master_time_ref_s[idx_right[not_first]] - myo_time_s[not_first])
    dl = np.abs(myo_time_s[not_first] - master_time_ref_s[idx_left[not_first]])
    choose_right[not_first] = dr < dl

    myo_time_index = np.where(choose_right, idx_right, idx_left).astype(int)
    assert (myo_time_index >= 0).all() and (myo_time_index < n).all()

    # Attach shared keys
    master_seq_np = master_index_grid["SEQ"].to_numpy(dtype=object)
    master_seq_index_np = master_index_grid["SEQ_index"].to_numpy(dtype=int)
    master_vc_np = master_index_grid["VC"].to_numpy(dtype=int)
    master_vc_count_np = master_index_grid["VC_count"].to_numpy(dtype=int)

    myo_taps_df["time_index"] = myo_time_index
    myo_taps_df["SEQ_index"] = master_seq_index_np[myo_time_index]
    myo_taps_df["SEQ"] = master_seq_np[myo_time_index]
    myo_taps_df["VC"] = master_vc_np[myo_time_index]
    myo_taps_df["VC_count"] = master_vc_count_np[myo_time_index]


    # Per-test means (1 row per test)
    agg_dict = {
        "myo_ts": ("myo_ts", "median"),
        "myo_time_s": ("myo_time_s", "median"),
        "time_index": ("time_index", "median"),
        "SEQ_index": ("SEQ_index", "median"),
        "SEQ": ("SEQ", "first"),
        "VC": ("VC", "median"),
        "VC_count": ("VC_count", "median"),
    }
    for metric_col in myo_metric_cols:
        agg_dict[metric_col] = (metric_col, "mean")

    myo_tests_df = (
        myo_taps_df
        .groupby("myo_test_id", as_index=False)
        .agg(**agg_dict)
    )
    myo_tests_df["time_index"] = myo_tests_df["time_index"].round().astype(int)


    # Pack compact (taps + tests)
    taps_keep = [
        "time_index", "SEQ_index", "SEQ", "VC", "VC_count",
        "myo_test_id", "myo_tap_in_test", "myo_time_s",
    ] + myo_metric_cols

    tests_keep = [
        "time_index", "SEQ_index", "SEQ", "VC", "VC_count",
        "myo_test_id", "myo_time_s",
    ] + myo_metric_cols

    taps_out = myo_taps_df[taps_keep].copy()
    taps_out.insert(0, "row_type", "tap")

    tests_out = myo_tests_df[tests_keep].copy()
    tests_out["myo_tap_in_test"] = np.nan
    tests_out.insert(0, "row_type", "test")

    col_order = ["row_type", "time_index", "SEQ_index", "SEQ", "VC", "VC_count",
                 "myo_test_id", "myo_tap_in_test", "myo_time_s"] + myo_metric_cols

    myoton_compact_df = pd.concat([taps_out, tests_out], ignore_index=True)[col_order]

    # Cache write
    myoton_compact_df.to_parquet(CACHE_MYOTON, index=False)
    print(f"[07_myoton_import] Cache has been saved : {CACHE_MYOTON.name}. Force_recompute=True or deleting the file will trigger re-import and sync.")


    # QC plot
    fig = None

    assert "torque_compact_df" in ctx, "plot=True requires ctx['torque_compact_df']"
    torque_compact_df = ctx["torque_compact_df"]
    assert "time_index" in torque_compact_df.columns and "torque_raw" in torque_compact_df.columns

    plt.close("all")

    metrics = myo_metric_cols
    n_panels = len(metrics)

    taps_df = myoton_compact_df[myoton_compact_df["row_type"] == "tap"]
    tests_df = myoton_compact_df[myoton_compact_df["row_type"] == "test"]

    torque_time_index = torque_compact_df["time_index"].to_numpy(dtype=int)
    torque_background = torque_compact_df["torque_raw"].to_numpy(dtype=float)

    master_time_index = master_index_grid["time_index"].to_numpy(dtype=int)
    master_seq_index = master_index_grid["SEQ_index"].to_numpy(dtype=int)
    master_vc_mask = master_index_grid["VC_count"].to_numpy(dtype=int) > 0

    seq_change_idx = np.where(np.diff(master_seq_index) != 0)[0] + 1
    seq_bounds = np.r_[0, seq_change_idx, len(master_seq_index)]

    seq_boxes = []
    for seq_start, seq_end in zip(seq_bounds[:-1], seq_bounds[1:]):
        vc_local = master_vc_mask[seq_start:seq_end]
        if not np.any(vc_local):
            continue
        vc_first = seq_start + np.where(vc_local)[0][0]
        vc_last = seq_start + np.where(vc_local)[0][-1]
        seq_boxes.append((master_time_index[vc_first], master_time_index[vc_last]))

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(13, 3.2 * n_panels),
        sharex=True,
    )
    if n_panels == 1:
        axes = [axes]

    legend_axis_right = None

    dec = max(1, int(plot_master_decim))

    for ax_left, metric_col in zip(axes, metrics):
        ax_left.plot(
            torque_time_index[::dec],
            torque_background[::dec],
            color="0.85",
            linewidth=1,
            zorder=1,
        )

        for x_start, x_end in seq_boxes:
            ax_left.axvspan(
                x_start, x_end,
                facecolor="orange",
                alpha=0.06,
                edgecolor="none",
                linewidth=0,
                zorder=5,
            )

        ax_left.set_ylabel("Torque", color="0.5")
        ax_left.tick_params(axis="y", labelcolor="0.5")

        ax_right = ax_left.twinx()
        if legend_axis_right is None:
            legend_axis_right = ax_right

        ax_right.scatter(
            taps_df["time_index"].to_numpy(dtype=int),
            taps_df[metric_col].to_numpy(dtype=float),
            s=int(tap_size),
            alpha=0.8,
            label="taps",
            zorder=20,
        )
        ax_right.scatter(
            tests_df["time_index"].to_numpy(dtype=int),
            tests_df[metric_col].to_numpy(dtype=float),
            s=int(test_size),
            marker="D",
            label="test mean",
            zorder=21,
        )
        ax_right.plot(
            tests_df["time_index"].to_numpy(dtype=int),
            tests_df[metric_col].to_numpy(dtype=float),
            linewidth=1,
            alpha=0.35,
            zorder=19,
        )

        ax_right.set_ylabel(metric_col)
        ax_right.grid(alpha=0.25)

    axes[-1].set_xlabel("time_index (master grid)")
    if legend_axis_right is not None:
        legend_axis_right.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    return myoton_compact_df, fig
