# _06a_nirs_import.py - NIRS IMPORT
# Load PortaLite NIRS data from the Oxysoft TXT export and map to the master timeline.
#
# Inputs
#   ctx keys:  RUN_ID, CACHE_DIR, master_index_grid, participants_df
#   files:     data/raw_signal/nirs/<RUN_ID>*.txt  (Oxysoft export)
#
# Outputs (ctx keys set)
#   - nirs_df  (O2Hb, HHb, tHb, HbDiff per transmitter, mapped to time_index)
#
# Cache
#   - 06a_nirs_compact.parquet
#
# Notes
#   - Timestamps are reconstructed from recording start time and nominal sampling rate (~10 Hz).
#   - SRS-derived TSI is not computed; HHb is the primary oxygenation signal used downstream.

from pathlib import Path
import re
import numpy as np
import pandas as pd


def run_nirs_import(
    ctx,
    raw_nirs_dir,
    ts_ref,
    force_recompute,
):
    """
    Import NIRS Oxysoft txt and map to master grid.

    Parameters
    ----------
    ctx : dict
        Must contain:
            - RUN_ID
            - CACHE_DIR
            - master_index_grid

    raw_nirs_dir : Path
    ts_ref : pd.Timestamp
    force_recompute : bool
    """

    # ----------------------------
    # Required ctx
    # ----------------------------

    for k in ["RUN_ID", "CACHE_DIR", "master_index_grid"]:
        assert k in ctx, f"{k} missing from ctx"

    run_id = ctx["RUN_ID"]
    cache_dir = ctx["CACHE_DIR"]
    master_index_grid = ctx["master_index_grid"]

    assert isinstance(raw_nirs_dir, Path)
    assert raw_nirs_dir.exists()
    assert isinstance(ts_ref, pd.Timestamp)

    for col in ["time_ref_s", "SEQ_index", "VC", "VC_count"]:
        assert col in master_index_grid.columns

    master_time_ref_s = master_index_grid["time_ref_s"].to_numpy(dtype=float)
    assert np.all(np.diff(master_time_ref_s) > 0)


    # Constants
    FS_NIRS_HZ = 10.0 
    NIRS_TRIM_HEAD_S = 0.5 #remove the first 0.5s of NIRS data to avoid initial artifacts. Magic number but works well in all tested segments. 

    NIRS_COL_NAMES = [
        "sample",
        "nirs_hbdiff_tx1", "nirs_thb_tx1", "nirs_hhb_tx1", "nirs_o2hb_tx1",
        "nirs_hbdiff_tx2", "nirs_thb_tx2", "nirs_hhb_tx2", "nirs_o2hb_tx2",
        "nirs_hbdiff_tx3", "nirs_thb_tx3", "nirs_hhb_tx3", "nirs_o2hb_tx3",
        "event",
    ]


    # Paths
    nirs_txt_path = raw_nirs_dir / f"{run_id}_NIRS.txt"
    assert nirs_txt_path.exists()

    cache_path = cache_dir / "06a_nirs_compact.parquet"

    if cache_path.exists() and not force_recompute:
        print(f"[06a_nirs_import] cache exists and has been loaded. \nSet force_recompute=True or delete cache to re-run import")
        return pd.read_parquet(cache_path)


    # Parse header
    lines = nirs_txt_path.read_text(errors="replace").splitlines()

    start_dt = None
    header_fs = None

    for line in lines[:50]:
        if line.startswith("Start of measurement:"):
            raw = line.split("Start of measurement:")[1].strip().lstrip("\t")
            start_dt = pd.to_datetime(raw, errors="raise")

        if line.startswith("Export sample rate:") or line.startswith("Datafile sample rate:"):
            parts = re.split(r"\t+", line.strip())
            for p in parts:
                try:
                    header_fs = float(p)
                    break
                except Exception:
                    pass

    assert start_dt is not None
    assert header_fs is not None
    assert abs(header_fs - FS_NIRS_HZ) < 1e-6

    # ----------------------------
    # Locate numeric block
    # ----------------------------

    first_numeric_idx = None
    for i, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if stripped and stripped.split()[0] == "0":
            first_numeric_idx = i
            break

    assert first_numeric_idx is not None

    core_token_count = 13
    core_rows = []
    event_tokens = []

    for raw_line in lines[first_numeric_idx:]:
        stripped = raw_line.strip()
        if not stripped:
            continue

        tokens = stripped.split()
        if len(tokens) < core_token_count:
            continue

        core_rows.append(tokens[:core_token_count])
        event_tokens.append(tokens[-1] if len(tokens) > core_token_count else np.nan)

    assert len(core_rows) > 0

    df = pd.DataFrame(core_rows, columns=NIRS_COL_NAMES[:13])
    df["event"] = pd.Series(event_tokens).replace({"": np.nan})

    df["sample"] = pd.to_numeric(df["sample"], errors="raise").astype(int)

    metric_cols = [
        "nirs_hbdiff_tx1", "nirs_thb_tx1", "nirs_hhb_tx1", "nirs_o2hb_tx1",
        "nirs_hbdiff_tx2", "nirs_thb_tx2", "nirs_hhb_tx2", "nirs_o2hb_tx2",
        "nirs_hbdiff_tx3", "nirs_thb_tx3", "nirs_hhb_tx3", "nirs_o2hb_tx3",
    ]

    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ----------------------------
    # Reconstruct time
    # ----------------------------

    nirs_time_s = df["sample"].to_numpy(dtype=float) / FS_NIRS_HZ

    signal_matrix = df[metric_cols].to_numpy(dtype=float)
    nonzero_mask = np.any(np.abs(signal_matrix) > 0, axis=1)

    if not np.all(nonzero_mask):
        first_valid = int(np.argmax(nonzero_mask)) if np.any(nonzero_mask) else 0
        df = df.iloc[first_valid:].copy()
        nirs_time_s = nirs_time_s[first_valid:]

    keep_mask = nirs_time_s >= NIRS_TRIM_HEAD_S
    df = df.loc[keep_mask].copy()
    nirs_time_s = nirs_time_s[keep_mask]

    sample_timestamp = start_dt + pd.to_timedelta(nirs_time_s, unit="s")
    time_on_master_s = (sample_timestamp - ts_ref).total_seconds().to_numpy(dtype=float)

    assert np.all(np.diff(time_on_master_s) > 0)

    # ----------------------------
    # In-range filter
    # ----------------------------

    inrange_mask = (
        (time_on_master_s >= master_time_ref_s[0]) &
        (time_on_master_s <= master_time_ref_s[-1])
    )

    df = df.loc[inrange_mask].copy()
    time_on_master_s = time_on_master_s[inrange_mask]

    # ----------------------------
    # Map to master
    # ----------------------------

    cand = np.searchsorted(master_time_ref_s, time_on_master_s, side="left")
    cand = np.clip(cand, 1, len(master_time_ref_s) - 1)

    left = master_time_ref_s[cand - 1]
    right = master_time_ref_s[cand]

    choose_left = (time_on_master_s - left) <= (right - time_on_master_s)
    mapped_idx = np.where(choose_left, cand - 1, cand).astype(int)

    df.insert(0, "time_index", mapped_idx)
    df.insert(1, "nirs_time_on_master_s", time_on_master_s)

    df["SEQ_index"] = master_index_grid.loc[mapped_idx, "SEQ_index"].to_numpy()
    df["SEQ"] = master_index_grid.loc[mapped_idx, "SEQ"].to_numpy()
    df["VC"] = master_index_grid.loc[mapped_idx, "VC"].to_numpy()
    df["VC_count"] = master_index_grid.loc[mapped_idx, "VC_count"].to_numpy()

    df.to_parquet(cache_path, index=False)
    print("[06a_nirs_import] Cache has been saved:", cache_path.name)

    return df
