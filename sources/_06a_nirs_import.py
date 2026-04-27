# _06a_nirs_import.py - NIRS IMPORT
# Load PortaLite NIRS data from the Oxysoft TXT export and map to the master timeline.
#
# The PortaLite records four haemoglobin metrics (O2Hb, HHb, tHb, HbDiff) for each of
# three optode transmitters at a fixed nominal rate (FS_NIRS_HZ = 10 Hz). The Oxysoft
# software exports measurements as a tab-delimited TXT with a variable-length header
# preamble followed by a numeric table. There is no hardware timestamp per row: the
# file contains a sample counter starting at 0, and absolute timestamps are
# reconstructed by adding (sample / FS_NIRS_HZ) to the session start time found in the
# preamble.
#
# Data cleaning passes (in order)
#   1. Zero-row filter: the PortaLite emits all-zero rows before its optical sensors
#      produce valid data. Any row where every metric column is exactly zero is dropped.
#   2. Head trim: the first NIRS_TRIM_HEAD_S seconds of valid data are dropped because
#      the optical signal takes a short time to stabilise after device startup.
#      Empirically validated across all tested subjects.
#   3. In-range filter: NIRS rows that fall outside the master timeline are dropped.
#      This happens when the PortaLite starts recording before or stops after the Delsys
#      system.
#
# After cleaning, each NIRS row is assigned to the nearest master grid point via the
# same nearest-neighbour index matching used by BIA and Myoton in this pipeline.
#
# Inputs
#   ctx keys  : RUN_ID, CACHE_DIR, master_index_grid (with SEQ, VC, VC_count)
#   parameter : raw_nirs_dir — folder containing <RUN_ID>_NIRS.txt (Oxysoft export)
#               ts_ref       — session reference timestamp (Delsys clock anchor)
#               force_recompute — bypass cache and re-read the TXT file
#   files     : data/raw_signal/nirs/<RUN_ID>_NIRS.txt
#
# Outputs
#   return value : nirs_compact_df — haemoglobin signals for three transmitters,
#                  indexed on time_index with SEQ/VC/VC_count from the master grid
#
# Cache
#   - 06a_nirs_compact.parquet

import re
import numpy as np
import pandas as pd


# --- Constants ---

# Nominal NIRS sampling rate declared in the Oxysoft TXT header ("Export sample rate").
# If the header value differs from this constant, time reconstruction from the sample
# counter would be silently wrong — the import function raises a ValueError in that case.
FS_NIRS_HZ = 10.0

# Duration of the head trim applied after removing zero-signal startup rows.
# The PortaLite's optical sensors take ~0.5 s to settle after startup; trimming avoids
# these artefacts propagating into downstream signal processing.
NIRS_TRIM_HEAD_S = 0.5

# Column layout for the Oxysoft TXT numeric block.
# Every data row has exactly 13 required tokens (sample counter + 4 metrics × 3
# transmitters). An optional 14th token encodes an event label when present.
# e.g. row → "0  0.12  0.45  0.78  0.01  ..."  →  sample=0, 4 metrics per tx
NIRS_COL_NAMES = [
    "sample",
    "nirs_hbdiff_tx1", "nirs_thb_tx1", "nirs_hhb_tx1", "nirs_o2hb_tx1",
    "nirs_hbdiff_tx2", "nirs_thb_tx2", "nirs_hhb_tx2", "nirs_o2hb_tx2",
    "nirs_hbdiff_tx3", "nirs_thb_tx3", "nirs_hhb_tx3", "nirs_o2hb_tx3",
    "event",
]


def run_nirs_import(
    ctx,
    raw_nirs_dir,
    ts_ref,
    force_recompute,
):
    """
    Import a PortaLite Oxysoft TXT file and align to the master timeline.

    Timestamps are reconstructed from the sample counter column and the session start
    time found in the header preamble. Three cleaning passes remove startup artefacts,
    head-trim noise, and out-of-range rows before nearest-neighbour alignment.

    Returns nirs_compact_df with one row per NIRS sample, columns for all 12
    haemoglobin metrics (4 metrics × 3 transmitters), and SEQ/VC/VC_count labels
    propagated from the master grid via time_index.
    """

    # --- Resolve ctx ---
    run_id            = ctx["RUN_ID"]
    cache_dir         = ctx["CACHE_DIR"]
    master_index_grid = ctx["master_index_grid"]

    # Extract master grid arrays once; used for alignment and SEQ/VC label stamping.
    master_time_ref_s = master_index_grid["time_ref_s"].to_numpy(dtype=float)
    master_seq_index  = master_index_grid["SEQ_index"].to_numpy()
    master_seq        = master_index_grid["SEQ"].to_numpy()
    master_vc         = master_index_grid["VC"].to_numpy()
    master_vc_count   = master_index_grid["VC_count"].to_numpy()

    # --- Cache paths ---
    nirs_txt_path = raw_nirs_dir / f"{run_id}_NIRS.txt"
    cache_path    = cache_dir / "06a_nirs_compact.parquet"

    ctx.setdefault("parquet_path", {})
    ctx["parquet_path"]["CACHE_NIRS"] = cache_path

    # --- Cache check ---
    if cache_path.exists() and not force_recompute:
        print("[06a_nirs_import] Cache hit — loaded from cache. Set force_recompute=True to re-import.")
        return pd.read_parquet(cache_path)

    # --- Parse header ---
    # Scan only the first 50 lines — the preamble is short and reading further
    # before the numeric block wastes memory on large recording files.
    lines = nirs_txt_path.read_text(errors="replace").splitlines()

    start_dt  = None
    header_fs = None

    for line in lines[:50]:
        if line.startswith("Start of measurement:"):
            raw      = line.split("Start of measurement:")[1].strip().lstrip("\t")
            start_dt = pd.to_datetime(raw, errors="raise")

        if line.startswith("Export sample rate:") or line.startswith("Datafile sample rate:"):
            for p in re.split(r"\t+", line.strip()):
                try:
                    header_fs = float(p)
                    break
                except Exception:
                    pass

    # Validate declared sample rate before using it for time reconstruction.
    # A mismatched rate would silently produce wrong timestamps with no further error.
    if header_fs is None or abs(header_fs - FS_NIRS_HZ) > 1e-6:
        raise ValueError(
            f"[06a_nirs_import] Unexpected NIRS sample rate: {header_fs!r} "
            f"(expected {FS_NIRS_HZ} Hz). Check the Oxysoft export settings."
        )

    # --- Locate numeric block ---
    # The numeric block starts at the first line whose leading token is "0" (sample counter).
    first_numeric_idx = next(
        (i for i, ln in enumerate(lines) if ln.strip() and ln.strip().split()[0] == "0"),
        None,
    )

    # --- Parse numeric rows ---
    # Each row must have at least 13 tokens (sample + 12 metrics); rows with fewer tokens
    # are skipped. An optional 14th token is the event label when an event was recorded.
    core_rows    = []
    event_tokens = []

    for raw_line in lines[first_numeric_idx:]:
        stripped = raw_line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        if len(tokens) < 13:
            continue
        core_rows.append(tokens[:13])
        event_tokens.append(tokens[-1] if len(tokens) > 13 else np.nan)

    df = pd.DataFrame(core_rows, columns=NIRS_COL_NAMES[:13])
    df["event"] = pd.Series(event_tokens).replace({"": np.nan})
    df["sample"] = pd.to_numeric(df["sample"], errors="raise").astype(int)

    # metric_cols = the 12 haemoglobin columns between "sample" and "event"
    metric_cols = NIRS_COL_NAMES[1:13]
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Reconstruct timestamps ---
    # Absolute time from recording start = sample_counter / FS_NIRS_HZ.
    # e.g. sample=100 at 10 Hz → t = 10.0 s from the start of the recording.
    nirs_time_s = df["sample"].to_numpy(dtype=float) / FS_NIRS_HZ

    # --- Zero-row filter ---
    # Before the PortaLite sensors settle, every metric column reads exactly zero.
    # These startup rows carry no physiological information and are trimmed.
    signal_matrix = df[metric_cols].to_numpy(dtype=float)
    nonzero_mask  = np.any(np.abs(signal_matrix) > 0, axis=1)

    if not np.all(nonzero_mask):
        first_valid = int(np.argmax(nonzero_mask)) if np.any(nonzero_mask) else 0
        df          = df.iloc[first_valid:].copy()
        nirs_time_s = nirs_time_s[first_valid:]

    # --- Head trim ---
    # Drop the first NIRS_TRIM_HEAD_S seconds after the zero-row cutoff; the optical
    # signal shows transient noise even after the device emits non-zero values.
    keep_mask   = nirs_time_s >= NIRS_TRIM_HEAD_S
    df          = df.loc[keep_mask].copy()
    nirs_time_s = nirs_time_s[keep_mask]

    # --- Convert to session seconds ---
    # Wall-clock timestamp per row → seconds from the Delsys session anchor (ts_ref).
    sample_timestamp = start_dt + pd.to_timedelta(nirs_time_s, unit="s")
    time_on_master_s = (sample_timestamp - ts_ref).total_seconds().to_numpy(dtype=float)

    # --- In-range filter ---
    # Drop NIRS rows that fall before the first or after the last master grid point;
    # they cannot be assigned a valid time_index and would produce out-of-bounds indices.
    inrange_mask     = (
        (time_on_master_s >= master_time_ref_s[0]) &
        (time_on_master_s <= master_time_ref_s[-1])
    )
    df               = df.loc[inrange_mask].copy()
    time_on_master_s = time_on_master_s[inrange_mask]

    # --- Map to master ---
    # Nearest-neighbour matching: for each NIRS sample, find the two surrounding master
    # grid points and assign to whichever is closer. Ties go to the left (earlier) point.
    cand        = np.searchsorted(master_time_ref_s, time_on_master_s, side="left")
    cand        = np.clip(cand, 1, len(master_time_ref_s) - 1)
    left        = master_time_ref_s[cand - 1]
    right       = master_time_ref_s[cand]
    choose_left = (time_on_master_s - left) <= (right - time_on_master_s)
    mapped_idx  = np.where(choose_left, cand - 1, cand).astype(int)

    # Insert time_index at column 0 and the continuous time at column 1 so that the
    # join key is always the leftmost column in downstream inspection and merging.
    df.insert(0, "time_index",            mapped_idx)
    df.insert(1, "nirs_time_on_master_s", time_on_master_s)

    # Direct numpy array indexing is used here instead of master_index_grid.loc[idx, col]
    # because .loc with an integer array assumes a RangeIndex, which is fragile if the
    # master grid was ever filtered or reindexed upstream.
    df["SEQ_index"] = master_seq_index[mapped_idx]
    df["SEQ"]       = master_seq[mapped_idx]
    df["VC"]        = master_vc[mapped_idx]
    df["VC_count"]  = master_vc_count[mapped_idx]

    # --- Write cache ---
    df.to_parquet(cache_path, index=False)
    print(f"[06a_nirs_import] Done — {len(df)} rows written to {cache_path.name}.")
    return df
