# ============================================================
# 07_nirs_import — load Oxysoft NIRS txt + map to master time_index (sparse)
# - Uses absolute "Start of measurement" timestamp in header (critical)
# - Reconstruct per-sample timestamps using FS_NIRS_HZ
# - Convert to master-relative seconds via ts_ref
# - Deterministic nearest mapping to master time_ref_s (like BIA)
# - NO interpolation / NO fill / NO master padding
# - Drop out-of-master-range samples (explicit, reported)
# Cache: nirs_compact.parquet + nirs_import_report.parquet under CACHE_07_NIRS_IMPORT.parent
# ============================================================

from pathlib import Path
import re
import numpy as np
import pandas as pd

# ----------------------------
# REQUIRED INPUTS (fail loud)
# ----------------------------
assert "RUN_ID" in globals() and isinstance(RUN_ID, str) and RUN_ID, "RUN_ID missing/invalid"

assert "ts_ref" in globals() and isinstance(ts_ref, pd.Timestamp), "ts_ref missing/invalid (absolute session anchor Timestamp)"

assert "RAW_NIRS_DIR" in globals() and isinstance(RAW_NIRS_DIR, Path), "RAW_NIRS_DIR missing/invalid (expected RAW_ROOT / 'nirs')"
assert RAW_NIRS_DIR.exists(), f"RAW_NIRS_DIR does not exist: {RAW_NIRS_DIR}"

assert "CACHE_07_NIRS_IMPORT" in globals() and isinstance(CACHE_07_NIRS_IMPORT, Path), "CACHE_07_NIRS_IMPORT missing/invalid (Path base handle)"
CACHE_07_NIRS_IMPORT.parent.mkdir(parents=True, exist_ok=True)

assert "master_index_grid" in globals() and isinstance(master_index_grid, pd.DataFrame), "master_index_grid missing/invalid"
for c in ["time_ref_s", "SEQ_index", "VC", "VC_count"]:
    assert c in master_index_grid.columns, f"master_index_grid missing required column: {c}"

master_time_ref_s = master_index_grid["time_ref_s"].to_numpy(dtype=float)
assert np.all(np.diff(master_time_ref_s) > 0), "master_index_grid.time_ref_s must be strictly increasing"

# ----------------------------
# Dataset invariants (script-side constants)
# ----------------------------
FS_NIRS_HZ = 10.0
NIRS_TRIM_HEAD_S = 0.5  # remove first 0.5 s startup spike (Oxysoft export artifact)

# Fixed column layout (Oxysoft legend)
# Column 1 is sample number; columns 2..13 are Hb metrics; col 14 is Event (optional).
NIRS_COL_NAMES = [
    "sample",
    "nirs_hbdiff_tx1", "nirs_thb_tx1", "nirs_hhb_tx1", "nirs_o2hb_tx1",
    "nirs_hbdiff_tx2", "nirs_thb_tx2", "nirs_hhb_tx2", "nirs_o2hb_tx2",
    "nirs_hbdiff_tx3", "nirs_thb_tx3", "nirs_hhb_tx3", "nirs_o2hb_tx3",
    "event",
]

# ----------------------------
# Paths
# ----------------------------
nirs_txt_path = RAW_NIRS_DIR / f"{RUN_ID}_NIRS.txt"
assert nirs_txt_path.exists(), f"NIRS file not found: {nirs_txt_path}"

cache_compact_path = CACHE_07_NIRS_IMPORT.parent / "nirs_compact.parquet"
cache_report_path  = CACHE_07_NIRS_IMPORT.parent / "nirs_import_report.parquet"

# ----------------------------
# Cache hit
# ----------------------------
if cache_compact_path.exists() and cache_report_path.exists():
    nirs_compact_df_out = pd.read_parquet(cache_compact_path)
    nirs_import_report_df_out = pd.read_parquet(cache_report_path)
    print("[07_nirs_import] cache hit:", cache_compact_path.name)
else:
    # ----------------------------
    # Parse header (start time + sample rate + find data start)
    # ----------------------------
    header_lines = nirs_txt_path.read_text(errors="replace").splitlines()

    start_dt = None
    header_fs = None
    data_col_numbers_line = None  # line index where "1\t2\t3..." appears

    for i, line in enumerate(header_lines[:400]):  # header is always early
        if line.startswith("Start of measurement:"):
            # Example: "Start of measurement:\t2025-10-23 14:39:51.150"
            raw = line.split("Start of measurement:")[1].strip()
            raw = raw.lstrip("\t").strip()
            start_dt = pd.to_datetime(raw, errors="raise")
        if line.startswith("Export sample rate:") or line.startswith("Datafile sample rate:"):
            # Example: "Export sample rate:\t10.00\tHz"
            parts = re.split(r"\t+", line.strip())
            # parts like ["Export sample rate:", "10.00", "Hz"]
            for p in parts:
                try:
                    header_fs = float(p)
                    break
                except Exception:
                    pass
        # Column numbers line (right before numeric data)
        if re.match(r"^\s*1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12\t13\t14\s*$", line):
            data_col_numbers_line = i
            break

    assert start_dt is not None, "Could not parse 'Start of measurement' timestamp from NIRS header"
    assert header_fs is not None, "Could not parse sample rate from NIRS header"
    assert abs(header_fs - FS_NIRS_HZ) < 1e-6, f"NIRS header sample rate {header_fs} != expected FS_NIRS_HZ {FS_NIRS_HZ}"
    assert data_col_numbers_line is not None, "Could not find NIRS data column-number line ('1 2 3 ... 14')"

    # Data begins after that line; there may be blank lines
    data_start_line = data_col_numbers_line + 1
    while data_start_line < len(header_lines) and header_lines[data_start_line].strip() == "":
        data_start_line += 1

    # ----------------------------
    # Load numeric table
    # ----------------------------
    # We read from disk again using pandas for speed/stability.
    # Some rows may have missing 'event' field -> allow NaN.
    df = pd.read_csv(
        nirs_txt_path,
        sep=r"\t+",
        engine="python",
        header=None,
        names=NIRS_COL_NAMES,
        skiprows=data_start_line,
        dtype={ "sample": "Int64" },  # nullable int
    )

    # Drop fully empty rows (can occur at file end)
    df = df.dropna(subset=["sample"]).copy()
    df["sample"] = df["sample"].astype(int)

    # ----------------------------
    # Reconstruct NIRS time + apply trim
    # ----------------------------
    time_s = df["sample"].to_numpy(dtype=float) / FS_NIRS_HZ

    # Optional: drop leading rows that are all-zero across signals (startup artifact)
    sig_cols = [c for c in NIRS_COL_NAMES if c.startswith("nirs_")]
    sig_mat = df[sig_cols].to_numpy(dtype=float)
    nonzero_any = np.any(np.abs(sig_mat) > 0, axis=1)
    if not np.all(nonzero_any):
        first_valid = int(np.argmax(nonzero_any)) if np.any(nonzero_any) else 0
        df = df.iloc[first_valid:].copy()
        time_s = time_s[first_valid:]

    # Trim head (startup spike)
    keep_mask = time_s >= NIRS_TRIM_HEAD_S
    df = df.loc[keep_mask].copy()
    time_s = time_s[keep_mask]

    # Absolute per-sample timestamps from header start
    sample_ts = start_dt + pd.to_timedelta(time_s, unit="s")

    # Convert to master-relative seconds via ts_ref
    nirs_time_on_master_s = (sample_ts - ts_ref).total_seconds().to_numpy(dtype=float)

    # Monotonic check (fail loud)
    assert np.all(np.diff(nirs_time_on_master_s) > 0), "NIRS time_on_master_s must be strictly increasing"

    # ----------------------------
    # Deterministic nearest mapping to master grid (BIA-style)
    # ----------------------------
    # First: drop out-of-range (avoid edge-mapping tails to 0/last)
    low_mask = nirs_time_on_master_s < master_time_ref_s[0]
    high_mask = nirs_time_on_master_s > master_time_ref_s[-1]
    drop_low = int(np.sum(low_mask))
    drop_high = int(np.sum(high_mask))
    inrange_mask = ~(low_mask | high_mask)

    df_in = df.loc[inrange_mask].copy()
    nirs_time_on_master_s_in = nirs_time_on_master_s[inrange_mask]

    # Nearest time_index
    cand = np.searchsorted(master_time_ref_s, nirs_time_on_master_s_in, side="left")
    cand = np.clip(cand, 1, len(master_time_ref_s) - 1)
    left = master_time_ref_s[cand - 1]
    right = master_time_ref_s[cand]
    choose_left = (nirs_time_on_master_s_in - left) <= (right - nirs_time_on_master_s_in)
    time_index = np.where(choose_left, cand - 1, cand).astype(int)

    # Attach keys + labels from master at mapped indices
    df_in.insert(0, "time_index", time_index)
    df_in.insert(1, "nirs_time_on_master_s", nirs_time_on_master_s_in)

    df_in["SEQ_index"] = master_index_grid.loc[time_index, "SEQ_index"].to_numpy()
    df_in["VC"] = master_index_grid.loc[time_index, "VC"].to_numpy()
    df_in["VC_count"] = master_index_grid.loc[time_index, "VC_count"].to_numpy()

    # Contract checks
    assert df_in["time_index"].dtype.kind in ("i", "u"), "time_index must be integer"
    assert df_in["time_index"].min() >= 0 and df_in["time_index"].max() <= (len(master_index_grid) - 1), "time_index out of master bounds after in-range filter"

    # ----------------------------
    # Cache + report
    # ----------------------------
    nirs_compact_df_out = df_in

    nirs_import_report_df_out = pd.DataFrame([{
        "RUN_ID": RUN_ID,
        "nirs_txt": nirs_txt_path.name,
        "fs_nirs_hz": float(FS_NIRS_HZ),
        "trim_head_s": float(NIRS_TRIM_HEAD_S),
        "start_of_measurement": str(start_dt),
        "n_raw_rows": int(len(df)),
        "n_kept_rows": int(len(df_in)),
        "drop_low_import": drop_low,
        "drop_high_import": drop_high,
        "master_time_ref_s_min": float(master_time_ref_s[0]),
        "master_time_ref_s_max": float(master_time_ref_s[-1]),
        "nirs_time_on_master_s_min": float(np.min(nirs_time_on_master_s_in)) if len(nirs_time_on_master_s_in) else np.nan,
        "nirs_time_on_master_s_max": float(np.max(nirs_time_on_master_s_in)) if len(nirs_time_on_master_s_in) else np.nan,
    }])

    nirs_compact_df_out.to_parquet(cache_compact_path, index=False)
    nirs_import_report_df_out.to_parquet(cache_report_path, index=False)

    print("[07_nirs_import] wrote:")
    print("  -", cache_compact_path.name)
    print("  -", cache_report_path.name)

# Optional dict-style output (if you like)
nirs_tables_out = {
    "nirs_compact_df": nirs_compact_df_out,
    "nirs_import_report_df": nirs_import_report_df_out,
}