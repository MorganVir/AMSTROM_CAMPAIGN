# ============================================================
# 07_nirs_import — load Oxysoft NIRS txt + map to master time_index 
# - Uses absolute "Start of measurement" timestamp in header (critical)
# - Reconstruct per-sample timestamps using FS_NIRS_HZ
# - Convert to master-relative seconds via ts_ref
# - Deterministic nearest mapping to master time_ref_s (like BIA)
# - NO interpolation / NO fill / NO master padding
# - Drop out-of-master-range samples (explicit, reported)
# Cache: 07_nirs_compact.parquet + 07_nirs_import_report.parquet under CACHE_07_NIRS_IMPORT.parent
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
for required_master_col in ["time_ref_s", "SEQ_index", "VC", "VC_count"]:
    assert required_master_col in master_index_grid.columns, f"master_index_grid missing required column: {required_master_col}"

master_time_ref_s = master_index_grid["time_ref_s"].to_numpy(dtype=float)
assert np.all(np.diff(master_time_ref_s) > 0), "master_index_grid.time_ref_s must be strictly increasing"

# ----------------------------
# Dataset invariants (script-side constants)
# ----------------------------
FS_NIRS_HZ = 10.0
NIRS_TRIM_HEAD_S = 0.5  # remove first 0.5 s startup spike (Oxysoft export artifact)

# Compact output schema (downstream contract)
# First 13 tokens = sample + 12 Hb metrics, Event = optional last token
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

cache_compact_path = CACHE_07_NIRS_IMPORT.parent / "07_nirs_compact.parquet"
cache_report_path = CACHE_07_NIRS_IMPORT.parent / "07_nirs_import_report.parquet"

# ----------------------------
# Cache hit
# ----------------------------
if cache_compact_path.exists() and cache_report_path.exists():
    nirs_compact_df_out = pd.read_parquet(cache_compact_path)
    nirs_import_report_df_out = pd.read_parquet(cache_report_path)
    print("[07_nirs_import] cache hit:", cache_compact_path.name)

else:
    # ----------------------------
    # Parse header (start time + sample rate)
    # ----------------------------
    header_lines = nirs_txt_path.read_text(errors="replace").splitlines()

    start_dt = None
    header_fs = None

    for header_line_index, header_line in enumerate(header_lines[:400]):  # header is always early
        if header_line.startswith("Start of measurement:"):
            raw_start = header_line.split("Start of measurement:")[1].strip()
            raw_start = raw_start.lstrip("\t").strip()
            start_dt = pd.to_datetime(raw_start, errors="raise")

        if header_line.startswith("Export sample rate:") or header_line.startswith("Datafile sample rate:"):
            header_parts = re.split(r"\t+", header_line.strip())
            for part in header_parts:
                try:
                    header_fs = float(part)
                    break
                except Exception:
                    pass

    assert start_dt is not None, "Could not parse 'Start of measurement' timestamp from NIRS header"
    assert header_fs is not None, "Could not parse sample rate from NIRS header"
    assert abs(header_fs - FS_NIRS_HZ) < 1e-6, f"NIRS header sample rate {header_fs} != expected FS_NIRS_HZ {FS_NIRS_HZ}"

    # ----------------------------
    # Locate numeric block first line whose first token is "0"
    # ----------------------------
    first_numeric_data_line_index = None
    for line_index, raw_line in enumerate(header_lines):
        stripped_line = raw_line.strip()
        if stripped_line == "":
            continue

        token_list = stripped_line.split()
        if len(token_list) == 0:
            continue

        if token_list[0] == "0":
            first_numeric_data_line_index = line_index
            break

    assert first_numeric_data_line_index is not None, "Could not find first numeric NIRS data line starting with sample '0'"

    # ----------------------------
    # Parse numeric rows: HARD CUT first 13 tokens (sample + 12 Hb metrics)
    # Event = optional last token if extra tokens exist; else NaN
    # ----------------------------
    core_token_count = 13
    core_row_token_list = []
    event_token_list = []

    short_line_count = 0

    for raw_line in header_lines[first_numeric_data_line_index:]:
        stripped_line = raw_line.strip()
        if stripped_line == "":
            continue

        token_list = stripped_line.split()
        if len(token_list) < core_token_count:
            short_line_count += 1
            continue

        core_row_token_list.append(token_list[:core_token_count])

        has_extra_tokens_beyond_core = (len(token_list) > core_token_count)
        event_token_value = token_list[-1] if has_extra_tokens_beyond_core else np.nan
        event_token_list.append(event_token_value)

    assert len(core_row_token_list) > 0, "Parsed zero NIRS numeric rows (unexpected)"

    nirs_compact_raw_df = pd.DataFrame(
        core_row_token_list,
        columns=NIRS_COL_NAMES[:13],
    )
    nirs_compact_raw_df["event"] = pd.Series(event_token_list, index=nirs_compact_raw_df.index).replace({"": np.nan})

    # Type conversions (fail loud on sample; allow NaN for metrics)
    nirs_compact_raw_df["sample"] = pd.to_numeric(nirs_compact_raw_df["sample"], errors="raise").astype(int)

    nirs_metric_column_list = [
        "nirs_hbdiff_tx1", "nirs_thb_tx1", "nirs_hhb_tx1", "nirs_o2hb_tx1",
        "nirs_hbdiff_tx2", "nirs_thb_tx2", "nirs_hhb_tx2", "nirs_o2hb_tx2",
        "nirs_hbdiff_tx3", "nirs_thb_tx3", "nirs_hhb_tx3", "nirs_o2hb_tx3",
    ]
    for metric_col in nirs_metric_column_list:
        nirs_compact_raw_df[metric_col] = pd.to_numeric(nirs_compact_raw_df[metric_col], errors="coerce")

    assert list(nirs_compact_raw_df.columns) == NIRS_COL_NAMES, f"NIRS compact schema mismatch: {nirs_compact_raw_df.columns.tolist()}"

    # ----------------------------
    # Reconstruct NIRS time + apply trim
    # ----------------------------
    nirs_time_s = nirs_compact_raw_df["sample"].to_numpy(dtype=float) / float(FS_NIRS_HZ)

    # Drop leading all-zero rows across Hb metrics (startup artifact)
    nirs_signal_matrix = nirs_compact_raw_df[nirs_metric_column_list].to_numpy(dtype=float)
    nirs_nonzero_any_mask = np.any(np.abs(nirs_signal_matrix) > 0, axis=1)

    if not np.all(nirs_nonzero_any_mask):
        first_valid_row_index = int(np.argmax(nirs_nonzero_any_mask)) if np.any(nirs_nonzero_any_mask) else 0
        nirs_compact_raw_df = nirs_compact_raw_df.iloc[first_valid_row_index:].copy()
        nirs_time_s = nirs_time_s[first_valid_row_index:]

    # Trim head (startup spike)
    keep_mask = nirs_time_s >= float(NIRS_TRIM_HEAD_S)
    nirs_compact_raw_df = nirs_compact_raw_df.loc[keep_mask].copy()
    nirs_time_s = nirs_time_s[keep_mask]

    # Absolute per-sample timestamps from header start
    nirs_sample_timestamp = start_dt + pd.to_timedelta(nirs_time_s, unit="s")

    # Convert to master-relative seconds via ts_ref
    nirs_time_on_master_s = (nirs_sample_timestamp - ts_ref).total_seconds().to_numpy(dtype=float)

    # Monotonic check (fail loud)
    assert np.all(np.diff(nirs_time_on_master_s) > 0), "NIRS time_on_master_s must be strictly increasing"

    # ----------------------------
    # Deterministic nearest mapping to master grid 
    # ----------------------------
    low_mask = nirs_time_on_master_s < master_time_ref_s[0]
    high_mask = nirs_time_on_master_s > master_time_ref_s[-1]
    drop_low = int(np.sum(low_mask))
    drop_high = int(np.sum(high_mask))
    inrange_mask = ~(low_mask | high_mask)

    nirs_compact_inrange_df = nirs_compact_raw_df.loc[inrange_mask].copy()
    nirs_time_on_master_s_inrange = nirs_time_on_master_s[inrange_mask]

    cand = np.searchsorted(master_time_ref_s, nirs_time_on_master_s_inrange, side="left")
    cand = np.clip(cand, 1, len(master_time_ref_s) - 1)
    left = master_time_ref_s[cand - 1]
    right = master_time_ref_s[cand]
    choose_left = (nirs_time_on_master_s_inrange - left) <= (right - nirs_time_on_master_s_inrange)
    mapped_time_index = np.where(choose_left, cand - 1, cand).astype(int)

    nirs_compact_inrange_df.insert(0, "time_index", mapped_time_index)
    nirs_compact_inrange_df.insert(1, "nirs_time_on_master_s", nirs_time_on_master_s_inrange)

    nirs_compact_inrange_df["SEQ_index"] = master_index_grid.loc[mapped_time_index, "SEQ_index"].to_numpy()
    nirs_compact_inrange_df["VC"] = master_index_grid.loc[mapped_time_index, "VC"].to_numpy()
    nirs_compact_inrange_df["VC_count"] = master_index_grid.loc[mapped_time_index, "VC_count"].to_numpy()

    assert nirs_compact_inrange_df["time_index"].dtype.kind in ("i", "u"), "time_index must be integer"
    assert (
        nirs_compact_inrange_df["time_index"].min() >= 0
        and nirs_compact_inrange_df["time_index"].max() <= (len(master_index_grid) - 1)
    ), "time_index out of master bounds after in-range filter"

    # ----------------------------
    # Cache + report
    # ----------------------------
    nirs_compact_df_out = nirs_compact_inrange_df

    nirs_import_report_df_out = pd.DataFrame([{
        "RUN_ID": RUN_ID,
        "nirs_txt": nirs_txt_path.name,
        "fs_nirs_hz": float(FS_NIRS_HZ),
        "trim_head_s": float(NIRS_TRIM_HEAD_S),
        "start_of_measurement": str(start_dt),
        "n_raw_rows": int(len(nirs_compact_raw_df)),
        "n_kept_rows": int(len(nirs_compact_inrange_df)),
        "drop_low_import": drop_low,
        "drop_high_import": drop_high,
        "n_short_lines_dropped": int(short_line_count),
        "master_time_ref_s_min": float(master_time_ref_s[0]),
        "master_time_ref_s_max": float(master_time_ref_s[-1]),
        "nirs_time_on_master_s_min": float(np.min(nirs_time_on_master_s_inrange)) if len(nirs_time_on_master_s_inrange) else np.nan,
        "nirs_time_on_master_s_max": float(np.max(nirs_time_on_master_s_inrange)) if len(nirs_time_on_master_s_inrange) else np.nan,
    }])

    nirs_compact_df_out.to_parquet(cache_compact_path, index=False)
    nirs_import_report_df_out.to_parquet(cache_report_path, index=False)

    print("[07_nirs_import] wrote:")
    print("  -", cache_compact_path.name)
    print("  -", cache_report_path.name)

# Optional dict-style output 
nirs_tables_out = {
    "nirs_compact_df": nirs_compact_df_out,
    "nirs_import_report_df": nirs_import_report_df_out,
}