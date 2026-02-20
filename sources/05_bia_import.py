
# 05_bia_import - load BIA 2PT/4PT + map to master time_index

from pathlib import Path
import pickle
import re
import numpy as np
import pandas as pd

# ----------------------------
# REQUIRED INPUTS (asserts only)
# ----------------------------
assert "RUN_ID" in globals(), "RUN_ID missing"
assert isinstance(RUN_ID, str) and len(RUN_ID) >= 7, "RUN_ID must start with subject key (>=7 chars)"

assert "ts_ref" in globals(), "ts_ref missing (absolute session anchor Timestamp)"
assert isinstance(ts_ref, pd.Timestamp), "ts_ref must be a pandas Timestamp"

assert "RAW_BIA_DIR" in globals(), "RAW_BIA_DIR missing (expected RAW_ROOT / 'bia')"
assert isinstance(RAW_BIA_DIR, Path), "RAW_BIA_DIR must be a pathlib.Path"
assert RAW_BIA_DIR.exists(), f"RAW_BIA_DIR does not exist: {RAW_BIA_DIR}"

assert "CACHE_05_BIA_IMPORT" in globals(), "CACHE_05_BIA_IMPORT missing (Path to base parquet handle)"
assert isinstance(CACHE_05_BIA_IMPORT, Path), "CACHE_05_BIA_IMPORT must be a pathlib.Path"
CACHE_05_BIA_IMPORT.parent.mkdir(parents=True, exist_ok=True)

assert "master_index_grid" in globals(), "master_index_grid missing"
assert isinstance(master_index_grid, pd.DataFrame), "master_index_grid must be a pandas DataFrame"

# ----------------------------
# Master grid contract 
# ----------------------------
required_master_columns = ["time_ref_s", "SEQ_index", "VC", "VC_count"]
for required_column in required_master_columns:
    assert required_column in master_index_grid.columns, f"master_index_grid must contain {required_column}"

master_time_ref_seconds = master_index_grid["time_ref_s"].to_numpy(dtype=float)
assert master_time_ref_seconds.ndim == 1 and len(master_time_ref_seconds) == len(master_index_grid), "time_ref_s length mismatch"
assert np.all(np.diff(master_time_ref_seconds) > 0), "time_ref_s must be strictly increasing"

master_seq_index = master_index_grid["SEQ_index"].to_numpy(dtype=int)
master_vc = master_index_grid["VC"].to_numpy(dtype=int)
master_vc_count = master_index_grid["VC_count"].to_numpy(dtype=int)


# Build raw file paths from RUN_ID (minimal, deterministic)
subject_key = RUN_ID[:7]
run_parts = RUN_ID.split("_")
assert len(run_parts) >= 2, "RUN_ID must contain '_' with date part, e.g. 001AaBb_20251023"
experiment_date_yyyymmdd = run_parts[1]
assert re.fullmatch(r"\d{8}", experiment_date_yyyymmdd), f"RUN_ID date part is not YYYYMMDD: {experiment_date_yyyymmdd}"

bia2_pickle_path = RAW_BIA_DIR / f"{subject_key}_{experiment_date_yyyymmdd}_BIA_expe_Z2PT.pkl"
bia4_pickle_path = RAW_BIA_DIR / f"{subject_key}_{experiment_date_yyyymmdd}_BIA_expe_Z4PT.pkl"

assert bia2_pickle_path.exists(), f"Missing BIA 2PT file: {bia2_pickle_path}"
assert bia4_pickle_path.exists(), f"Missing BIA 4PT file: {bia4_pickle_path}"

print(f"[05_bia_import] RUN_ID={RUN_ID}")
print(f"[05_bia_import] ts_ref = {ts_ref}")
print(f"[05_bia_import] RAW_BIA_DIR = {RAW_BIA_DIR}")
print(f"[05_bia_import] Loading 2PT = {bia2_pickle_path.name}")
print(f"[05_bia_import] Loading 4PT = {bia4_pickle_path.name}")


# Cache paths derived from the single cache handle

cache_parent_dir = CACHE_05_BIA_IMPORT.parent

cache_bia2_compact_path = cache_parent_dir / "05_bia2_compact.parquet"
cache_bia4_compact_path = cache_parent_dir / "05_bia4_compact.parquet"
cache_bia2_freqs_path   = cache_parent_dir / "05_bia2_freqs_hz.parquet"
cache_bia4_freqs_path   = cache_parent_dir / "05_bia4_freqs_hz.parquet"


cache_all_present = (
    cache_bia2_compact_path.exists()
    and cache_bia4_compact_path.exists()
    and cache_bia2_freqs_path.exists()
    and cache_bia4_freqs_path.exists()
)


# Local helpers (explicit names)

def load_bia_pickle_full_spectrum(bia_pickle_file_path: Path):
    loaded_object = pickle.load(open(bia_pickle_file_path, "rb"))
    assert isinstance(loaded_object, pd.DataFrame), f"Pickle did not contain a DataFrame: {bia_pickle_file_path.name}"

    assert "timestamp" in loaded_object.columns, f"'timestamp' column missing in {bia_pickle_file_path.name}"
    sample_timestamps_dt = pd.to_datetime(loaded_object["timestamp"]).to_numpy()

    frequency_column_names = [
        column_name
        for column_name in loaded_object.columns
        if isinstance(column_name, str) and column_name.startswith("f_")
    ]
    assert len(frequency_column_names) > 0, f"No 'f_*' columns found in {bia_pickle_file_path.name}"

    frequency_values_hz = np.array(
        [float(column_name.split("_")[1]) for column_name in frequency_column_names],
        dtype=float,
    )
    frequency_sort_order = np.argsort(frequency_values_hz)

    frequency_values_hz = frequency_values_hz[frequency_sort_order]
    frequency_column_names = [frequency_column_names[int(index_value)] for index_value in frequency_sort_order]

    impedance_complex_matrix = loaded_object[frequency_column_names].to_numpy(dtype=np.complex128)
    expected_shape = (len(sample_timestamps_dt), len(frequency_values_hz))
    assert impedance_complex_matrix.shape == expected_shape, f"Impedance matrix shape mismatch: got {impedance_complex_matrix.shape}, expected {expected_shape}"

    return sample_timestamps_dt, frequency_values_hz, impedance_complex_matrix


def format_frequency_tag(frequency_hz: float) -> str:
    if float(frequency_hz).is_integer():
        return f"{int(frequency_hz)}"
    return str(frequency_hz).replace(".", "p")


def map_times_to_master_time_index(bia_time_on_master_seconds: np.ndarray, master_time_ref_seconds: np.ndarray) -> np.ndarray:
    candidate_indices = np.searchsorted(master_time_ref_seconds, bia_time_on_master_seconds, side="left")
    candidate_indices = np.clip(candidate_indices, 1, len(master_time_ref_seconds) - 1)

    left_times = master_time_ref_seconds[candidate_indices - 1]
    right_times = master_time_ref_seconds[candidate_indices]

    choose_left_mask = (bia_time_on_master_seconds - left_times) <= (right_times - bia_time_on_master_seconds)
    mapped_time_index = np.where(choose_left_mask, candidate_indices - 1, candidate_indices).astype(int)
    return mapped_time_index


def build_bia_compact_table(prefix: str, sample_timestamps_dt: np.ndarray, frequency_values_hz: np.ndarray, impedance_complex_matrix: np.ndarray):
    assert isinstance(prefix, str) and len(prefix) > 0, "prefix must be a non-empty string"

    sample_timestamps_series = pd.to_datetime(sample_timestamps_dt)
    bia_time_on_master_seconds = (sample_timestamps_series - ts_ref).total_seconds().to_numpy(dtype=float)

    # Fail-loud monotonic check
    bia_time_deltas_seconds = np.diff(bia_time_on_master_seconds)
    non_monotonic_indices = np.where(bia_time_deltas_seconds < 0)[0]

    print(
        f"[05_bia_import] {prefix} monotonic check: n={bia_time_on_master_seconds.size} | "
        f"bad_count={int(non_monotonic_indices.size)} | "
        f"min_dt_s={float(np.min(bia_time_deltas_seconds)) if bia_time_deltas_seconds.size else np.nan}"
    )

    if non_monotonic_indices.size:
        first_bad_index = int(non_monotonic_indices[0])
        debug_slice = slice(max(0, first_bad_index - 3), min(bia_time_on_master_seconds.size, first_bad_index + 6))

        print("[05_bia_import] first bad idx =", first_bad_index, "| dt_s =", float(bia_time_deltas_seconds[first_bad_index]))
        print(f"[05_bia_import] {prefix}_time_on_master_s around bad:", bia_time_on_master_seconds[debug_slice])
        print(f"[05_bia_import] {prefix} sample timestamps around bad:", sample_timestamps_series.to_numpy()[debug_slice])
        raise ValueError(f"{prefix} time_on_master_seconds not monotonic (cannot map reliably).")

    mapped_time_index = map_times_to_master_time_index(bia_time_on_master_seconds, master_time_ref_seconds)

    compact_dataframe = pd.DataFrame({
        "time_index": mapped_time_index,
        "SEQ_index": master_seq_index[mapped_time_index],
        "VC": master_vc[mapped_time_index],
        "VC_count": master_vc_count[mapped_time_index],
        f"{prefix}_time_on_master_s": bia_time_on_master_seconds,
    })

    resistance_ohm = impedance_complex_matrix.real.astype(float, copy=False)
    reactance_ohm = impedance_complex_matrix.imag.astype(float, copy=False)
    phase_angle_deg = np.degrees(np.arctan2(reactance_ohm, resistance_ohm))

    for frequency_column_index, frequency_hz in enumerate(frequency_values_hz):
        frequency_tag = format_frequency_tag(frequency_hz)
        compact_dataframe[f"{prefix}_R_ohm__f_{frequency_tag}Hz"] = resistance_ohm[:, frequency_column_index]
        compact_dataframe[f"{prefix}_Xc_ohm__f_{frequency_tag}Hz"] = reactance_ohm[:, frequency_column_index]
        compact_dataframe[f"{prefix}_PhA_deg__f_{frequency_tag}Hz"] = phase_angle_deg[:, frequency_column_index]

    compact_dataframe = compact_dataframe.sort_values("time_index", kind="mergesort").reset_index(drop=True)
    return compact_dataframe



# Cache load (if present) else compute + write cache

if cache_all_present:
    print("[05_bia_import] cache hit -> loading parquet siblings")

    bia2_compact_loaded = pd.read_parquet(cache_bia2_compact_path)
    bia4_compact_loaded = pd.read_parquet(cache_bia4_compact_path)

    bia2_freqs_loaded = pd.read_parquet(cache_bia2_freqs_path)
    bia4_freqs_loaded = pd.read_parquet(cache_bia4_freqs_path)

    # Fail-loud schema checks
    for required_column in ["time_index", "SEQ_index", "VC", "VC_count", "bia2_time_on_master_s"]:
        assert required_column in bia2_compact_loaded.columns, f"cache bia2 missing column: {required_column}"
    for required_column in ["time_index", "SEQ_index", "VC", "VC_count", "bia4_time_on_master_s"]:
        assert required_column in bia4_compact_loaded.columns, f"cache bia4 missing column: {required_column}"

    assert "freq_hz" in bia2_freqs_loaded.columns, "cache bia2 freqs missing column: freq_hz"
    assert "freq_hz" in bia4_freqs_loaded.columns, "cache bia4 freqs missing column: freq_hz"

    bia2_compact_df = bia2_compact_loaded
    bia4_compact_df = bia4_compact_loaded
    bia2_freqs_hz = bia2_freqs_loaded["freq_hz"].to_numpy(dtype=float)
    bia4_freqs_hz = bia4_freqs_loaded["freq_hz"].to_numpy(dtype=float)

else:
    print("[05_bia_import] cache miss -> computing and writing parquet siblings")

    bia2_sample_timestamps_dt, bia2_frequency_values_hz, bia2_impedance_complex_matrix = load_bia_pickle_full_spectrum(bia2_pickle_path)
    bia4_sample_timestamps_dt, bia4_frequency_values_hz, bia4_impedance_complex_matrix = load_bia_pickle_full_spectrum(bia4_pickle_path)

    bia2_compact_df = build_bia_compact_table("bia2", bia2_sample_timestamps_dt, bia2_frequency_values_hz, bia2_impedance_complex_matrix)
    bia4_compact_df = build_bia_compact_table("bia4", bia4_sample_timestamps_dt, bia4_frequency_values_hz, bia4_impedance_complex_matrix)

    bia2_freqs_hz = bia2_frequency_values_hz
    bia4_freqs_hz = bia4_frequency_values_hz

    bia2_compact_df.to_parquet(cache_bia2_compact_path, index=False)
    bia4_compact_df.to_parquet(cache_bia4_compact_path, index=False)
    pd.DataFrame({"freq_hz": bia2_freqs_hz}).to_parquet(cache_bia2_freqs_path, index=False)
    pd.DataFrame({"freq_hz": bia4_freqs_hz}).to_parquet(cache_bia4_freqs_path, index=False)

    print("[05_bia_import] wrote:")
    print("  -", cache_bia2_compact_path.name)
    print("  -", cache_bia4_compact_path.name)
    print("  -", cache_bia2_freqs_path.name)
    print("  -", cache_bia4_freqs_path.name)


# Explicit outputs
# ----------------------------
bia2_compact_df_out = bia2_compact_df
bia4_compact_df_out = bia4_compact_df
bia2_freqs_hz_out = bia2_freqs_hz
bia4_freqs_hz_out = bia4_freqs_hz

bia_tables_out = {
    "bia2_compact_df": bia2_compact_df_out,
    "bia4_compact_df": bia4_compact_df_out,
    "bia2_freqs_hz": bia2_freqs_hz_out,
    "bia4_freqs_hz": bia4_freqs_hz_out,
}
