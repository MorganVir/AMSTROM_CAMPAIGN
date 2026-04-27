# _05a_bia_import.py - BIA IMPORT
# Load bioimpedance (BIA) measurement files and align them to the master timeline.
# BIA measures the electrical resistance and reactance of muscle tissue across a
# range of frequencies (multi-frequency bioimpedance spectroscopy). Two electrode
# configurations are recorded in parallel:
#   2PT (two-point): current injection and voltage measurement on the same pair of
#        electrodes — includes skin contact impedance, less precise.
#   4PT (four-point): separate injection and sensing electrodes — cancels contact
#        impedance, the more reliable measurement for deep tissue.
#
# Inputs
#   ctx keys  : RUN_ID, CACHE_DIR, master_index_grid
#   parameter : raw_bia_dir — folder containing <subject_key>_<date>_BIA_expe_Z2PT.pkl
#                             and the matching Z4PT.pkl
#               ts_ref       — session reference timestamp (Delsys clock anchor)
#               force_recompute — if True, bypasses cache and re-reads pkl files
#   files     : data/raw_signal/bia/<subject_key>_<date>_BIA_expe_Z2PT.pkl
#               data/raw_signal/bia/<subject_key>_<date>_BIA_expe_Z4PT.pkl
#
# Outputs (ctx keys set by the notebook after this call)
#   - bia2_df         : aligned 2PT impedance table with time_index and SEQ/VC columns
#   - bia4_df         : aligned 4PT impedance table (same layout)
#
# Cache
#   - 05a_bia2_compact.parquet    (2PT signal table)
#   - 05a_bia4_compact.parquet    (4PT signal table)
#   - 05a_bia2_freqs_hz.parquet   (ordered frequency vector for 2PT)
#   - 05a_bia4_freqs_hz.parquet   (ordered frequency vector for 4PT)
#
# Column naming convention in the compact DataFrames
#   Each impedance component gets one column per frequency:
#     <prefix>_R_ohm__f_<tag>Hz   → resistance (real part of Z), in ohms
#     <prefix>_Xc_ohm__f_<tag>Hz  → reactance (imaginary part of Z), in ohms
#     <prefix>_PhA_deg__f_<tag>Hz → phase angle = arctan(Xc / R), in degrees
#   <tag> is the frequency as an integer if whole (e.g. 1000 Hz → "1000"),
#   or dot-replaced if fractional (e.g. 2.5 Hz → "2p5").
#
# Notes
#   - BIA files are pkl DataFrames produced by the LIRMM prototype acquisition system.
#     Each row is one measurement snapshot; columns are named "f_<freq_hz>" and hold
#     complex impedance values (Z = R + jXc).
#   - Timestamps in the pkl are wall-clock datetimes; ts_ref converts them to session
#     seconds before nearest-neighbour alignment to the master grid.
#   - All four cache files must be present together for a cache hit — a partial cache
#     is treated as a miss and the pkl files are re-read.

import pickle
import re
import numpy as np
import pandas as pd


# Number of characters in RUN_ID that identify the subject (not the session date).
# Example: RUN_ID = "011BeSa_20251023" → subject_key = "011BeSa" (first 7 chars).
# Must match SUBJECT_KEY_LEN in _01_participant.py; kept local to avoid cross-imports.
SUBJECT_KEY_LEN = 7


def run_bia_import(
    ctx,
    raw_bia_dir,
    ts_ref,
    force_recompute,
):
    """
    Import BIA 2PT and 4PT pkl files and align to the master timeline.

    Each pkl is a DataFrame of complex impedance snapshots across many frequencies.
    The function extracts resistance (R), reactance (Xc), and phase angle (PhA) for
    every frequency, maps each BIA sample to the nearest master grid point via
    nearest-neighbour time matching, and stamps SEQ and VC labels from the master grid.

    Returns (bia2_compact_df, bia4_compact_df, bia2_freqs_hz, bia4_freqs_hz).
    bia2/bia4_freqs_hz are 1-D float arrays listing the measurement frequencies in Hz,
    in ascending order — needed by downstream steps to index columns by frequency.
    """

    # --- Resolve ctx ---
    run_id            = ctx["RUN_ID"]
    cache_dir         = ctx["CACHE_DIR"]
    master_index_grid = ctx["master_index_grid"]

    # --- Cache paths ---
    # Registered in ctx["parquet_path"] so the notebook dashboard can reference them
    # without hardcoding filenames (e.g. to display cache status or force re-runs).
    CACHE_BIA_2       = cache_dir / "05a_bia2_compact.parquet"
    CACHE_BIA_4       = cache_dir / "05a_bia4_compact.parquet"
    CACHE_BIA_2_FREQS = cache_dir / "05a_bia2_freqs_hz.parquet"
    CACHE_BIA_4_FREQS = cache_dir / "05a_bia4_freqs_hz.parquet"

    ctx.setdefault("parquet_path", {})
    ctx["parquet_path"]["CACHE_BIA_2"]       = CACHE_BIA_2
    ctx["parquet_path"]["CACHE_BIA_4"]       = CACHE_BIA_4
    ctx["parquet_path"]["CACHE_BIA_2_FREQS"] = CACHE_BIA_2_FREQS
    ctx["parquet_path"]["CACHE_BIA_4_FREQS"] = CACHE_BIA_4_FREQS

    # --- Cache check ---
    # All four files must be present together — a partial cache is treated as a miss.
    cache_all_present = (
        CACHE_BIA_2.exists()
        and CACHE_BIA_4.exists()
        and CACHE_BIA_2_FREQS.exists()
        and CACHE_BIA_4_FREQS.exists()
    )
    if cache_all_present and not force_recompute:
        bia2_compact_df = pd.read_parquet(CACHE_BIA_2)
        bia4_compact_df = pd.read_parquet(CACHE_BIA_4)
        bia2_freqs_hz   = pd.read_parquet(CACHE_BIA_2_FREQS)["freq_hz"].to_numpy(dtype=float)
        bia4_freqs_hz   = pd.read_parquet(CACHE_BIA_4_FREQS)["freq_hz"].to_numpy(dtype=float)
        print("[05a_bia_import] Cache hit — loaded from cache. Set force_recompute=True to re-import.")
        return bia2_compact_df, bia4_compact_df, bia2_freqs_hz, bia4_freqs_hz

    # --- Resolve file paths ---
    # BIA pkl files are named by subject key + date, not the full RUN_ID.
    # The date portion is the 8-digit segment after the first underscore in RUN_ID.
    subject_key = run_id[:SUBJECT_KEY_LEN]
    date_part   = run_id.split("_")[1]

    bia2_pickle_path = raw_bia_dir / f"{subject_key}_{date_part}_BIA_expe_Z2PT.pkl"
    bia4_pickle_path = raw_bia_dir / f"{subject_key}_{date_part}_BIA_expe_Z4PT.pkl"

    print(f"[05a_bia_import] Loading 2PT: {bia2_pickle_path.name}")
    print(f"[05a_bia_import] Loading 4PT: {bia4_pickle_path.name}")

    # --- Prepare master grid arrays ---
    # Extracted once here so inner functions can close over them without re-fetching.
    master_time_ref_seconds = master_index_grid["time_ref_s"].to_numpy(dtype=float)
    master_seq_index        = master_index_grid["SEQ_index"].to_numpy(dtype=int)
    master_seq              = master_index_grid["SEQ"].to_numpy(dtype=object)
    master_vc               = master_index_grid["VC"].to_numpy(dtype=int)
    master_vc_count         = master_index_grid["VC_count"].to_numpy(dtype=int)

    # --- Inner helpers ---

    def load_pickle_full_spectrum(path):
        """
        Read one BIA pkl file and return (timestamps, freqs, Z).

        timestamps : numpy array of datetime64 — one entry per measurement snapshot
        freqs      : float array of frequencies in Hz, sorted ascending
        Z          : complex128 array shaped (n_snapshots, n_freqs)

        Frequency columns in the pkl are named "f_<value>" (e.g. "f_1000.0", "f_2.5").
        They are sorted ascending so downstream steps can rely on a consistent order.
        """
        with open(path, "rb") as fh:
            obj = pickle.load(fh)

        timestamps = pd.to_datetime(obj["timestamp"]).to_numpy()

        # Collect and sort frequency columns.
        # Sorting is done on the numeric frequency value, not the string, to avoid
        # lexicographic misordering (e.g. "f_10" < "f_2" alphabetically).
        freq_cols = [c for c in obj.columns if isinstance(c, str) and c.startswith("f_")]
        freqs     = np.array([float(c.split("_")[1]) for c in freq_cols], dtype=float)
        order     = np.argsort(freqs)
        freqs     = freqs[order]
        freq_cols = [freq_cols[i] for i in order]

        Z = obj[freq_cols].to_numpy(dtype=np.complex128)
        return timestamps, freqs, Z

    def map_to_master(bia_time_s):
        """
        Map BIA sample times (in session seconds) to the nearest master grid time_index.

        Uses the same nearest-neighbour pattern as torque in _02_: for each BIA sample,
        find the two surrounding master grid points and take whichever is closer.
        Returns an int array of time_index values, one per BIA sample.
        """
        idx   = np.searchsorted(master_time_ref_seconds, bia_time_s, side="left")
        idx   = np.clip(idx, 1, len(master_time_ref_seconds) - 1)
        left  = master_time_ref_seconds[idx - 1]
        right = master_time_ref_seconds[idx]
        # Choose the closer grid point; ties go to the left (earlier) sample.
        choose_left = (bia_time_s - left) <= (right - bia_time_s)
        return np.where(choose_left, idx - 1, idx).astype(int)

    def build_compact(prefix, timestamps, freqs, Z):
        """
        Build the compact aligned DataFrame for one BIA configuration (2PT or 4PT).

        Converts wall-clock timestamps to session seconds using ts_ref, maps each
        sample to the master grid, stamps SEQ/VC labels, then appends one column
        triplet (R, Xc, PhA) per measurement frequency.

        prefix : "bia2" or "bia4" — used as a column name prefix
        Z      : complex128 array shaped (n_samples, n_freqs)

        Column naming example (prefix="bia2", f=1000.0 Hz):
            bia2_R_ohm__f_1000Hz, bia2_Xc_ohm__f_1000Hz, bia2_PhA_deg__f_1000Hz
        Column naming example (prefix="bia4", f=2.5 Hz):
            bia4_R_ohm__f_2p5Hz  (dot replaced with "p" so the name is parquet-safe)
        """
        # Convert wall-clock timestamps to session seconds anchored at ts_ref.
        time_s     = (pd.to_datetime(timestamps) - ts_ref).total_seconds().to_numpy(dtype=float)
        mapped_idx = map_to_master(time_s)

        df = pd.DataFrame({
            "time_index":                    mapped_idx,
            "SEQ_index":                     master_seq_index[mapped_idx],
            "SEQ":                           master_seq[mapped_idx],
            "VC":                            master_vc[mapped_idx],
            "VC_count":                      master_vc_count[mapped_idx],
            f"{prefix}_time_on_master_s":    time_s,
        })

        R   = Z.real.astype(float, copy=False)
        X   = Z.imag.astype(float, copy=False)
        # Phase angle = arctan(Xc / R) in degrees — a summary metric used in BIA
        # literature as a proxy for cell membrane integrity and hydration status.
        PhA = np.degrees(np.arctan2(X, R))

        for i, f in enumerate(freqs):
            # Build a parquet-safe frequency tag: whole numbers stay as integers
            # (1000.0 → "1000"), fractional values replace the dot with "p" (2.5 → "2p5").
            tag = str(int(f)) if float(f).is_integer() else str(f).replace(".", "p")
            df[f"{prefix}_R_ohm__f_{tag}Hz"]   = R[:, i]
            df[f"{prefix}_Xc_ohm__f_{tag}Hz"]  = X[:, i]
            df[f"{prefix}_PhA_deg__f_{tag}Hz"]  = PhA[:, i]

        # Sort by time_index so downstream steps can assume chronological order.
        return df.sort_values("time_index", kind="mergesort").reset_index(drop=True)

    # --- Compute ---
    t2, f2, Z2 = load_pickle_full_spectrum(bia2_pickle_path)
    t4, f4, Z4 = load_pickle_full_spectrum(bia4_pickle_path)

    bia2_compact_df = build_compact("bia2", t2, f2, Z2)
    bia4_compact_df = build_compact("bia4", t4, f4, Z4)

    # --- Write cache ---
    bia2_compact_df.to_parquet(CACHE_BIA_2, index=False)
    bia4_compact_df.to_parquet(CACHE_BIA_4, index=False)
    pd.DataFrame({"freq_hz": f2}).to_parquet(CACHE_BIA_2_FREQS, index=False)
    pd.DataFrame({"freq_hz": f4}).to_parquet(CACHE_BIA_4_FREQS, index=False)

    print("[05a_bia_import] Done — BIA cache written.")
    return bia2_compact_df, bia4_compact_df, f2, f4
