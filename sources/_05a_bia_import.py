# 05a_bia_import.py
# Load BIA 2PT / 4PT and map to master time_index

from pathlib import Path
import pickle
import re
import numpy as np
import pandas as pd


def run_bia_import(
    ctx,
    raw_bia_dir,
    ts_ref,
    force_recompute,
):
    """
    Import BIA 2PT and 4PT raw pickle files and map to master backbone.

    Parameters
    ----------
    ctx : dict
        Must contain:
            - RUN_ID
            - CACHE_DIR
            - master_index_grid

    raw_bia_dir : Path
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

    assert isinstance(run_id, str) and len(run_id) >= 7
    assert isinstance(cache_dir, Path)
    assert isinstance(raw_bia_dir, Path)
    assert isinstance(ts_ref, pd.Timestamp)
    assert raw_bia_dir.exists()

    required_master_cols = ["time_ref_s", "SEQ_index", "VC", "VC_count"]
    for c in required_master_cols:
        assert c in master_index_grid.columns

    master_time_ref_seconds = master_index_grid["time_ref_s"].to_numpy(dtype=float)
    assert np.all(np.diff(master_time_ref_seconds) > 0)

    master_seq_index = master_index_grid["SEQ_index"].to_numpy(dtype=int)
    master_vc = master_index_grid["VC"].to_numpy(dtype=int)
    master_vc_count = master_index_grid["VC_count"].to_numpy(dtype=int)


    # Cache paths 
    CACHE_BIA_2 = cache_dir / "05a_bia2_compact.parquet"
    CACHE_BIA_4 = cache_dir / "05a_bia4_compact.parquet"
    CACHE_BIA_2_FREQS = cache_dir / "05a_bia2_freqs_hz.parquet"
    CACHE_BIA_4_FREQS = cache_dir / "05a_bia4_freqs_hz.parquet"

    # register specific cache paths (so dashboard commit can reuse them without hardcoding )
    ctx.setdefault("parquet_path", {})
    ctx["parquet_path"]["CACHE_BIA_2"] = CACHE_BIA_2
    ctx["parquet_path"]["CACHE_BIA_4"] = CACHE_BIA_4
    ctx["parquet_path"]["CACHE_BIA_2_FREQS"] = CACHE_BIA_2_FREQS
    ctx["parquet_path"]["CACHE_BIA_4_FREQS"] = CACHE_BIA_4_FREQS




    cache_all_present = (
        CACHE_BIA_2.exists()
        and CACHE_BIA_4.exists()
        and CACHE_BIA_2_FREQS.exists()
        and CACHE_BIA_4_FREQS.exists()
    )
    # cache hit
    if cache_all_present and not force_recompute:

        print("[05a_bia_import] BIA cache hit, loading from cache. Force recomputing or delete cache files to re-run import.")

        bia2_compact_df = pd.read_parquet(CACHE_BIA_2)
        bia4_compact_df = pd.read_parquet(CACHE_BIA_4)

        bia2_freqs_hz = pd.read_parquet(CACHE_BIA_2_FREQS)["freq_hz"].to_numpy(dtype=float)
        bia4_freqs_hz = pd.read_parquet(CACHE_BIA_4_FREQS)["freq_hz"].to_numpy(dtype=float)

        return bia2_compact_df, bia4_compact_df, bia2_freqs_hz, bia4_freqs_hz


    # Select raw files from RUN_ID
    subject_key = run_id[:7]
    run_parts = run_id.split("_")
    assert len(run_parts) >= 2
    date_part = run_parts[1]
    assert re.fullmatch(r"\d{8}", date_part)

    bia2_pickle_path = raw_bia_dir / f"{subject_key}_{date_part}_BIA_expe_Z2PT.pkl"
    bia4_pickle_path = raw_bia_dir / f"{subject_key}_{date_part}_BIA_expe_Z4PT.pkl"

    assert bia2_pickle_path.exists()
    assert bia4_pickle_path.exists()

    print(f"[05a_bia_import] loading 2PT: {bia2_pickle_path.name}")
    print(f"[05a_bia_import] loading 4PT: {bia4_pickle_path.name}")


    # load raw pickles and map to master time_index
    def load_pickle_full_spectrum(path: Path):

        obj = pickle.load(open(path, "rb"))
        assert isinstance(obj, pd.DataFrame)
        assert "timestamp" in obj.columns

        timestamps = pd.to_datetime(obj["timestamp"]).to_numpy()

        freq_cols = [
            c for c in obj.columns
            if isinstance(c, str) and c.startswith("f_")
        ]
        assert len(freq_cols) > 0

        freqs = np.array(
            [float(c.split("_")[1]) for c in freq_cols],
            dtype=float,
        )
        order = np.argsort(freqs)

        freqs = freqs[order]
        freq_cols = [freq_cols[i] for i in order]

        Z = obj[freq_cols].to_numpy(dtype=np.complex128)
        assert Z.shape == (len(timestamps), len(freqs))

        return timestamps, freqs, Z

    def map_to_master(bia_time_s):

        idx = np.searchsorted(master_time_ref_seconds, bia_time_s, side="left")
        idx = np.clip(idx, 1, len(master_time_ref_seconds) - 1)

        left = master_time_ref_seconds[idx - 1]
        right = master_time_ref_seconds[idx]

        choose_left = (bia_time_s - left) <= (right - bia_time_s)
        return np.where(choose_left, idx - 1, idx).astype(int)

    def build_compact(prefix, timestamps, freqs, Z):

        time_s = (pd.to_datetime(timestamps) - ts_ref).total_seconds().to_numpy(dtype=float)

        assert np.all(np.diff(time_s) >= 0), f"{prefix} non-monotonic time"

        mapped_idx = map_to_master(time_s)

        df = pd.DataFrame({
            "time_index": mapped_idx,
            "SEQ_index": master_seq_index[mapped_idx],
            "VC": master_vc[mapped_idx],
            "VC_count": master_vc_count[mapped_idx],
            f"{prefix}_time_on_master_s": time_s,
        })

        R = Z.real.astype(float, copy=False)
        X = Z.imag.astype(float, copy=False)
        PhA = np.degrees(np.arctan2(X, R)) #convert into phase angle (degrees)

        for i, f in enumerate(freqs):
            tag = str(int(f)) if float(f).is_integer() else str(f).replace(".", "p")
            df[f"{prefix}_R_ohm__f_{tag}Hz"] = R[:, i]
            df[f"{prefix}_Xc_ohm__f_{tag}Hz"] = X[:, i]
            df[f"{prefix}_PhA_deg__f_{tag}Hz"] = PhA[:, i]

        return df.sort_values("time_index", kind="mergesort").reset_index(drop=True)


    # Compute
    t2, f2, Z2 = load_pickle_full_spectrum(bia2_pickle_path)
    t4, f4, Z4 = load_pickle_full_spectrum(bia4_pickle_path)

    bia2_compact_df = build_compact("bia2", t2, f2, Z2)
    bia4_compact_df = build_compact("bia4", t4, f4, Z4)

    bia2_freqs_hz = f2
    bia4_freqs_hz = f4


    # Cache write

    bia2_compact_df.to_parquet(CACHE_BIA_2, index=False)
    bia4_compact_df.to_parquet(CACHE_BIA_4, index=False)
    pd.DataFrame({"freq_hz": f2}).to_parquet(CACHE_BIA_2_FREQS, index=False)
    pd.DataFrame({"freq_hz": f4}).to_parquet(CACHE_BIA_4_FREQS, index=False)

    print("[05a_bia_import] cached BIA files to cache directory. Force recomputing or delete cache files to re-run import.")

    return bia2_compact_df, bia4_compact_df, bia2_freqs_hz, bia4_freqs_hz