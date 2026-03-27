# _09_export.py - FINAL EXPORT
# Write final QC pdf and selected parquet files from step caches.
#
# Inputs
#   export_final_qc_pdf:
#     - qc_figs
#     - run_id
#
#   export_final_parquet:
#     - ctx with RUN_ID and CACHE_DIR
#     - export_selection = "all" or list of parquet names
#
# Outputs
#   - results/QC_EXPORT/QC_PLOT_EXPORT_<RUN_ID>.pdf
#   - results/DATA_EXPORT/<RUN_ID>/<selected parquet files>
#
# Notes
#   - exported parquet tables receive run_id = RUN_ID
#   - export folder keeps the full RUN_ID

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


PARQUET_EXPORT_LIST = [
    "01_participants.parquet",
    "02_emg_compact.parquet",
    "02_master_index_grid.parquet",
    "02_torque_compact.parquet",
    "05a_bia2_freqs_hz.parquet",
    "05a_bia4_freqs_hz.parquet",
    "05b_bia2_compact_aligned.parquet",
    "05b_bia4_compact_aligned.parquet",
    "06b_nirs_compact_aligned.parquet",
    "07_myoton_compact.parquet",
]


def build_master_seq_lookup(master_index_grid: pd.DataFrame) -> tuple[dict[int, str], list[str]]:
    seq_pairs = master_index_grid[["SEQ", "SEQ_index"]].drop_duplicates().copy()
    seq_pairs["SEQ"] = seq_pairs["SEQ"].astype(str)
    seq_pairs["SEQ_index"] = seq_pairs["SEQ_index"].astype(int)

    seq_pairs = seq_pairs.sort_values("SEQ_index", kind="mergesort").reset_index(drop=True)
    seq_index_to_seq = dict(zip(seq_pairs["SEQ_index"], seq_pairs["SEQ"]))
    seq_values = list(seq_pairs["SEQ"])
    return seq_index_to_seq, seq_values
def build_sequence_valid_map(
    *,
    ctx: dict,
    sequence_valid_overrides: dict | None = None,
) -> dict[str, int]:
    master_index_grid = ctx["master_index_grid"]
    _, seq_values = build_master_seq_lookup(master_index_grid)

    sequence_valid_by_seq = {seq: 1 for seq in seq_values}

    if sequence_valid_overrides is None:
        return sequence_valid_by_seq

    for seq_key, raw_value in sequence_valid_overrides.items():
        seq = str(seq_key)
        sequence_valid_by_seq[seq] = int(raw_value)

    return sequence_valid_by_seq


def append_sequence_valid(
    *,
    table: pd.DataFrame,
    parquet_name: str,
    sequence_valid_by_seq: dict[str, int] | None,
    seq_index_to_seq: dict[int, str],
) -> pd.DataFrame:
    if sequence_valid_by_seq is None:
        return table

    has_seq = "SEQ" in table.columns
    has_seq_index = "SEQ_index" in table.columns
    if not has_seq and not has_seq_index:
        return table

    if has_seq:
        seq_series = pd.Series(table["SEQ"]).astype(str)
    else:
        seq_index_series = pd.Series(table["SEQ_index"]).astype(int)
        seq_from_index = seq_index_series.map(seq_index_to_seq)
        if seq_from_index.isna().any():
            bad_seq_index = int(seq_index_series[seq_from_index.isna()].iloc[0])
            raise AssertionError(
                f"Cannot resolve SEQ from SEQ_index in export table {parquet_name}: {bad_seq_index}"
            )
        seq_series = seq_from_index

    sequence_valid = seq_series.map(sequence_valid_by_seq)
    if sequence_valid.isna().any():
        bad_seq = str(seq_series[sequence_valid.isna()].iloc[0])
        raise AssertionError(f"Missing sequence_valid mapping for SEQ={bad_seq!r} in {parquet_name}")

    table_out = table.copy()
    table_out["sequence_valid"] = sequence_valid.astype(int)
    return table_out


def infer_master_period_s(table: pd.DataFrame) -> float:
    master_time_ref_s = table["time_ref_s"].to_numpy(dtype=float)
    master_time_deltas = np.diff(master_time_ref_s)

    period_s = float(np.median(master_time_deltas))
    return period_s


def export_final_qc_pdf(
    *,
    qc_figs: list,
    run_id: str,
    results_dir: Path = Path("results"),
) -> Path:
    """
    Export already-built QC figures to one PDF.
    """

    qc_export_root = results_dir / "QC_EXPORT"
    qc_export_root.mkdir(parents=True, exist_ok=True)

    qc_pdf_path = qc_export_root / f"QC_PLOT_EXPORT_{run_id}.pdf"

    with PdfPages(qc_pdf_path) as pdf:
        for fig in qc_figs:
            pdf.savefig(fig)

    print(f"[QC EXPORT] PDF created -> {qc_pdf_path.resolve()}")
    return qc_pdf_path
def export_final_parquet(
    *,
    ctx: dict,
    export_selection="all",
    sequence_valid_by_seq: dict[str, int] | None = None,
    run_id: str | None = None,
    cache_dir: Path | None = None,
    results_dir: Path = Path("results"),
) -> Path:
    """
    Export selected cache parquet tables to results/DATA_EXPORT/<RUN_ID>.
    """

    if run_id is None:
        run_id = ctx["RUN_ID"]
    if cache_dir is None:
        cache_dir = ctx["CACHE_DIR"]

    if not cache_dir.exists():
        raise FileNotFoundError(f"cache_dir not found: {cache_dir}")

    seq_index_to_seq, _ = build_master_seq_lookup(ctx["master_index_grid"])

    if export_selection == "all" or export_selection is None:
        parquet_names = list(PARQUET_EXPORT_LIST)
    else:
        parquet_names = [str(name) for name in export_selection]

    data_export_root = results_dir / "DATA_EXPORT" / run_id
    data_export_root.mkdir(parents=True, exist_ok=True)

    print("Parquet files selected for export:")
    for parquet_name in parquet_names:
        print(f"- {parquet_name}")

    export_count = 0
    for parquet_name in parquet_names:
        source_path = cache_dir / parquet_name
        if not source_path.exists():
            raise FileNotFoundError(f"Missing export file: {source_path}")

        source_table = pd.read_parquet(source_path)
        exported_table = append_sequence_valid(
            table=source_table,
            parquet_name=parquet_name,
            sequence_valid_by_seq=sequence_valid_by_seq,
            seq_index_to_seq=seq_index_to_seq,
        )
        sequence_valid_added = (
            "sequence_valid" in exported_table.columns and "sequence_valid" not in source_table.columns
        )
        period_s = None

        if parquet_name == "02_master_index_grid.parquet":
            period_s = infer_master_period_s(exported_table)
            exported_table = exported_table.copy()
            exported_table["period_s"] = period_s

        exported_table = exported_table.copy()
        exported_table["run_id"] = run_id

        export_path = data_export_root / parquet_name
        exported_table.to_parquet(export_path, index=False)

        print(f"[DATA EXPORT] Wrote -> {export_path}")
        summary_parts = [
            f"RUN_ID={run_id}",
            f"run_id_added={'yes'}",
            f"sequence_valid_added={'yes' if sequence_valid_added else 'no'}",
            f"period_s_added={'yes' if period_s is not None else 'no'}",
        ]
        if period_s is not None:
            summary_parts.append(f"period_s={period_s:.15f}")
        print(f"[DATA EXPORT CHECK] {parquet_name} | " + " | ".join(summary_parts))
        export_count += 1

    print("\nExported parquet files and their columns:")
    for parquet_name in parquet_names:
        source_path = cache_dir / parquet_name
        source_table = pd.read_parquet(source_path)
        export_columns = list(source_table.columns)
        if parquet_name != "01_participants.parquet" and "sequence_valid" not in export_columns:
            has_seq = "SEQ" in source_table.columns
            has_seq_index = "SEQ_index" in source_table.columns
            if sequence_valid_by_seq is not None and (has_seq or has_seq_index):
                export_columns.append("sequence_valid")
        if parquet_name == "02_master_index_grid.parquet":
            export_columns.append("period_s")
        export_columns.append("run_id")
        print(f"- {parquet_name}: columns = {export_columns}")

    print(f"\n[DATA EXPORT COMPLETE] -> {data_export_root.resolve()}")
    return data_export_root
