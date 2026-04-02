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

PARQUET_EXPORT_RENAME = {
    "01_participants.parquet": "participant.parquet",
    "02_master_index_grid.parquet": "master_index_grid.parquet",
    "02_emg_compact.parquet": "preprocessed_emg.parquet",
    "02_torque_compact.parquet": "preprocessed_torque.parquet",
    "05a_bia2_freqs_hz.parquet": "preprocessed_bia2_freqs_hz.parquet",
    "05a_bia4_freqs_hz.parquet": "preprocessed_bia4_freqs_hz.parquet",
    "05b_bia2_compact_aligned.parquet": "preprocessed_bia2.parquet",
    "05b_bia4_compact_aligned.parquet": "preprocessed_bia4.parquet",
    "06b_nirs_compact_aligned.parquet": "preprocessed_nirs.parquet",
    "07_myoton_compact.parquet": "preprocessed_myoton.parquet",
}

SEQUENCE_VALID_TARGETS = [
    "emg6",
    "emg8",
    "emg10",
    "torque",
    "bia",
    "nirs",
    "myoton",
]

EMG_SEQUENCE_VALID_COLUMNS = {
    "emg6": "sequence_valid_emg6",
    "emg8": "sequence_valid_emg8",
    "emg10": "sequence_valid_emg10",
}

PARQUET_SEQUENCE_VALID_TARGET = {
    "02_torque_compact.parquet": "torque",
    "05b_bia2_compact_aligned.parquet": "bia",
    "05b_bia4_compact_aligned.parquet": "bia",
    "06b_nirs_compact_aligned.parquet": "nirs",
    "07_myoton_compact.parquet": "myoton",
}


def build_master_seq_lookup(master_index_grid: pd.DataFrame) -> tuple[dict[int, str], list[str]]:
    seq_pairs = master_index_grid[["SEQ", "SEQ_index"]].drop_duplicates().copy()
    seq_pairs["SEQ"] = seq_pairs["SEQ"].astype(str)
    seq_pairs["SEQ_index"] = seq_pairs["SEQ_index"].astype(int)

    seq_pairs = seq_pairs.sort_values("SEQ_index", kind="mergesort").reset_index(drop=True)
    seq_index_to_seq = dict(zip(seq_pairs["SEQ_index"], seq_pairs["SEQ"]))
    seq_values = list(seq_pairs["SEQ"])
    return seq_index_to_seq, seq_values


def build_sequence_valid_maps(
    *,
    ctx: dict,
    sequence_valid_overrides: dict | None = None,
) -> dict[str, dict[str, int]]:
    master_index_grid = ctx["master_index_grid"]
    _, seq_values = build_master_seq_lookup(master_index_grid)

    sequence_valid_by_target = {
        target: {seq: 1 for seq in seq_values} for target in SEQUENCE_VALID_TARGETS
    }

    if sequence_valid_overrides is None:
        return sequence_valid_by_target

    for target_key, overrides_for_target in sequence_valid_overrides.items():
        target = str(target_key)
        if target not in sequence_valid_by_target:
            raise AssertionError(f"Unknown sequence_valid target: {target!r}")

        for seq_key, raw_value in overrides_for_target.items():
            seq = str(seq_key)
            sequence_valid_by_target[target][seq] = int(raw_value)

    return sequence_valid_by_target


def resolve_seq_series(
    *,
    table: pd.DataFrame,
    parquet_name: str,
    seq_index_to_seq: dict[int, str],
) -> pd.Series | None:
    has_seq = "SEQ" in table.columns
    has_seq_index = "SEQ_index" in table.columns
    if not has_seq and not has_seq_index:
        return None

    if has_seq:
        return pd.Series(table["SEQ"]).astype(str)

    seq_index_series = pd.Series(table["SEQ_index"]).astype(int)
    seq_from_index = seq_index_series.map(seq_index_to_seq)
    if seq_from_index.isna().any():
        bad_seq_index = int(seq_index_series[seq_from_index.isna()].iloc[0])
        raise AssertionError(
            f"Cannot resolve SEQ from SEQ_index in export table {parquet_name}: {bad_seq_index}"
        )
    return seq_from_index


def append_sequence_valid_columns(
    *,
    table: pd.DataFrame,
    parquet_name: str,
    sequence_valid_by_target: dict[str, dict[str, int]] | None,
    seq_index_to_seq: dict[int, str],
) -> pd.DataFrame:
    if sequence_valid_by_target is None:
        return table

    seq_series = resolve_seq_series(
        table=table,
        parquet_name=parquet_name,
        seq_index_to_seq=seq_index_to_seq,
    )
    if seq_series is None:
        return table

    table_out = table.copy()

    if parquet_name == "02_emg_compact.parquet":
        for target, column_name in EMG_SEQUENCE_VALID_COLUMNS.items():
            sequence_valid = seq_series.map(sequence_valid_by_target[target])
            if sequence_valid.isna().any():
                bad_seq = str(seq_series[sequence_valid.isna()].iloc[0])
                raise AssertionError(
                    f"Missing {column_name} mapping for SEQ={bad_seq!r} in {parquet_name}"
                )
            table_out[column_name] = sequence_valid.astype(int)
        return table_out

    target = PARQUET_SEQUENCE_VALID_TARGET.get(parquet_name)
    if target is None:
        return table

    sequence_valid = seq_series.map(sequence_valid_by_target[target])
    if sequence_valid.isna().any():
        bad_seq = str(seq_series[sequence_valid.isna()].iloc[0])
        raise AssertionError(f"Missing sequence_valid mapping for SEQ={bad_seq!r} in {parquet_name}")

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
    sequence_valid_by_target: dict[str, dict[str, int]] | None = None,
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
        export_name = PARQUET_EXPORT_RENAME.get(parquet_name, parquet_name)
        print(f"- {parquet_name} -> {export_name}")

    export_count = 0
    for parquet_name in parquet_names:
        source_path = cache_dir / parquet_name
        if not source_path.exists():
            raise FileNotFoundError(f"Missing export file: {source_path}")

        source_table = pd.read_parquet(source_path)
        exported_table = append_sequence_valid_columns(
            table=source_table,
            parquet_name=parquet_name,
            sequence_valid_by_target=sequence_valid_by_target,
            seq_index_to_seq=seq_index_to_seq,
        )
        added_validity_columns = [
            column_name
            for column_name in exported_table.columns
            if column_name not in source_table.columns and column_name.startswith("sequence_valid")
        ]
        period_s = None

        if parquet_name == "02_master_index_grid.parquet":
            period_s = infer_master_period_s(exported_table)
            exported_table = exported_table.copy()
            exported_table["period_s"] = period_s

        exported_table = exported_table.copy()
        exported_table["run_id"] = run_id

        export_name = PARQUET_EXPORT_RENAME.get(parquet_name, parquet_name)
        export_path = data_export_root / export_name
        exported_table.to_parquet(export_path, index=False)

        print(f"[DATA EXPORT] Wrote -> {export_path}")
        summary_parts = [
            f"RUN_ID={run_id}",
            f"run_id_added={'yes'}",
            f"sequence_valid_added={'yes' if len(added_validity_columns) != 0 else 'no'}",
            f"period_s_added={'yes' if period_s is not None else 'no'}",
        ]
        if len(added_validity_columns) != 0:
            summary_parts.append(f"sequence_valid_columns={added_validity_columns}")
        if period_s is not None:
            summary_parts.append(f"period_s={period_s:.15f}")
        print(f"[DATA EXPORT CHECK] {export_name} | " + " | ".join(summary_parts))
        export_count += 1

    print("\nExported parquet files and their columns:")
    for parquet_name in parquet_names:
        export_name = PARQUET_EXPORT_RENAME.get(parquet_name, parquet_name)
        source_path = cache_dir / parquet_name
        source_table = pd.read_parquet(source_path)
        export_columns = list(source_table.columns)
        if sequence_valid_by_target is not None:
            if parquet_name == "02_emg_compact.parquet":
                for column_name in EMG_SEQUENCE_VALID_COLUMNS.values():
                    if column_name not in export_columns:
                        export_columns.append(column_name)
            elif parquet_name in PARQUET_SEQUENCE_VALID_TARGET and "sequence_valid" not in export_columns:
                export_columns.append("sequence_valid")
        if parquet_name == "02_master_index_grid.parquet":
            export_columns.append("period_s")
        export_columns.append("run_id")
        print(f"- {export_name}: columns = {export_columns}")

    print(f"\n[DATA EXPORT COMPLETE] -> {data_export_root.resolve()}")
    return data_export_root
