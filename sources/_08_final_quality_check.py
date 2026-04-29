# _08_final_quality_check.py - FINAL QUALITY CONTROL
# Generate multimodal QC figures from cached parquet files (plot-only, no recompute).
#
# This step reads exclusively from disk — not from in-memory ctx DataFrames — so the
# QC figures reflect what was actually written to the parquet cache, not transient
# pipeline state that may differ from what is on disk.
#
# Plot modules (plot_selection argument to run_final_qc):
#   overview_emg_torque  — torque + three EMG channels, full session overview
#   bia_time             — BIA 2PT and 4PT |Z| and phase angle, time domain (4 figs)
#   bia_nyquist          — BIA 4PT Nyquist plots per protocol sequence (2×3 grid)
#   nirs_tx              — four haemoglobin metrics per NIRS transmitter (3 figs)
#   myoton               — five Myoton metrics overlaid on torque background
#
# Active sequences (SEQ_INCLUDE_SET) are highlighted in all time-domain plots with an
# orange fill + border and a short text label above the top panel.
#
# Inputs
#   ctx keys  : RUN_ID, CACHE_DIR, master_index_grid, parquet_path (dict of cache paths)
#   parameter : plot_selection — list of module names, "all" (default), or None (= all)
#               close_all      — call plt.close("all") before plotting (default True)
#   files     : ctx["parquet_path"] keys: CACHE_EMG, CACHE_TORQUE, CACHE_BIA_2_ALIGNED,
#               CACHE_BIA_4_ALIGNED, CACHE_NIRS_ALIGNED, CACHE_MYOTON
#               cache_dir / "05a_bia4_freqs_hz.parquet"
#
# Outputs
#   return value : list of matplotlib Figure objects (passed to _09_export.py)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# --- Sequence shading constants ---
# All time-domain QC panels highlight active-protocol sequences with a semi-transparent
# orange fill, a solid orange border, and a short text label above the top panel.
SEQ_FILL_ALPHA = 0.06   # fill transparency (faint so signals remain readable)
SEQ_BORDER_LW  = 1.6    # border linewidth
SEQ_LABEL_FS   = 8      # label font size

# Sequences with voluntary contractions; only these receive orange highlighting.
SEQ_INCLUDE_SET = {
    "WU", "MVC_REF", "SVC_REF", "EX_DYN",
    "MVC_RECOV_DYN", "EX_STA", "MVC_RECOV_STA",
}

# --- BIA constants ---
# Target frequencies (Hz) for the time-domain |Z| and phase panels.
# Three values span low (9.76 kHz), mid (48.8 kHz), and high (97.6 kHz) ranges.
BIA_TARGET_FREQS_HZ = [9760, 48800, 97600]

# --- Decimation constants ---
# Torque (~2148 Hz) and EMG (~2000 Hz) are oversampled for overview plots.
# NIRS is already at 10 Hz so no decimation is applied there.
TORQUE_OVERVIEW_DECIM = 100
EMG_OVERVIEW_DECIM    = 200

# --- NIRS colour map ---
# Consistent haemoglobin metric colours across all transmitter panels.
NIRS_COLOR_MAP = {
    "O2Hb":   "tab:blue",
    "HHb":    "tab:orange",
    "tHb":    "tab:green",
    "HbDiff": "tab:purple",
}

# --- Myoton marker sizes ---
MYOTON_TAP_SIZE  = 18   # individual tap dots (matplotlib s= units, roughly pixels²)
MYOTON_TEST_SIZE = 60   # per-test mean diamonds

# --- Nyquist panel layout ---
# Six SEQ panels arranged in a 2×3 grid matching axesC_nyq.ravel() order.
NYQUIST_PANEL_SEQS = (
    "MVC_REF",      "EX_DYN",        "EX_STA",
    "SVC_REF",      "MVC_RECOV_DYN", "MVC_RECOV_STA",
)

# Representative EX_DYN repetition IDs for the Nyquist panel.
# These five IDs span the 62-rep fatigue sequence from fresh (1) to exhausted (62).
EX_DYN_REP_IDS = (1, 16, 31, 46, 62)

# Five equally spaced time-point labels for the EX_STA Nyquist panel.
EX_STA_LABELS = ("t0", "t25", "t50", "t75", "t100")

# --- Legend style ---
LEGEND_KWARGS = {
    "loc": "upper left",
    "frameon": True,
    "facecolor": "white",
    "edgecolor": "black",
    "framealpha": 1.0,
}
LEGEND_KWARGS_BTM = {
    "loc": "lower left",
    "frameon": True,
    "facecolor": "white",
    "edgecolor": "black",
    "framealpha": 1.0,
}

# --- Plot selection constants ---
# These are the public plot keys accepted by run_final_qc(plot_selection=...).
# Keep them explicit here so the notebook and this module share the same vocabulary.
PLOT_KEYS_ALL = [
    "overview_emg_torque",
    "bia_time",
    "bia_nyquist",
    "nirs_tx",
    "myoton",
]

# --- Myoton constants ---
# Five exported Myoton metrics plotted against the torque background in the final QC.
MYOTON_QC_METRICS = [
    "myo_Frequency",
    "myo_Stiffness",
    "myo_Decrement",
    "myo_Relaxation",
    "myo_Creep",
]

# Sequence families used by the Nyquist panel selection logic.
NYQUIST_REFERENCE_SEQS = ("MVC_REF", "MVC_RECOV_DYN", "MVC_RECOV_STA", "SVC_REF")
NYQUIST_DYNAMIC_SEQ = "EX_DYN"
NYQUIST_STATIC_SEQ = "EX_STA"


# --- Helpers ---


def _shade_seq_boxes(ax, seq_box_list, label=True):
    """
    Draw orange fill + border for each active-sequence span in seq_box_list.

    seq_box_list entries: (start_time_index, end_time_index, seq_label).
    When label=True (default), a short text label is placed just above the span using
    the x-axis transform of ax. Pass label=False for panels where only fill + border
    are wanted (e.g. the lower panels of a shared-axis stack).
    """
    for start, end, seq_label in seq_box_list:
        ax.axvspan(
            start, end,
            facecolor="orange", alpha=SEQ_FILL_ALPHA,
            edgecolor="none", linewidth=0,
        )
        ax.axvspan(
            start, end,
            facecolor="none", edgecolor="orange",
            alpha=1.0, linewidth=SEQ_BORDER_LW,
        )
        if label:
            ax.text(
                0.5 * (start + end), 1.01, str(seq_label),
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=SEQ_LABEL_FS, clip_on=False,
            )


def _build_spectrum(picked_row, freqs_hz):
    """
    Extract R and −Xc across all sweep frequencies from one BIA4 snapshot row.

    picked_row : pandas Series (one row of a BIA4 compact DataFrame)
    freqs_hz   : 1-D integer array of measurement frequencies in Hz

    Returns (r_vec, minus_xc_vec), both float arrays shaped (len(freqs_hz),).
    Negating Xc converts from the standard complex-impedance convention to the Nyquist
    plot convention where the −Im axis points upward.
    """
    r_vec        = np.array([float(picked_row[f"bia4_R_ohm__f_{int(f)}Hz"])   for f in freqs_hz])
    minus_xc_vec = np.array([-float(picked_row[f"bia4_Xc_ohm__f_{int(f)}Hz"]) for f in freqs_hz])
    return r_vec, minus_xc_vec


# --- Plot units ---


def plot_overview_emg_torque(prep):
    """
    Four-panel overview: torque (top) + EMG channels 10, 6, 8 (shared x-axis).

    Active sequences are highlighted on all four panels; the SEQ label text appears
    only above the torque panel to avoid clutter on the lower three.
    """
    run_id            = prep["run_id"]
    torque_compact_df = prep["torque_compact_df"]
    emg_compact_df    = prep["emg_compact_df"]
    seq_box_list      = prep["seq_box_list"]

    # --- Arrays ---
    torque_ti  = torque_compact_df["time_index"].to_numpy(dtype=int)
    torque_raw = torque_compact_df["torque_raw"].to_numpy(dtype=float)
    emg_ti     = emg_compact_df["time_index"].to_numpy(dtype=int)
    emg10      = emg_compact_df["emg10"].to_numpy(dtype=float)
    emg6       = emg_compact_df["emg6"].to_numpy(dtype=float)
    emg8       = emg_compact_df["emg8"].to_numpy(dtype=float)

    # --- Figure ---
    figA, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
    ax_torque, ax_emg10, ax_emg6, ax_emg8 = axes

    ax_torque.plot(torque_ti[::TORQUE_OVERVIEW_DECIM], torque_raw[::TORQUE_OVERVIEW_DECIM],
                   linewidth=1.0, color="0.2")
    ax_torque.set_ylabel("Torque (raw)")
    ax_torque.grid(alpha=0.25)

    ax_emg10.plot(emg_ti[::EMG_OVERVIEW_DECIM], emg10[::EMG_OVERVIEW_DECIM],
                  linewidth=1.0, color="tab:blue")
    ax_emg10.set_ylabel("emg10 (V)")
    ax_emg10.grid(alpha=0.25)

    ax_emg6.plot(emg_ti[::EMG_OVERVIEW_DECIM], emg6[::EMG_OVERVIEW_DECIM],
                 linewidth=1.0, color="tab:orange")
    ax_emg6.set_ylabel("emg6 (V)")
    ax_emg6.grid(alpha=0.25)

    ax_emg8.plot(emg_ti[::EMG_OVERVIEW_DECIM], emg8[::EMG_OVERVIEW_DECIM],
                 linewidth=1.0, color="tab:green")
    ax_emg8.set_ylabel("emg8 (V)")
    ax_emg8.grid(alpha=0.25)
    ax_emg8.set_xlabel("time_index")

    # SEQ boxes: fill + border on all four panels; label only on ax_torque.
    for ax in (ax_torque, ax_emg10, ax_emg6, ax_emg8):
        _shade_seq_boxes(ax, seq_box_list, label=(ax is ax_torque))

    ax_torque.legend(
        handles=[Patch(facecolor="orange", edgecolor="orange", alpha=0.35,
                       label="Active sequence\n(include voluntary contraction)")],
        loc="upper left",
        frameon=False,
    )

    figA.suptitle(f"EMG + Torque — {run_id}", y=0.995)
    figA.tight_layout()
    return figA


def plot_bia_time_domain(prep):
    """
    Four figures: BIA 2PT |Z|, BIA 2PT phase, BIA 4PT |Z|, BIA 4PT phase.

    Each figure shows time-domain traces for BIA_TARGET_FREQS_HZ frequencies,
    with active-sequence highlighting. Returns a list of four Figure objects.
    BIA rows are sorted by time_index before plotting because the BIA sync step
    may introduce a small lag offset that leaves rows non-monotone.
    """
    run_id                  = prep["run_id"]
    bia2_compact_aligned_df = prep["bia2_compact_aligned_df"]
    bia4_compact_aligned_df = prep["bia4_compact_aligned_df"]
    seq_box_list            = prep["seq_box_list"]

    active_patch = Patch(facecolor="orange", edgecolor="orange", alpha=0.35,
                         label="Active sequence\n(include voluntary contraction)")
    figs_out = []

    # --- BIA 2PT sort order ---
    bia2_ti        = bia2_compact_aligned_df["time_index"].to_numpy(dtype=int)
    bia2_order     = np.argsort(bia2_ti)
    bia2_ti_sorted = bia2_ti[bia2_order]

    # --- BIA 4PT sort order ---
    bia4_ti        = bia4_compact_aligned_df["time_index"].to_numpy(dtype=int)
    bia4_order     = np.argsort(bia4_ti)
    bia4_ti_sorted = bia4_ti[bia4_order]

    # --- BIA 2PT |Z| ---
    figB_bia2_zmod, ax = plt.subplots(1, 1, figsize=(16, 4))
    ax.grid(alpha=0.25)
    _shade_seq_boxes(ax, seq_box_list)
    bia2_zmod_handles = []
    for freq_hz in BIA_TARGET_FREQS_HZ:
        r  = bia2_compact_aligned_df[f"bia2_R_ohm__f_{freq_hz}Hz"].to_numpy(dtype=float)[bia2_order]
        xc = bia2_compact_aligned_df[f"bia2_Xc_ohm__f_{freq_hz}Hz"].to_numpy(dtype=float)[bia2_order]
        h = ax.step(bia2_ti_sorted, np.hypot(r, xc), where="post",
                    linewidth=1.0, label=f"{freq_hz/1000:.2f} kHz")[0]
        bia2_zmod_handles.append(h)
    ax.set_ylabel("|Z| 2PT (Ohm)")
    ax.set_xlabel("time_index")
    ax.legend(handles=[active_patch] + bia2_zmod_handles, **LEGEND_KWARGS)
    figB_bia2_zmod.suptitle(f"BIA 2PT — |Z| @ 3 freqs — {run_id}", y=0.995)
    figB_bia2_zmod.tight_layout()
    figs_out.append(figB_bia2_zmod)

    # --- BIA 2PT phase ---
    figB_bia2_phase, ax = plt.subplots(1, 1, figsize=(16, 4))
    ax.grid(alpha=0.25)
    _shade_seq_boxes(ax, seq_box_list)
    bia2_phase_handles = []
    for freq_hz in BIA_TARGET_FREQS_HZ:
        r  = bia2_compact_aligned_df[f"bia2_R_ohm__f_{freq_hz}Hz"].to_numpy(dtype=float)[bia2_order]
        xc = bia2_compact_aligned_df[f"bia2_Xc_ohm__f_{freq_hz}Hz"].to_numpy(dtype=float)[bia2_order]
        h = ax.step(bia2_ti_sorted, np.degrees(np.arctan2(xc, r)), where="post",
                    linewidth=1.0, label=f"{freq_hz/1000:.2f} kHz")[0]
        bia2_phase_handles.append(h)
    ax.set_ylabel("Phase 2PT (deg)")
    ax.set_xlabel("time_index")
    ax.legend(handles=[active_patch] + bia2_phase_handles, **LEGEND_KWARGS)
    figB_bia2_phase.suptitle(f"BIA 2PT — Phase @ 3 freqs — {run_id}", y=0.995)
    figB_bia2_phase.tight_layout()
    figs_out.append(figB_bia2_phase)

    # --- BIA 4PT |Z| ---
    figB_bia4_zmod, ax = plt.subplots(1, 1, figsize=(16, 4))
    ax.grid(alpha=0.25)
    _shade_seq_boxes(ax, seq_box_list)
    bia4_zmod_handles = []
    for freq_hz in BIA_TARGET_FREQS_HZ:
        r  = bia4_compact_aligned_df[f"bia4_R_ohm__f_{freq_hz}Hz"].to_numpy(dtype=float)[bia4_order]
        xc = bia4_compact_aligned_df[f"bia4_Xc_ohm__f_{freq_hz}Hz"].to_numpy(dtype=float)[bia4_order]
        h = ax.step(bia4_ti_sorted, np.hypot(r, xc), where="post",
                    linewidth=1.0, label=f"{freq_hz/1000:.2f} kHz")[0]
        bia4_zmod_handles.append(h)
    ax.set_ylabel("|Z| 4PT (Ohm)")
    ax.set_xlabel("time_index")
    ax.legend(handles=[active_patch] + bia4_zmod_handles, **LEGEND_KWARGS)
    figB_bia4_zmod.suptitle(f"BIA 4PT — |Z| @ 3 freqs — {run_id}", y=0.995)
    figB_bia4_zmod.tight_layout()
    figs_out.append(figB_bia4_zmod)

    # --- BIA 4PT phase ---
    figB_bia4_phase, ax = plt.subplots(1, 1, figsize=(16, 4))
    ax.grid(alpha=0.25)
    _shade_seq_boxes(ax, seq_box_list)
    bia4_phase_handles = []
    for freq_hz in BIA_TARGET_FREQS_HZ:
        r  = bia4_compact_aligned_df[f"bia4_R_ohm__f_{freq_hz}Hz"].to_numpy(dtype=float)[bia4_order]
        xc = bia4_compact_aligned_df[f"bia4_Xc_ohm__f_{freq_hz}Hz"].to_numpy(dtype=float)[bia4_order]
        h = ax.step(bia4_ti_sorted, np.degrees(np.arctan2(xc, r)), where="post",
                    linewidth=1.0, label=f"{freq_hz/1000:.2f} kHz")[0]
        bia4_phase_handles.append(h)
    ax.set_ylabel("Phase 4PT (deg)")
    ax.set_xlabel("time_index")
    ax.legend(handles=[active_patch] + bia4_phase_handles, **LEGEND_KWARGS)
    figB_bia4_phase.suptitle(f"BIA 4PT — Phase @ 3 freqs — {run_id}", y=0.995)
    figB_bia4_phase.tight_layout()
    figs_out.append(figB_bia4_phase)

    return figs_out


def plot_bia4_nyquist(prep):
    """
    BIA 4PT Nyquist plots: six panels (2×3 grid), one per protocol sequence.

    Curve selection strategy differs by SEQ:
      MVC_REF / SVC_REF / MVC_RECOV_*  : middle row of each VC_count block
      EX_DYN                            : EX_DYN_REP_IDS (representative repetitions)
      EX_STA                            : 5 equally spaced rows (EX_STA_LABELS)
    """
    run_id                  = prep["run_id"]
    bia4_compact_aligned_df = prep["bia4_compact_aligned_df"]
    bia4_freqs_hz_array     = prep["bia4_freqs_hz_array"]
    seq_name_to_index       = prep["seq_name_to_index"]
    ex_dyn_rep_ids          = prep["ex_dyn_rep_ids"]
    ex_sta_labels           = prep["ex_sta_labels"]

    # --- Figure ---
    figC_nyq, axesC_nyq = plt.subplots(2, 3, figsize=(12, 7))

    for seq_name, ax_nyq in zip(NYQUIST_PANEL_SEQS, axesC_nyq.ravel()):

        # --- Slice current sequence ---
        seq_idx = int(seq_name_to_index[seq_name])
        df_seq = bia4_compact_aligned_df[
            bia4_compact_aligned_df["SEQ_index"].to_numpy(dtype=int) == seq_idx
        ]
        df_seq_vc = df_seq[df_seq["VC_count"].to_numpy(dtype=int) > 0]

        curves = []
        labels = []

        # --- Pick representative rows for this sequence family ---
        if seq_name in NYQUIST_REFERENCE_SEQS:
            # One Nyquist curve per VC: pick the middle row of each contraction block
            # so the curve captures mid-contraction impedance rather than onset/offset.
            vc_ids = sorted(set(df_seq_vc["VC_count"].to_numpy(dtype=int).tolist()))
            for vc_id in vc_ids:
                df_block = df_seq_vc[df_seq_vc["VC_count"].to_numpy(dtype=int) == vc_id]
                picked_row = df_block.iloc[df_block.shape[0] // 2]
                curves.append(_build_spectrum(picked_row, bia4_freqs_hz_array))
                labels.append(f"100% (Rep #{vc_id})" if seq_name.startswith("MVC") else f"#{vc_id}")

        elif seq_name == NYQUIST_DYNAMIC_SEQ:
            # Sample a fixed set of representative repetitions spanning the fatigue arc.
            # If the requested IDs are absent (shorter session), fall back to 5 equally
            # spaced IDs from whatever repetitions are available.
            vc_ids_present = set(df_seq_vc["VC_count"].to_numpy(dtype=int).tolist())
            rep_ids = [rid for rid in ex_dyn_rep_ids if rid in vc_ids_present]
            if not rep_ids:
                vc_ids_sorted = sorted(vc_ids_present)
                rep_ids = [
                    vc_ids_sorted[i]
                    for i in np.linspace(0, len(vc_ids_sorted) - 1, 5).astype(int)
                ]
            for vc_id in rep_ids:
                df_block = df_seq_vc[df_seq_vc["VC_count"].to_numpy(dtype=int) == vc_id]
                picked_row = df_block.iloc[df_block.shape[0] // 2]
                curves.append(_build_spectrum(picked_row, bia4_freqs_hz_array))
                labels.append(f"#{vc_id}")

        elif seq_name == NYQUIST_STATIC_SEQ:
            # Static contraction: 5 equally spaced time points across the sequence.
            # Use VC rows if available (sustained contraction), otherwise all rows.
            df_use = df_seq_vc if df_seq_vc.shape[0] > 0 else df_seq
            n_rows = df_use.shape[0]
            row_idxs = [
                0,
                int(0.25 * (n_rows - 1)),
                int(0.50 * (n_rows - 1)),
                int(0.75 * (n_rows - 1)),
                n_rows - 1,
            ]
            for row_i, label_str in zip(row_idxs, ex_sta_labels):
                curves.append(_build_spectrum(df_use.iloc[row_i], bia4_freqs_hz_array))
                labels.append(str(label_str))

        else:
            raise ValueError(f"Unexpected Nyquist SEQ name: {seq_name!r}")

        # --- Plot panel ---
        for (r_vec, minus_xc_vec), label_str in zip(curves, labels):
            ax_nyq.plot(r_vec, minus_xc_vec, linewidth=1.8, label=label_str)

        ax_nyq.set_title(seq_name)
        ax_nyq.set_xlabel("Real (Ohm)")
        ax_nyq.set_ylabel("-Imag (Ohm)")
        ax_nyq.grid(alpha=0.25)
        if len(labels) > 1:
            ax_nyq.legend(**LEGEND_KWARGS_BTM)

    figC_nyq.suptitle(f"BIA 4PT Nyquist - {run_id}", y=0.98)
    figC_nyq.tight_layout()
    return figC_nyq


def plot_nirs_by_tx(prep):
    """
    Three figures (one per NIRS transmitter): O2Hb, HHb, tHb, HbDiff on a single axis.

    NIRS data is at 10 Hz — no decimation is applied. Active sequences are highlighted
    with orange fill + border + label.
    """
    run_id                  = prep["run_id"]
    nirs_compact_aligned_df = prep["nirs_compact_aligned_df"]
    seq_box_list            = prep["seq_box_list"]

    active_patch = Patch(facecolor="orange", edgecolor="orange", alpha=0.35,
                         label="Active sequence\n(include voluntary contraction)")
    figs_out = []

    for tx_id in (1, 2, 3):
        # --- Figure + shared x-array ---
        fig_nirs_tx, ax = plt.subplots(1, 1, figsize=(16, 4))
        ax.grid(alpha=0.25)

        nirs_ti = nirs_compact_aligned_df["time_index"].to_numpy(dtype=int)

        # --- Plot the four haemoglobin metrics for this transmitter ---
        line_o2hb = ax.plot(
            nirs_ti,
            nirs_compact_aligned_df[f"nirs_o2hb_tx{tx_id}"].to_numpy(dtype=float),
            linewidth=1.0,
            color=NIRS_COLOR_MAP["O2Hb"],
            label="O2Hb",
        )[0]
        line_hhb = ax.plot(
            nirs_ti,
            nirs_compact_aligned_df[f"nirs_hhb_tx{tx_id}"].to_numpy(dtype=float),
            linewidth=1.0,
            color=NIRS_COLOR_MAP["HHb"],
            label="HHb",
        )[0]
        line_thb = ax.plot(
            nirs_ti,
            nirs_compact_aligned_df[f"nirs_thb_tx{tx_id}"].to_numpy(dtype=float),
            linewidth=1.0,
            color=NIRS_COLOR_MAP["tHb"],
            label="tHb",
        )[0]
        line_hbdiff = ax.plot(
            nirs_ti,
            nirs_compact_aligned_df[f"nirs_hbdiff_tx{tx_id}"].to_numpy(dtype=float),
            linewidth=1.0,
            color=NIRS_COLOR_MAP["HbDiff"],
            label="HbDiff",
        )[0]

        # --- Shared styling ---
        _shade_seq_boxes(ax, seq_box_list)

        ax.set_ylabel("Hb (a.u.)")
        ax.set_xlabel("time_index")
        ax.legend(
            handles=[active_patch, line_o2hb, line_hhb, line_thb, line_hbdiff],
            **LEGEND_KWARGS,
        )

        fig_nirs_tx.suptitle(f"NIRS Tx{tx_id} — O2Hb / HHb / tHb / HbDiff — {run_id}", y=0.995)
        fig_nirs_tx.tight_layout()
        figs_out.append(fig_nirs_tx)

    return figs_out


def plot_myoton_over_torque(prep):
    """
    Five panels (one per Myoton metric) overlaid on the torque trace.

    Individual tap dots (small circles) are plotted on the right y-axis alongside
    per-test mean markers (diamonds). The torque background occupies the left y-axis.
    Active sequences are highlighted with orange fill + border + label.
    """
    myoton_compact_df = prep["myoton_compact_df"]
    torque_compact_df = prep["torque_compact_df"]
    seq_box_list      = prep["seq_box_list"]

    myoton_taps_df  = myoton_compact_df[myoton_compact_df["row_type"] == "tap"]
    myoton_tests_df = myoton_compact_df[myoton_compact_df["row_type"] == "test"]

    torque_ti  = torque_compact_df["time_index"].to_numpy(dtype=int)
    torque_raw = torque_compact_df["torque_raw"].to_numpy(dtype=float)

    # Pre-extract shared x-arrays to avoid redundant .to_numpy() calls per panel.
    taps_x  = myoton_taps_df["time_index"].to_numpy(dtype=int)
    tests_x = myoton_tests_df["time_index"].to_numpy(dtype=int)

    active_patch = Patch(facecolor="orange", edgecolor="orange", alpha=0.35,
                         label="Active sequence\n(include voluntary contraction)")

    n_panels = len(MYOTON_QC_METRICS)
    fig_myoton_qc, axes = plt.subplots(n_panels, 1, figsize=(13, 3.2 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    first_ax_right = None

    for ax_left, metric_col in zip(axes, MYOTON_QC_METRICS):
        # --- Left axis: torque background + active-sequence shading ---
        ax_left.plot(torque_ti, torque_raw, color="0.85", linewidth=1, zorder=1)
        _shade_seq_boxes(ax_left, seq_box_list)
        ax_left.set_ylabel("Torque", color="0.5")
        ax_left.tick_params(axis="y", labelcolor="0.5")

        # --- Right axis: Myoton taps + per-test means ---
        ax_right = ax_left.twinx()
        if first_ax_right is None:
            first_ax_right = ax_right

        tests_y      = myoton_tests_df[metric_col].to_numpy(dtype=float)
        taps_handle  = ax_right.scatter(taps_x, myoton_taps_df[metric_col].to_numpy(dtype=float),
                                        s=MYOTON_TAP_SIZE, alpha=0.8, label="taps", zorder=20)
        tests_handle = ax_right.scatter(tests_x, tests_y, s=MYOTON_TEST_SIZE,
                                        marker="D", label="test mean", zorder=21)
        ax_right.plot(tests_x, tests_y, linewidth=1, alpha=0.35, zorder=19)

        ax_right.set_ylabel(metric_col)
        ax_right.grid(alpha=0.25)

    axes[-1].set_xlabel("time_index (master grid)")
    first_ax_right.legend(handles=[active_patch, taps_handle, tests_handle], **LEGEND_KWARGS)

    fig_myoton_qc.tight_layout()
    return fig_myoton_qc


# --- Orchestrator ---


def run_final_qc(
    ctx,
    plot_selection="all",
    close_all=True,
):
    """
    Render selected QC plots from cached parquet files and return the figure list.

    Reads all data exclusively from disk — not from in-memory ctx DataFrames — so the
    figures validate what was actually written to the parquet cache at each pipeline step.

    plot_selection : list of module keys, the string "all", or None (= all).
                     Valid keys: overview_emg_torque, bia_time, bia_nyquist, nirs_tx, myoton.
    close_all      : call plt.close("all") before generating figures (default True).

    Returns a list of matplotlib Figure objects suitable for handoff to run_export in _09_.
    """

    # --- Resolve ctx ---
    run_id            = ctx["RUN_ID"]
    cache_dir         = ctx["CACHE_DIR"]
    master_index_grid = ctx["master_index_grid"]

    # --- Plot selection ---
    plot_keys = (
        PLOT_KEYS_ALL
        if (plot_selection is None or plot_selection == "all")
        else [str(x) for x in plot_selection]
    )

    if close_all:
        plt.close("all")

    # --- Load cache files ---
    # All data read from disk rather than ctx to validate what was actually saved.
    paths = ctx["parquet_path"]

    emg_compact_df          = pd.read_parquet(paths["CACHE_EMG"])
    torque_compact_df       = pd.read_parquet(paths["CACHE_TORQUE"])
    bia2_compact_aligned_df = pd.read_parquet(paths["CACHE_BIA_2_ALIGNED"])
    bia4_compact_aligned_df = pd.read_parquet(paths["CACHE_BIA_4_ALIGNED"])
    nirs_compact_aligned_df = pd.read_parquet(paths["CACHE_NIRS_ALIGNED"])
    myoton_compact_df       = pd.read_parquet(paths["CACHE_MYOTON"])

    # BIA4 full sweep frequency vector (needed for Nyquist panels).
    # Stored separately in _05a_ so the Nyquist function knows the full R/Xc column list.
    bia4_freqs_df       = pd.read_parquet(cache_dir / "05a_bia4_freqs_hz.parquet")
    bia4_freqs_hz_array = bia4_freqs_df["freq_hz"].to_numpy(dtype=float)
    bia4_freqs_hz_array = np.asarray(bia4_freqs_hz_array[np.isfinite(bia4_freqs_hz_array)], dtype=int)

    # --- Build seq_box_list ---
    # Run-length encode the SEQ label array into (start_ti, end_ti, label) tuples,
    # keeping only sequences in SEQ_INCLUDE_SET (those containing voluntary contractions).
    seq_ti     = master_index_grid["time_index"].to_numpy(dtype=int)
    seq_labels = master_index_grid["SEQ"].to_numpy()

    seq_box_list     = []
    current_label    = str(seq_labels[0])
    current_start_ti = int(seq_ti[0])

    for i in range(1, seq_ti.size):
        lab = str(seq_labels[i])
        if lab != current_label:
            if current_label in SEQ_INCLUDE_SET:
                seq_box_list.append((current_start_ti, int(seq_ti[i - 1]), current_label))
            current_label    = lab
            current_start_ti = int(seq_ti[i])

    if current_label in SEQ_INCLUDE_SET:
        seq_box_list.append((current_start_ti, int(seq_ti[-1]), current_label))

    # --- Build seq_name_to_index ---
    # Map each SEQ label to its integer SEQ_index for Nyquist panel row filtering.
    seq_name_to_index = (
        master_index_grid[["SEQ", "SEQ_index"]]
        .drop_duplicates("SEQ")
        .set_index("SEQ")["SEQ_index"]
        .astype(int)
        .to_dict()
    )

    # --- Build prep dict ---
    prep = {
        "run_id":                   run_id,
        "emg_compact_df":           emg_compact_df,
        "torque_compact_df":        torque_compact_df,
        "bia2_compact_aligned_df":  bia2_compact_aligned_df,
        "bia4_compact_aligned_df":  bia4_compact_aligned_df,
        "nirs_compact_aligned_df":  nirs_compact_aligned_df,
        "myoton_compact_df":        myoton_compact_df,
        "seq_box_list":             seq_box_list,
        "bia4_freqs_hz_array":      bia4_freqs_hz_array,
        "seq_name_to_index":        seq_name_to_index,
        "ex_dyn_rep_ids":           EX_DYN_REP_IDS,
        "ex_sta_labels":            EX_STA_LABELS,
    }

    # --- Dispatch ---
    plot_fn_by_key = {
        "overview_emg_torque": plot_overview_emg_torque,
        "bia_time":            plot_bia_time_domain,
        "bia_nyquist":         plot_bia4_nyquist,
        "nirs_tx":             plot_nirs_by_tx,
        "myoton":              plot_myoton_over_torque,
    }

    qc_figs = []
    for key in plot_keys:
        out = plot_fn_by_key[key](prep)
        if out is None:
            continue
        if isinstance(out, list):
            qc_figs.extend([f for f in out if f is not None])
        else:
            qc_figs.append(out)

    return qc_figs
