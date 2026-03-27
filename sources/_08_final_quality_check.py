# _08_final_quality_check.py - FINAL QC FIGURES
# Modular selection: overview_emg_torque / bia_time / bia_nyquist / nirs_tx / myoton


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# helpers 

def percentile_ylim_from_series(values, p_low, p_high):
    numeric = np.asarray(values, dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    assert numeric.size > 0, "Cannot compute y-limits: empty or non-finite data"
    lo = float(np.percentile(numeric, p_low))
    hi = float(np.percentile(numeric, p_high))
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    return lo, hi


def compute_vc_boxes_from_master(master_df: pd.DataFrame):
    vc_array = master_df["VC"].to_numpy()
    time_index_array = master_df["time_index"].to_numpy()
    assert vc_array.shape == time_index_array.shape, "master_index_grid VC/time_index length mismatch"

    vc_is_active = (vc_array == 1)
    vc_boxes_out = []

    currently_inside = False
    start_time_index = None

    for sample_i in range(len(vc_is_active)):
        if (not currently_inside) and vc_is_active[sample_i]:
            currently_inside = True
            start_time_index = int(time_index_array[sample_i])
        elif currently_inside and (not vc_is_active[sample_i]):
            end_time_index = int(time_index_array[sample_i - 1])
            vc_boxes_out.append((start_time_index, end_time_index))
            currently_inside = False
            start_time_index = None

    if currently_inside:
        vc_boxes_out.append((start_time_index, int(time_index_array[-1])))

    return vc_boxes_out


def _normalize_plot_selection(plot_selection):
    valid = {
        "overview_emg_torque",
        "bia_time",
        "bia_nyquist",
        "nirs_tx",
        "myoton",
    }

    if plot_selection is None or plot_selection == "all":
        return list(valid)

    assert isinstance(plot_selection, (list, tuple, set)), "plot_selection must be list/tuple/set or 'all'"
    out = [str(x) for x in plot_selection]
    for k in out:
        assert k in valid, f"Unknown plot key: {k} (valid={sorted(valid)})"
    return out


def _require_ctx(ctx, keys):
    for k in keys:
        assert k in ctx, f"ctx missing key: {k}"



# plot unit  
def plot_overview_emg_torque(prep):
    from matplotlib.patches import Patch

    run_id = prep["run_id"]
    torque_compact_df = prep["torque_compact_df"]
    emg_compact_df = prep["emg_compact_df"]
    seq_box_list = prep["seq_box_list"]

    TORQUE_OVERVIEW_DECIMATE_EVERY = int(prep["TORQUE_OVERVIEW_DECIMATE_EVERY"])
    EMG_OVERVIEW_DECIMATE_EVERY = int(prep["EMG_OVERVIEW_DECIMATE_EVERY"])

    figA, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
    ax_torque = axes[0]
    ax_emg10  = axes[1]
    ax_emg6   = axes[2]
    ax_emg8   = axes[3]

    torque_time_index_array = torque_compact_df["time_index"].to_numpy(dtype=int)
    torque_raw_array = torque_compact_df["torque_raw"].to_numpy(dtype=float)

    emg_time_index_array = emg_compact_df["time_index"].to_numpy(dtype=int)
    emg10_array = emg_compact_df["emg10"].to_numpy(dtype=float)
    emg6_array  = emg_compact_df["emg6"].to_numpy(dtype=float)
    emg8_array  = emg_compact_df["emg8"].to_numpy(dtype=float)

    # Torque
    ax_torque.plot(
        torque_time_index_array[::TORQUE_OVERVIEW_DECIMATE_EVERY],
        torque_raw_array[::TORQUE_OVERVIEW_DECIMATE_EVERY],
        linewidth=1.0,
        color="0.2",
    )
    ax_torque.set_ylabel("Torque (raw)")
    ax_torque.grid(alpha=0.25)

    # EMG panels
    ax_emg10.plot(
        emg_time_index_array[::EMG_OVERVIEW_DECIMATE_EVERY],
        emg10_array[::EMG_OVERVIEW_DECIMATE_EVERY],
        linewidth=1.0,
        color="tab:blue",
    )
    ax_emg10.set_ylabel("emg10 (V)")
    ax_emg10.grid(alpha=0.25)

    ax_emg6.plot(
        emg_time_index_array[::EMG_OVERVIEW_DECIMATE_EVERY],
        emg6_array[::EMG_OVERVIEW_DECIMATE_EVERY],
        linewidth=1.0,
        color="tab:orange",
    )
    ax_emg6.set_ylabel("emg6 (V)")
    ax_emg6.grid(alpha=0.25)

    ax_emg8.plot(
        emg_time_index_array[::EMG_OVERVIEW_DECIMATE_EVERY],
        emg8_array[::EMG_OVERVIEW_DECIMATE_EVERY],
        linewidth=1.0,
        color="tab:green",
    )
    ax_emg8.set_ylabel("emg8 (V)")
    ax_emg8.grid(alpha=0.25)
    ax_emg8.set_xlabel("time_index")

    # SEQ boxes: light fill + solid border + minimalist label on top axis
    SEQ_FILL_ALPHA = 0.06
    SEQ_BORDER_LINEWIDTH = 1.6
    SEQ_LABEL_FONTSIZE = 8

    for (seq_box_start_time_index, seq_box_end_time_index, seq_box_label) in seq_box_list:
        for ax in (ax_torque, ax_emg10, ax_emg6, ax_emg8):
            ax.axvspan(
                seq_box_start_time_index,
                seq_box_end_time_index,
                facecolor="orange",
                alpha=SEQ_FILL_ALPHA,
                edgecolor="none",
                linewidth=0,
            )
            ax.axvspan(
                seq_box_start_time_index,
                seq_box_end_time_index,
                facecolor="none",
                edgecolor="orange",
                alpha=1.0,
                linewidth=SEQ_BORDER_LINEWIDTH,
            )

        seq_box_mid_time_index = 0.5 * (float(seq_box_start_time_index) + float(seq_box_end_time_index))

        ax_torque.text(
            seq_box_mid_time_index,
            1.01,
            str(seq_box_label),
            transform=ax_torque.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=SEQ_LABEL_FONTSIZE,
            clip_on=False,
        )

    active_seq_legend_handle = Patch(
        facecolor="orange",
        edgecolor="orange",
        alpha=0.35,
        label="Active sequence\n(include voluntary contraction)",
    )

    ax_torque.legend(
        handles=[active_seq_legend_handle],
        loc="upper left",
        frameon=False,
    )

    figA.suptitle(f"EMG + Torque — {run_id}", y=0.995)
    figA.tight_layout()

    return figA


def plot_bia_time_domain(prep):
    from matplotlib.patches import Patch

    run_id = prep["run_id"]
    bia2_compact_aligned_df = prep["bia2_compact_aligned_df"]
    bia4_compact_aligned_df = prep["bia4_compact_aligned_df"]
    seq_box_list = prep["seq_box_list"]
    bia_target_freqs_hz = prep["bia_target_freqs_hz"]
    LEGEND_KWARGS = prep["LEGEND_KWARGS"]

    # Legend patch (visible in legend, not tied to shading alpha)
    active_seq_legend_handle = Patch(
        facecolor="orange",
        edgecolor="orange",
        alpha=0.35,
        label="Active sequence\n(include voluntary contraction)",
    )

    # Common SEQ box knobs
    SEQ_FILL_ALPHA = 0.06
    SEQ_BORDER_LINEWIDTH = 1.6
    SEQ_LABEL_FONTSIZE = 8

    figs_out = []


    # BIA 2PT — |Z|
    bia2_time_index_array = bia2_compact_aligned_df["time_index"].to_numpy(dtype=int)
    bia2_sort_order = np.argsort(bia2_time_index_array)
    bia2_time_index_sorted = bia2_time_index_array[bia2_sort_order]

    figB_bia2_zmod, ax_bia2_zmod = plt.subplots(1, 1, figsize=(16, 4), sharex=True)
    ax_bia2_zmod.grid(alpha=0.25)

    for (seq_box_start_time_index, seq_box_end_time_index, seq_box_label) in seq_box_list:
        ax_bia2_zmod.axvspan(
            seq_box_start_time_index,
            seq_box_end_time_index,
            facecolor="orange",
            alpha=SEQ_FILL_ALPHA,
            edgecolor="none",
            linewidth=0,
        )
        ax_bia2_zmod.axvspan(
            seq_box_start_time_index,
            seq_box_end_time_index,
            facecolor="none",
            edgecolor="orange",
            alpha=1.0,
            linewidth=SEQ_BORDER_LINEWIDTH,
        )

        seq_box_mid_time_index = 0.5 * (float(seq_box_start_time_index) + float(seq_box_end_time_index))
        ax_bia2_zmod.text(
            seq_box_mid_time_index,
            1.01,
            str(seq_box_label),
            transform=ax_bia2_zmod.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=SEQ_LABEL_FONTSIZE,
            clip_on=False,
        )

    bia2_line_handles = []
    for freq_hz in bia_target_freqs_hz:
        freq_hz = int(freq_hz)
        bia2_r_col = f"bia2_R_ohm__f_{freq_hz}Hz"
        bia2_xc_col = f"bia2_Xc_ohm__f_{freq_hz}Hz"
        assert bia2_r_col in bia2_compact_aligned_df.columns, f"BIA2 missing required col: {bia2_r_col}"
        assert bia2_xc_col in bia2_compact_aligned_df.columns, f"BIA2 missing required col: {bia2_xc_col}"

        bia2_r_sorted = bia2_compact_aligned_df[bia2_r_col].to_numpy(dtype=float)[bia2_sort_order]
        bia2_xc_sorted = bia2_compact_aligned_df[bia2_xc_col].to_numpy(dtype=float)[bia2_sort_order]
        bia2_zmod_sorted = np.sqrt(bia2_r_sorted * bia2_r_sorted + bia2_xc_sorted * bia2_xc_sorted)

        line_handle = ax_bia2_zmod.step(
            bia2_time_index_sorted,
            bia2_zmod_sorted,
            where="post",
            linewidth=1.0,
            label=f"{freq_hz/1000:.2f} kHz",
        )[0]
        bia2_line_handles.append(line_handle)

    ax_bia2_zmod.set_ylabel("|Z| 2PT (Ohm)")
    ax_bia2_zmod.set_xlabel("time_index")
    ax_bia2_zmod.legend(handles=[active_seq_legend_handle] + bia2_line_handles, **LEGEND_KWARGS)
    figB_bia2_zmod.suptitle(f"BIA 2PT — |Z| @ 3 freqs — {run_id}", y=0.995)
    figB_bia2_zmod.tight_layout()
    figs_out.append(figB_bia2_zmod)


    # BIA 2PT - PHASE ANGLE
    figB_bia2_phase, ax_bia2_phase = plt.subplots(1, 1, figsize=(16, 4), sharex=True)
    ax_bia2_phase.grid(alpha=0.25)

    for (seq_box_start_time_index, seq_box_end_time_index, seq_box_label) in seq_box_list:
        ax_bia2_phase.axvspan(
            seq_box_start_time_index,
            seq_box_end_time_index,
            facecolor="orange",
            alpha=SEQ_FILL_ALPHA,
            edgecolor="none",
            linewidth=0,
        )
        ax_bia2_phase.axvspan(
            seq_box_start_time_index,
            seq_box_end_time_index,
            facecolor="none",
            edgecolor="orange",
            alpha=1.0,
            linewidth=SEQ_BORDER_LINEWIDTH,
        )

        seq_box_mid_time_index = 0.5 * (float(seq_box_start_time_index) + float(seq_box_end_time_index))
        ax_bia2_phase.text(
            seq_box_mid_time_index,
            1.01,
            str(seq_box_label),
            transform=ax_bia2_phase.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=SEQ_LABEL_FONTSIZE,
            clip_on=False,
        )

    bia2_phase_handles = []
    for freq_hz in bia_target_freqs_hz:
        freq_hz = int(freq_hz)
        bia2_r_col = f"bia2_R_ohm__f_{freq_hz}Hz"
        bia2_xc_col = f"bia2_Xc_ohm__f_{freq_hz}Hz"
        assert bia2_r_col in bia2_compact_aligned_df.columns
        assert bia2_xc_col in bia2_compact_aligned_df.columns

        bia2_r_sorted = bia2_compact_aligned_df[bia2_r_col].to_numpy(dtype=float)[bia2_sort_order]
        bia2_xc_sorted = bia2_compact_aligned_df[bia2_xc_col].to_numpy(dtype=float)[bia2_sort_order]
        bia2_phase_deg_sorted = np.degrees(np.arctan2(bia2_xc_sorted, bia2_r_sorted))

        line_handle = ax_bia2_phase.step(
            bia2_time_index_sorted,
            bia2_phase_deg_sorted,
            where="post",
            linewidth=1.0,
            label=f"{freq_hz/1000:.2f} kHz",
        )[0]
        bia2_phase_handles.append(line_handle)

    ax_bia2_phase.set_ylabel("Phase 2PT (deg)")
    ax_bia2_phase.set_xlabel("time_index")
    ax_bia2_phase.legend(handles=[active_seq_legend_handle] + bia2_phase_handles, **LEGEND_KWARGS)
    figB_bia2_phase.suptitle(f"BIA 2PT — Phase @ 3 freqs — {run_id}", y=0.995)
    figB_bia2_phase.tight_layout()
    figs_out.append(figB_bia2_phase)


    # BIA 4PT - |Z| (modulus)
    bia4_time_index_array = bia4_compact_aligned_df["time_index"].to_numpy(dtype=int)
    bia4_sort_order = np.argsort(bia4_time_index_array)
    bia4_time_index_sorted = bia4_time_index_array[bia4_sort_order]

    figB_bia4_zmod, ax_bia4_zmod = plt.subplots(1, 1, figsize=(16, 4), sharex=True)
    ax_bia4_zmod.grid(alpha=0.25)

    for (seq_box_start_time_index, seq_box_end_time_index, seq_box_label) in seq_box_list:
        ax_bia4_zmod.axvspan(
            seq_box_start_time_index,
            seq_box_end_time_index,
            facecolor="orange",
            alpha=SEQ_FILL_ALPHA,
            edgecolor="none",
            linewidth=0,
        )
        ax_bia4_zmod.axvspan(
            seq_box_start_time_index,
            seq_box_end_time_index,
            facecolor="none",
            edgecolor="orange",
            alpha=1.0,
            linewidth=SEQ_BORDER_LINEWIDTH,
        )

        seq_box_mid_time_index = 0.5 * (float(seq_box_start_time_index) + float(seq_box_end_time_index))
        ax_bia4_zmod.text(
            seq_box_mid_time_index,
            1.01,
            str(seq_box_label),
            transform=ax_bia4_zmod.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=SEQ_LABEL_FONTSIZE,
            clip_on=False,
        )

    bia4_line_handles = []
    for freq_hz in bia_target_freqs_hz:
        freq_hz = int(freq_hz)
        bia4_r_col = f"bia4_R_ohm__f_{freq_hz}Hz"
        bia4_xc_col = f"bia4_Xc_ohm__f_{freq_hz}Hz"
        assert bia4_r_col in bia4_compact_aligned_df.columns, f"BIA4 missing required col: {bia4_r_col}"
        assert bia4_xc_col in bia4_compact_aligned_df.columns, f"BIA4 missing required col: {bia4_xc_col}"

        bia4_r_sorted = bia4_compact_aligned_df[bia4_r_col].to_numpy(dtype=float)[bia4_sort_order]
        bia4_xc_sorted = bia4_compact_aligned_df[bia4_xc_col].to_numpy(dtype=float)[bia4_sort_order]
        bia4_zmod_sorted = np.sqrt(bia4_r_sorted * bia4_r_sorted + bia4_xc_sorted * bia4_xc_sorted)

        line_handle = ax_bia4_zmod.step(
            bia4_time_index_sorted,
            bia4_zmod_sorted,
            where="post",
            linewidth=1.0,
            label=f"{freq_hz/1000:.2f} kHz",
        )[0]
        bia4_line_handles.append(line_handle)

    ax_bia4_zmod.set_ylabel("|Z| 4PT (Ohm)")
    ax_bia4_zmod.set_xlabel("time_index")
    ax_bia4_zmod.legend(handles=[active_seq_legend_handle] + bia4_line_handles, **LEGEND_KWARGS)
    figB_bia4_zmod.suptitle(f"BIA 4PT — |Z| @ 3 freqs — {run_id}", y=0.995)
    figB_bia4_zmod.tight_layout()
    figs_out.append(figB_bia4_zmod)


    # BIA 4PT - PHASE ANGLE
    figB_bia4_phase, ax_bia4_phase = plt.subplots(1, 1, figsize=(16, 4), sharex=True)
    ax_bia4_phase.grid(alpha=0.25)

    for (seq_box_start_time_index, seq_box_end_time_index, seq_box_label) in seq_box_list:
        ax_bia4_phase.axvspan(
            seq_box_start_time_index,
            seq_box_end_time_index,
            facecolor="orange",
            alpha=SEQ_FILL_ALPHA,
            edgecolor="none",
            linewidth=0,
        )
        ax_bia4_phase.axvspan(
            seq_box_start_time_index,
            seq_box_end_time_index,
            facecolor="none",
            edgecolor="orange",
            alpha=1.0,
            linewidth=SEQ_BORDER_LINEWIDTH,
        )

        seq_box_mid_time_index = 0.5 * (float(seq_box_start_time_index) + float(seq_box_end_time_index))
        ax_bia4_phase.text(
            seq_box_mid_time_index,
            1.01,
            str(seq_box_label),
            transform=ax_bia4_phase.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=SEQ_LABEL_FONTSIZE,
            clip_on=False,
        )

    bia4_phase_handles = []
    for freq_hz in bia_target_freqs_hz:
        freq_hz = int(freq_hz)
        bia4_r_col = f"bia4_R_ohm__f_{freq_hz}Hz"
        bia4_xc_col = f"bia4_Xc_ohm__f_{freq_hz}Hz"
        assert bia4_r_col in bia4_compact_aligned_df.columns, f"BIA4 missing required col: {bia4_r_col}"
        assert bia4_xc_col in bia4_compact_aligned_df.columns, f"BIA4 missing required col: {bia4_xc_col}"

        bia4_r_sorted = bia4_compact_aligned_df[bia4_r_col].to_numpy(dtype=float)[bia4_sort_order]
        bia4_xc_sorted = bia4_compact_aligned_df[bia4_xc_col].to_numpy(dtype=float)[bia4_sort_order]
        bia4_phase_deg_sorted = np.degrees(np.arctan2(bia4_xc_sorted, bia4_r_sorted))

        line_handle = ax_bia4_phase.step(
            bia4_time_index_sorted,
            bia4_phase_deg_sorted,
            where="post",
            linewidth=1.0,
            label=f"{freq_hz/1000:.2f} kHz",
        )[0]
        bia4_phase_handles.append(line_handle)

    ax_bia4_phase.set_ylabel("Phase 4PT (deg)")
    ax_bia4_phase.set_xlabel("time_index")
    ax_bia4_phase.legend(handles=[active_seq_legend_handle] + bia4_phase_handles, **LEGEND_KWARGS)
    figB_bia4_phase.suptitle(f"BIA 4PT — Phase @ 3 freqs — {run_id}", y=0.995)
    figB_bia4_phase.tight_layout()
    figs_out.append(figB_bia4_phase)

    return figs_out


def plot_bia4_nyquist(prep):
    run_id = prep["run_id"]
    bia4_compact_aligned_df = prep["bia4_compact_aligned_df"]
    bia4_freqs_hz_array = prep["bia4_freqs_hz_array"]
    seq_name_to_index = prep["seq_name_to_index"]
    ex_dyn_rep_ids = prep["ex_dyn_rep_ids"]
    ex_sta_labels = prep["ex_sta_labels"]
    LEGEND_KWARGS_BTM_LEFT = prep["LEGEND_KWARGS_BTM_LEFT"]

    nyquist_panel_seq_names = (
        "MVC_REF",
        "EX_DYN",
        "EX_STA",
        "SVC_REF",
        "MVC_RECOV_DYN",
        "MVC_RECOV_STA",
    )

    figC_nyq, axesC_nyq = plt.subplots(2, 3, figsize=(12, 7))
    nyq_axes_list = [
        axesC_nyq[0, 0],
        axesC_nyq[0, 1],
        axesC_nyq[0, 2],
        axesC_nyq[1, 0],
        axesC_nyq[1, 1],
        axesC_nyq[1, 2],
    ]

    for seq_name, ax_nyq in zip(nyquist_panel_seq_names, nyq_axes_list):

        assert seq_name in seq_name_to_index, f"SEQ name not found in master map: {seq_name}"
        seq_idx = int(seq_name_to_index[seq_name])

        seq_idx_array = bia4_compact_aligned_df["SEQ_index"].to_numpy(dtype=int)
        df_seq = bia4_compact_aligned_df[seq_idx_array == seq_idx]
        assert df_seq.shape[0] > 0, f"No BIA4 rows found for SEQ_index={seq_idx} ({seq_name})"

        vc_count_array = df_seq["VC_count"].to_numpy(dtype=int)
        df_seq_vc = df_seq[vc_count_array > 0]

        curves = []   # list of (R_vec, minus_Xc_vec)
        labels = []

        if seq_name in ("MVC_REF", "MVC_RECOV_DYN", "MVC_RECOV_STA", "SVC_REF"):
            assert df_seq_vc.shape[0] > 0, f"No VC rows for {seq_name} (need VC_count > 0)"
            vc_ids = sorted(list(set(df_seq_vc["VC_count"].to_numpy(dtype=int).tolist())))
            assert len(vc_ids) > 0

            for vc_id in vc_ids:
                df_block = df_seq_vc[df_seq_vc["VC_count"].to_numpy(dtype=int) == int(vc_id)]
                assert df_block.shape[0] > 0
                picked_row = df_block.iloc[int(df_block.shape[0] // 2)]

                r_vec = np.empty((bia4_freqs_hz_array.size,), dtype=float)
                minus_xc_vec = np.empty((bia4_freqs_hz_array.size,), dtype=float)

                for fi, freq_hz in enumerate(bia4_freqs_hz_array):
                    r_col = f"bia4_R_ohm__f_{int(freq_hz)}Hz"
                    xc_col = f"bia4_Xc_ohm__f_{int(freq_hz)}Hz"
                    r_vec[fi] = float(picked_row[r_col])
                    minus_xc_vec[fi] = -float(picked_row[xc_col])

                curves.append((r_vec, minus_xc_vec))
                if seq_name.startswith("MVC"):
                    labels.append(f"100% (Rep #{int(vc_id)})")
                else:
                    labels.append(f"#{int(vc_id)}")

        elif seq_name == "EX_DYN":
            assert df_seq_vc.shape[0] > 0, "EX_DYN Nyquist requires VC rows (VC_count > 0)"
            vc_ids_present = set(df_seq_vc["VC_count"].to_numpy(dtype=int).tolist())
            rep_ids = [rid for rid in ex_dyn_rep_ids if int(rid) in vc_ids_present]

            if len(rep_ids) == 0:
                vc_ids_sorted = sorted(list(vc_ids_present))
                assert len(vc_ids_sorted) > 0
                pick_idx = np.linspace(0, len(vc_ids_sorted) - 1, 5).astype(int)
                rep_ids = [vc_ids_sorted[int(i)] for i in pick_idx]

            for vc_id in rep_ids:
                df_block = df_seq_vc[df_seq_vc["VC_count"].to_numpy(dtype=int) == int(vc_id)]
                assert df_block.shape[0] > 0
                picked_row = df_block.iloc[int(df_block.shape[0] // 2)]

                r_vec = np.empty((bia4_freqs_hz_array.size,), dtype=float)
                minus_xc_vec = np.empty((bia4_freqs_hz_array.size,), dtype=float)

                for fi, freq_hz in enumerate(bia4_freqs_hz_array):
                    r_col = f"bia4_R_ohm__f_{int(freq_hz)}Hz"
                    xc_col = f"bia4_Xc_ohm__f_{int(freq_hz)}Hz"
                    r_vec[fi] = float(picked_row[r_col])
                    minus_xc_vec[fi] = -float(picked_row[xc_col])

                curves.append((r_vec, minus_xc_vec))
                labels.append(f"#{int(vc_id)}")

        elif seq_name == "EX_STA":
            df_use = df_seq_vc if df_seq_vc.shape[0] > 0 else df_seq
            n_rows = int(df_use.shape[0])
            assert n_rows > 0

            row_idxs = [
                0,
                int(0.25 * (n_rows - 1)),
                int(0.50 * (n_rows - 1)),
                int(0.75 * (n_rows - 1)),
                int(n_rows - 1),
            ]

            for row_i, label_str in zip(row_idxs, ex_sta_labels):
                picked_row = df_use.iloc[int(row_i)]

                r_vec = np.empty((bia4_freqs_hz_array.size,), dtype=float)
                minus_xc_vec = np.empty((bia4_freqs_hz_array.size,), dtype=float)

                for fi, freq_hz in enumerate(bia4_freqs_hz_array):
                    r_col = f"bia4_R_ohm__f_{int(freq_hz)}Hz"
                    xc_col = f"bia4_Xc_ohm__f_{int(freq_hz)}Hz"
                    r_vec[fi] = float(picked_row[r_col])
                    minus_xc_vec[fi] = -float(picked_row[xc_col])

                curves.append((r_vec, minus_xc_vec))
                labels.append(str(label_str))

        else:
            raise AssertionError(f"Unexpected Nyquist SEQ name: {seq_name}")

        # plot panel
        for (r_vec, minus_xc_vec), label_str in zip(curves, labels):
            ax_nyq.plot(r_vec, minus_xc_vec, linewidth=1.8, label=str(label_str))

        ax_nyq.set_title(seq_name)
        ax_nyq.set_xlabel("Real (Ohm)")
        ax_nyq.set_ylabel("-Imag (Ohm)")
        ax_nyq.grid(alpha=0.25)

        if len(labels) > 1:
            ax_nyq.legend(**LEGEND_KWARGS_BTM_LEFT)

    figC_nyq.suptitle(f"BIA 4PT Nyquist - {run_id}", y=0.98)
    figC_nyq.tight_layout()
    return figC_nyq


def plot_nirs_by_tx(prep):
    from matplotlib.patches import Patch

    run_id = prep["run_id"]
    nirs_compact_aligned_df = prep["nirs_compact_aligned_df"]
    seq_box_list = prep["seq_box_list"]
    LEGEND_KWARGS = prep["LEGEND_KWARGS"]

    # explicit decimation knob (keep legacy default)
    NIRS_OVERVIEW_DECIMATE_EVERY = int(prep.get("NIRS_OVERVIEW_DECIMATE_EVERY", 1))
    assert NIRS_OVERVIEW_DECIMATE_EVERY >= 1

    # active sequence legend handle
    active_seq_legend_handle = Patch(
        facecolor="orange",
        edgecolor="orange",
        alpha=0.35,
        label="Active sequence\n(include voluntary contraction)",
    )

    # explicit color map (legacy style)
    NIRS_COLOR_MAP = {
        "O2Hb": "tab:blue",
        "HHb": "tab:orange",
        "tHb": "tab:green",
        "HbDiff": "tab:purple",
    }

    SEQ_FILL_ALPHA = 0.06
    SEQ_BORDER_LINEWIDTH = 1.6
    SEQ_LABEL_FONTSIZE = 8

    figs_out = []

    for tx_id in (1, 2, 3):

        fig_nirs_tx, ax_nirs_tx = plt.subplots(1, 1, figsize=(16, 4))
        ax_nirs_tx.grid(alpha=0.25)

        nirs_time_index_array = nirs_compact_aligned_df["time_index"].to_numpy(dtype=int)

        nirs_o2hb_col = f"nirs_o2hb_tx{tx_id}"
        nirs_hhb_col = f"nirs_hhb_tx{tx_id}"
        nirs_thb_col = f"nirs_thb_tx{tx_id}"
        nirs_hbdiff_col = f"nirs_hbdiff_tx{tx_id}"

        assert nirs_o2hb_col in nirs_compact_aligned_df.columns
        assert nirs_hhb_col in nirs_compact_aligned_df.columns
        assert nirs_thb_col in nirs_compact_aligned_df.columns
        assert nirs_hbdiff_col in nirs_compact_aligned_df.columns

        nirs_o2hb_array = nirs_compact_aligned_df[nirs_o2hb_col].to_numpy(dtype=float)
        nirs_hhb_array = nirs_compact_aligned_df[nirs_hhb_col].to_numpy(dtype=float)
        nirs_thb_array = nirs_compact_aligned_df[nirs_thb_col].to_numpy(dtype=float)
        nirs_hbdiff_array = nirs_compact_aligned_df[nirs_hbdiff_col].to_numpy(dtype=float)

        # plot signals
        line_o2hb = ax_nirs_tx.plot(
            nirs_time_index_array[::NIRS_OVERVIEW_DECIMATE_EVERY],
            nirs_o2hb_array[::NIRS_OVERVIEW_DECIMATE_EVERY],
            linewidth=1.0,
            color=NIRS_COLOR_MAP["O2Hb"],
            label="O2Hb",
        )[0]

        line_hhb = ax_nirs_tx.plot(
            nirs_time_index_array[::NIRS_OVERVIEW_DECIMATE_EVERY],
            nirs_hhb_array[::NIRS_OVERVIEW_DECIMATE_EVERY],
            linewidth=1.0,
            color=NIRS_COLOR_MAP["HHb"],
            label="HHb",
        )[0]

        line_thb = ax_nirs_tx.plot(
            nirs_time_index_array[::NIRS_OVERVIEW_DECIMATE_EVERY],
            nirs_thb_array[::NIRS_OVERVIEW_DECIMATE_EVERY],
            linewidth=1.0,
            color=NIRS_COLOR_MAP["tHb"],
            label="tHb",
        )[0]

        line_hbdiff = ax_nirs_tx.plot(
            nirs_time_index_array[::NIRS_OVERVIEW_DECIMATE_EVERY],
            nirs_hbdiff_array[::NIRS_OVERVIEW_DECIMATE_EVERY],
            linewidth=1.0,
            color=NIRS_COLOR_MAP["HbDiff"],
            label="HbDiff",
        )[0]

        # shading + labels (active sequences only)
        for (seq_box_start_time_index, seq_box_end_time_index, seq_box_label) in seq_box_list:

            ax_nirs_tx.axvspan(
                seq_box_start_time_index,
                seq_box_end_time_index,
                facecolor="orange",
                alpha=SEQ_FILL_ALPHA,
                edgecolor="none",
                linewidth=0,
            )

            ax_nirs_tx.axvspan(
                seq_box_start_time_index,
                seq_box_end_time_index,
                facecolor="none",
                edgecolor="orange",
                alpha=1.0,
                linewidth=SEQ_BORDER_LINEWIDTH,
            )

            seq_box_mid_time_index = 0.5 * (
                float(seq_box_start_time_index) + float(seq_box_end_time_index)
            )

            ax_nirs_tx.text(
                seq_box_mid_time_index,
                1.01,
                str(seq_box_label),
                transform=ax_nirs_tx.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=SEQ_LABEL_FONTSIZE,
                clip_on=False,
            )

        ax_nirs_tx.set_ylabel("Hb (a.u.)")
        ax_nirs_tx.set_xlabel("time_index")

        ax_nirs_tx.legend(
            handles=[active_seq_legend_handle, line_o2hb, line_hhb, line_thb, line_hbdiff],
            **LEGEND_KWARGS,
        )

        fig_nirs_tx.suptitle(
            f"NIRS Tx{tx_id} — O2Hb / HHb / tHb / HbDiff — {run_id}",
            y=0.995,
        )

        fig_nirs_tx.tight_layout()
        figs_out.append(fig_nirs_tx)

    return figs_out


def plot_myoton_over_torque(prep):
    from matplotlib.patches import Patch

    myoton_compact_df = prep["myoton_compact_df"]
    torque_compact_df = prep["torque_compact_df"]
    seq_box_list = prep["seq_box_list"]
    LEGEND_KWARGS = prep["LEGEND_KWARGS"]

    # reuse same seq knobs as other figs (keep legacy)
    SEQ_FILL_ALPHA = 0.06
    SEQ_BORDER_LINEWIDTH = 1.6
    SEQ_LABEL_FONTSIZE = 8


    # SETTINGS
    MYOTON_QC_TAP_SIZE = 18
    MYOTON_QC_TEST_SIZE = 60

    MYOTON_QC_METRICS = [
        "myo_Frequency",
        "myo_Stiffness",
        "myo_Decrement",
        "myo_Relaxation",
        "myo_Creep",
    ]


    # DATA
    myoton_taps_df = myoton_compact_df[myoton_compact_df["row_type"] == "tap"]
    myoton_tests_df = myoton_compact_df[myoton_compact_df["row_type"] == "test"]

    torque_time_index = torque_compact_df["time_index"].to_numpy(dtype=int)
    torque_background = torque_compact_df["torque_raw"].to_numpy(dtype=float)


    # LEGEND HANDLE
    active_seq_legend_handle = Patch(
        facecolor="orange",
        edgecolor="orange",
        alpha=0.35,
        label="Active sequence\n(include voluntary contraction)",
    )


    # FIGURE
    n_panels = len(MYOTON_QC_METRICS)

    fig_myoton_qc, axes_myoton_qc = plt.subplots(
        n_panels,
        1,
        figsize=(13, 3.2 * n_panels),
        sharex=True,
    )

    if n_panels == 1:
        axes_myoton_qc = [axes_myoton_qc]

    legend_axis_right = None
    taps_handle = None
    tests_handle = None

    for ax_left, metric_col in zip(axes_myoton_qc, MYOTON_QC_METRICS):

        ax_left.plot(
            torque_time_index,
            torque_background,
            color="0.85",
            linewidth=1,
            zorder=1,
        )

        for seq_start, seq_end, seq_label in seq_box_list:

            ax_left.axvspan(
                seq_start,
                seq_end,
                facecolor="orange",
                alpha=SEQ_FILL_ALPHA,
                edgecolor="none",
                linewidth=0,
                zorder=5,
            )

            ax_left.axvspan(
                seq_start,
                seq_end,
                facecolor="none",
                edgecolor="orange",
                linewidth=SEQ_BORDER_LINEWIDTH,
                zorder=6,
            )

            ax_left.text(
                0.5 * (seq_start + seq_end),
                1.01,
                str(seq_label),
                transform=ax_left.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=SEQ_LABEL_FONTSIZE,
                clip_on=False,
            )

        ax_left.set_ylabel("Torque", color="0.5")
        ax_left.tick_params(axis="y", labelcolor="0.5")

        ax_right = ax_left.twinx()
        if legend_axis_right is None:
            legend_axis_right = ax_right

        taps_handle = ax_right.scatter(
            myoton_taps_df["time_index"].to_numpy(dtype=int),
            myoton_taps_df[metric_col].to_numpy(dtype=float),
            s=MYOTON_QC_TAP_SIZE,
            alpha=0.8,
            label="taps",
            zorder=20,
        )

        tests_handle = ax_right.scatter(
            myoton_tests_df["time_index"].to_numpy(dtype=int),
            myoton_tests_df[metric_col].to_numpy(dtype=float),
            s=MYOTON_QC_TEST_SIZE,
            marker="D",
            label="test mean",
            zorder=21,
        )

        ax_right.plot(
            myoton_tests_df["time_index"].to_numpy(dtype=int),
            myoton_tests_df[metric_col].to_numpy(dtype=float),
            linewidth=1,
            alpha=0.35,
            zorder=19,
        )

        ax_right.set_ylabel(metric_col)
        ax_right.grid(alpha=0.25)

    axes_myoton_qc[-1].set_xlabel("time_index (master grid)")

    assert legend_axis_right is not None
    legend_axis_right.legend(
        handles=[active_seq_legend_handle, taps_handle, tests_handle],
        **LEGEND_KWARGS,
    )

    fig_myoton_qc.tight_layout()
    return fig_myoton_qc



# orchestrator
def run_final_qc(
    ctx,
    plot_selection="all",
    close_all=True,
):
    """
    PLOT ONLY (no recompute). Reads cached compact tables if not provided in ctx.

    Required ctx:
      - RUN_ID (str)
      - CACHE_DIR (Path)
      - master_index_grid (DataFrame)

    Optional ctx (depends of the plot selection by user):
      - emg_compact_df, torque_compact_df
      - bia2_compact_df (freq only, no time serie), bia4_compact_df (aligned)
      - nirs_compact_df (aligned)
      - myoton_compact_df

    """

    _require_ctx(ctx, ["RUN_ID", "CACHE_DIR", "master_index_grid"])
    run_id = ctx["RUN_ID"]
    cache_dir = ctx["CACHE_DIR"]
    master_index_grid = ctx["master_index_grid"]

    assert isinstance(run_id, str) and run_id
    assert isinstance(cache_dir, Path)
    assert isinstance(master_index_grid, pd.DataFrame)

    # master files
    required_master_cols = ["time_index", "time_ref_s", "SEQ", "SEQ_index", "VC", "VC_count"]
    for c in required_master_cols:
        assert c in master_index_grid.columns, f"master_index_grid missing required col: {c}"

    # plotting knobs (constants)
    bia_target_freqs_hz = [9760, 48800, 97600]

    nirs_tx_list = [1, 2, 3]
    nirs_metric_list = ["O2Hb", "HHb", "tHb", "HbDiff"]
    nirs_metric_key_map = {"O2Hb": "o2hb", "HHb": "hhb", "tHb": "thb", "HbDiff": "hbdiff"}

    vc_box_color = "orange"
    vc_box_fill_alpha = 0.12
    vc_box_edge_alpha = 0.90
    vc_box_linewidth = 2.0

    ylow_percentile = 2.0
    yhigh_percentile = 98.0

    PLOT_DECIMATE_EVERY = 8 #can be overriden by specific plot knobs below. bia & nirs already low hz, don't go too high. 
    TORQUE_OVERVIEW_DECIMATE_EVERY = 100
    EMG_OVERVIEW_DECIMATE_EVERY = 200

    # selection
    plot_keys = _normalize_plot_selection(plot_selection)

    if close_all:
        plt.close("all")


    # Load required compact tables (ONLY FROM THE DISK, NO RAM VARIABLES))
    paths = ctx["parquet_path"]

    required_keys = [
        "CACHE_EMG",
        "CACHE_TORQUE",
        "CACHE_BIA_2_ALIGNED",
        "CACHE_BIA_4_ALIGNED",
        "CACHE_NIRS_ALIGNED",
        "CACHE_MYOTON",
    ]

    for k in required_keys:
        assert k in paths, f"Missing parquet path in CTX['parquet_path']: {k}"
        assert paths[k].exists(), f"Missing cache file for {k}: {paths[k]}"

    # EMG
    emg_compact_df = pd.read_parquet(paths["CACHE_EMG"])

    # Torque
    torque_compact_df = pd.read_parquet(paths["CACHE_TORQUE"])

    # BIA2 aligned
    bia2_compact_aligned_df = pd.read_parquet(paths["CACHE_BIA_2_ALIGNED"])

    # BIA4 aligned
    bia4_compact_aligned_df = pd.read_parquet(paths["CACHE_BIA_4_ALIGNED"])

    # NIRS aligned
    nirs_compact_aligned_df = pd.read_parquet(paths["CACHE_NIRS_ALIGNED"])

    # MYOTON compact
    myoton_compact_df = pd.read_parquet(paths["CACHE_MYOTON"])


    # columns (unchanged)
    assert "time_index" in emg_compact_df.columns
    for c in ["emg10", "emg6", "emg8"]:
        assert c in emg_compact_df.columns, f"EMG missing channel: {c}"

    assert "time_index" in torque_compact_df.columns and "torque_raw" in torque_compact_df.columns
    for df_name, df in [
        ("bia2", bia2_compact_aligned_df),
        ("bia4", bia4_compact_aligned_df),
        ("nirs", nirs_compact_aligned_df),
    ]:
        assert "time_index" in df.columns, f"{df_name} missing time_index"

    assert "time_index" in myoton_compact_df.columns
    assert "row_type" in myoton_compact_df.columns


    # Shared prep 

    vc_boxes = compute_vc_boxes_from_master(master_index_grid)

    seq_include_set = {
        "WU", "MVC_REF", "SVC_REF", "EX_DYN",
        "MVC_RECOV_DYN", "EX_STA", "MVC_RECOV_STA",
    }

    seq_time_index_array = master_index_grid["time_index"].to_numpy(dtype=int)
    seq_label_array_raw = master_index_grid["SEQ"].to_numpy()

    assert seq_time_index_array.size > 0
    assert seq_time_index_array.shape == seq_label_array_raw.shape

    seq_box_list = []
    seq_current_label = str(seq_label_array_raw[0])
    seq_start_time_index = int(seq_time_index_array[0])

    for i in range(1, int(seq_time_index_array.size)):
        lab = str(seq_label_array_raw[i])
        if lab != seq_current_label:
            if seq_current_label in seq_include_set:
                seq_box_list.append((seq_start_time_index, int(seq_time_index_array[i - 1]), seq_current_label))
            seq_current_label = lab
            seq_start_time_index = int(seq_time_index_array[i])

    if seq_current_label in seq_include_set:
        seq_box_list.append((seq_start_time_index, int(seq_time_index_array[-1]), seq_current_label))

    # BIA4 sweep freqs needed for Nyquist panels
    bia4_freqs_path = cache_dir / "05a_bia4_freqs_hz.parquet"
    assert bia4_freqs_path.exists(), f"Missing BIA4 freqs cache: {bia4_freqs_path}"
    bia4_freqs_df = pd.read_parquet(bia4_freqs_path)
    assert "freq_hz" in bia4_freqs_df.columns

    bia4_freqs_hz_array = bia4_freqs_df["freq_hz"].to_numpy(dtype=float)
    bia4_freqs_hz_array = bia4_freqs_hz_array[np.isfinite(bia4_freqs_hz_array)]
    bia4_freqs_hz_array = np.asarray(bia4_freqs_hz_array, dtype=int)
    assert bia4_freqs_hz_array.size > 0

    seq_name_index_df = master_index_grid[["SEQ", "SEQ_index"]].drop_duplicates()
    assert seq_name_index_df.shape[0] > 0

    seq_name_to_index = {}
    for _, row in seq_name_index_df.iterrows():
        k = str(row["SEQ"])
        v = int(row["SEQ_index"])
        if k not in seq_name_to_index:
            seq_name_to_index[k] = v

    assert len(seq_name_to_index) > 0

    # Validate sweep columns exist (fail loud)
    for f in bia4_freqs_hz_array:
        r_col = f"bia4_R_ohm__f_{int(f)}Hz"
        xc_col = f"bia4_Xc_ohm__f_{int(f)}Hz"
        assert r_col in bia4_compact_aligned_df.columns, f"Missing BIA4 sweep col: {r_col}"
        assert xc_col in bia4_compact_aligned_df.columns, f"Missing BIA4 sweep col: {xc_col}"

    # selection knobs used in frequency sweep panels (Nyquist)
    ex_dyn_rep_ids = (1, 16, 31, 46, 62)
    ex_sta_labels = ("t0", "t25", "t50", "t75", "t100")

    prep = {
        "run_id": run_id,
        "cache_dir": cache_dir,
        "master_index_grid": master_index_grid,
        "emg_compact_df": emg_compact_df,
        "torque_compact_df": torque_compact_df,
        "bia2_compact_aligned_df": bia2_compact_aligned_df,
        "bia4_compact_aligned_df": bia4_compact_aligned_df,
        "nirs_compact_aligned_df": nirs_compact_aligned_df,
        "myoton_compact_df": myoton_compact_df,

        "vc_boxes": vc_boxes,
        "seq_box_list": seq_box_list,

        "bia_target_freqs_hz": bia_target_freqs_hz,
        "bia4_freqs_hz_array": bia4_freqs_hz_array,

        "nirs_tx_list": nirs_tx_list,
        "nirs_metric_list": nirs_metric_list,
        "nirs_metric_key_map": nirs_metric_key_map,

        "vc_box_color": vc_box_color,
        "vc_box_fill_alpha": vc_box_fill_alpha,
        "vc_box_edge_alpha": vc_box_edge_alpha,
        "vc_box_linewidth": vc_box_linewidth,

        "ylow_percentile": ylow_percentile,
        "yhigh_percentile": yhigh_percentile,

        "PLOT_DECIMATE_EVERY": int(PLOT_DECIMATE_EVERY),
        "TORQUE_OVERVIEW_DECIMATE_EVERY": int(TORQUE_OVERVIEW_DECIMATE_EVERY),
        "EMG_OVERVIEW_DECIMATE_EVERY": int(EMG_OVERVIEW_DECIMATE_EVERY),

        "seq_name_to_index": seq_name_to_index,
        "ex_dyn_rep_ids": ex_dyn_rep_ids,
        "ex_sta_labels": ex_sta_labels,

        "LEGEND_KWARGS": {
            "loc": "upper left",
            "frameon": True,
            "facecolor": "white",
            "edgecolor": "black",
            "framealpha": 1.0,
        },
        "LEGEND_KWARGS_BTM_LEFT": {
            "loc": "lower left",
            "frameon": True,
            "facecolor": "white",
            "edgecolor": "black",
            "framealpha": 1.0,
        },
    }


    # plot selection dispatch map (plot key -> plot function)
    plot_fn_by_key = {
        "overview_emg_torque": plot_overview_emg_torque,
        "bia_time": plot_bia_time_domain,
        "bia_nyquist": plot_bia4_nyquist,
        "nirs_tx": plot_nirs_by_tx,
        "myoton": plot_myoton_over_torque,
    }


    # dispatch 
    qc_figs = []

    for key in plot_keys:
        out = plot_fn_by_key[key](prep)

        if out is None:
            continue

        if isinstance(out, list):
            qc_figs.extend([f for f in out if f is not None])
        else:
            qc_figs.append(out)

    # auto display (QC terminal step)
    from IPython.display import clear_output, display

    clear_output(wait=True)

    for fig in qc_figs:
        display(fig)

    return qc_figs