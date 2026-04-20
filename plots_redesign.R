# plots_redesign.R
# Standalone redesign sandbox — synthetic mock data, no parquet files needed.
# Branch: plot_redesign
# Run the whole file; all 22 plots print to the active device.
# ──────────────────────────────────────────────────────────────────────

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(ggplot2)
  library(scales)
})

set.seed(2025)

# ══════════════════════════════════════════════════════════════════════
# 0.  THEME & PALETTE
# ══════════════════════════════════════════════════════════════════════

CHANNELS <- c("emg6", "emg8", "emg10")
N_DYN    <- 62L
N_STA    <- 60L
TQ_MVC   <- 148.5                                         # Nm
RMS_MVC  <- c(emg6 = 1.05e-3, emg8 = 1.38e-3, emg10 = 8.2e-4)

PAL <- list(
  obs      = "#1b6ca8",   # observed RMS / blue series
  mdf      = "#d17a22",   # MDF / orange series
  expected = "#888888",   # expected-from-reference series
  ref_line = "#444444",   # reference fit line
  ref_pt   = "#aaaaaa",   # reference data points
  recov    = "#e6ab02",   # recovery star
  ci       = "#1b6ca8"    # smooth CI ribbon tint
)

theme_amstrom <- function(base_size = 12) {
  theme_bw(base_size = base_size) %+replace%
    theme(
      panel.background  = element_rect(fill = "white", colour = NA),
      plot.background   = element_rect(fill = "white", colour = NA),
      panel.grid.major  = element_line(colour = "#EBEBEB", linewidth = 0.4),
      panel.grid.minor  = element_blank(),
      panel.border      = element_blank(),
      axis.line         = element_line(colour = "#444444", linewidth = 0.45),
      axis.ticks        = element_line(colour = "#444444", linewidth = 0.4),
      axis.ticks.length = unit(3.5, "pt"),
      axis.text         = element_text(colour = "#333333", size = rel(0.85)),
      axis.title        = element_text(colour = "#111111", size = rel(0.95), face = "bold"),
      strip.background  = element_rect(fill = "#1b6ca8", colour = NA),
      strip.text        = element_text(colour = "white", face = "bold",
                                       size = rel(0.88), margin = margin(4, 4, 4, 4)),
      legend.background = element_blank(),
      legend.key        = element_blank(),
      legend.title      = element_text(face = "bold", size = rel(0.85)),
      legend.text       = element_text(size = rel(0.82)),
      legend.position   = "top",
      legend.key.size   = unit(14, "pt"),
      plot.title        = element_text(face = "bold", size = rel(1.0), hjust = 0,
                                       colour = "#111111", margin = margin(b = 3)),
      plot.subtitle     = element_text(size = rel(0.80), hjust = 0,
                                       colour = "#666666", margin = margin(b = 8)),
      plot.margin       = margin(10, 14, 10, 10),
      panel.spacing     = unit(1.0, "lines")
    )
}
theme_set(theme_amstrom())

# Gradient for fatigue progression: cool (early) → warm (late), turbo-like
grad_scale <- function(name = "Repetition") {
  scale_color_viridis_c(option = "turbo", begin = 0.1, end = 0.85, name = name)
}

# ══════════════════════════════════════════════════════════════════════
# 1.  MOCK DATA
# ══════════════════════════════════════════════════════════════════════

ref_fracs  <- c(0.30, 0.50, 0.70, 1.00)
ref_labels <- c("30%", "50%", "70%", "100%")

# ── 1a. Reference levels ─────────────────────────────────────────────
reference_level_feature_table <-
  tidyr::expand_grid(reference_level_id = 1:4, channel = CHANNELS) |>
  dplyr::mutate(
    reference_level = factor(ref_labels[reference_level_id], levels = ref_labels),
    SEQ             = dplyr::if_else(reference_level_id == 4L, "MVC_REF", "SVC_REF"),
    VC_count        = as.integer(reference_level_id),
    torque_frac     = ref_fracs[reference_level_id],
    torque_mean_Nm  = TQ_MVC * torque_frac + rnorm(n(), 0, 0.8),
    rms_mid_1s      = RMS_MVC[channel] * (0.02 + 0.98 * torque_frac) + rnorm(n(), 0, RMS_MVC[channel] * 0.015),
    mdf_mid_1s_hz   = 108 + 14 * torque_frac + rnorm(n(), 0, 2.5),
    n_samples       = 2148L
  ) |>
  dplyr::select(-torque_frac) |>
  dplyr::arrange(channel, reference_level_id)

reference_level_feature_table_norm <- reference_level_feature_table |>
  dplyr::mutate(
    rms_mvc_ref_mid1s_baseline = RMS_MVC[channel],
    rms_norm    = rms_mid_1s / rms_mvc_ref_mid1s_baseline,
    torque_norm = torque_mean_Nm / TQ_MVC
  )

# ── 1b. Calibration fits ─────────────────────────────────────────────
reference_calibration_table <- reference_level_feature_table |>
  tidyr::pivot_longer(c(rms_mid_1s, mdf_mid_1s_hz),
                      names_to = "response_variable", values_to = "rv") |>
  dplyr::group_by(channel, response_variable) |>
  dplyr::summarise(
    slope              = coef(lm(rv ~ torque_mean_Nm))[[2]],
    intercept          = coef(lm(rv ~ torque_mean_Nm))[[1]],
    r_squared          = summary(lm(rv ~ torque_mean_Nm))$r.squared,
    fit_type           = "linear",
    predictor_variable = "torque_mean_Nm",
    .groups = "drop"
  )

reference_calibration_table_norm <- reference_level_feature_table_norm |>
  dplyr::group_by(channel) |>
  dplyr::summarise(
    slope              = coef(lm(rms_norm ~ torque_norm))[[2]],
    intercept          = coef(lm(rms_norm ~ torque_norm))[[1]],
    r_squared          = summary(lm(rms_norm ~ torque_norm))$r.squared,
    response_variable  = "rms_norm",
    predictor_variable = "torque_norm",
    fit_type           = "linear",
    .groups = "drop"
  )

ref_norm_coefs <- reference_calibration_table_norm |>
  dplyr::select(channel, slope_ref = slope, intercept_ref = intercept)

ref_rms_ci_df <- reference_level_feature_table_norm |>
  dplyr::group_by(channel) |>
  dplyr::group_modify(~{
    fit <- lm(rms_norm ~ torque_norm, data = .x)

    x_grid <- tibble::tibble(
      torque_norm = seq(
        min(.x$torque_norm, na.rm = TRUE),
        max(.x$torque_norm, na.rm = TRUE),
        length.out = 100
      )
    )

    pred <- predict(
      fit,
      newdata = x_grid,
      interval = "confidence",
      level = 0.95
    )

    dplyr::bind_cols(
      x_grid,
      tibble::as_tibble(pred)
    )
  }) |>
  dplyr::ungroup()

# ── 1c. MVC level summary ────────────────────────────────────────────
mvc_level_summary <-
  tidyr::expand_grid(
    tibble::tibble(
      SEQ      = c("MVC_REF", "MVC_RECOV_DYN", "MVC_RECOV_STA"),
      tq_mult  = c(1.00, 0.71, 0.85),
      rms_mult = c(1.00, 0.77, 0.91),
      mdf_mult = c(1.00, 0.88, 0.94)
    ),
    channel = CHANNELS
  ) |>
  dplyr::mutate(
    n_reps             = 3L,
    torque_mean_Nm     = TQ_MVC * tq_mult + rnorm(n(), 0, 2.0),
    rms_mid_1s_mean    = RMS_MVC[channel] * rms_mult + rnorm(n(), 0, RMS_MVC[channel] * 0.02),
    mdf_mid_1s_hz_mean = 118 * mdf_mult + rnorm(n(), 0, 1.5),
    SEQ = factor(SEQ, levels = c("MVC_REF", "MVC_RECOV_DYN", "MVC_RECOV_STA"))
  ) |>
  dplyr::select(SEQ, channel, n_reps, torque_mean_Nm, rms_mid_1s_mean, mdf_mid_1s_hz_mean)

# ── 1d. MVC recovery tables (norm) ───────────────────────────────────
make_mvc_recov_norm <- function(seq_id, tq_mult, rms_mult) {
  tidyr::expand_grid(VC_count = 1:3L, channel = CHANNELS) |>
    dplyr::mutate(
      SEQ                        = seq_id,
      torque_mean_Nm             = TQ_MVC * tq_mult + rnorm(n(), 0, 2),
      rms_mid_1s                 = RMS_MVC[channel] * rms_mult + rnorm(n(), 0, RMS_MVC[channel] * 0.02),
      mdf_mid_1s_hz              = 118 * 0.90 + rnorm(n(), 0, 2),
      n_samples                  = 2148L,
      rms_mvc_ref_mid1s_baseline = RMS_MVC[channel],
      rms_norm                   = rms_mid_1s / rms_mvc_ref_mid1s_baseline,
      torque_norm                = torque_mean_Nm / TQ_MVC,
      emg_torque_ratio           = rms_norm / torque_norm
    )
}
mvc_recov_dyn_mid1s_feature_table_norm <- make_mvc_recov_norm("MVC_RECOV_DYN", 0.71, 0.77)
mvc_recov_sta_mid1s_feature_table_norm <- make_mvc_recov_norm("MVC_RECOV_STA", 0.85, 0.91)

# ── 1e. EX_DYN feature table (norm) ──────────────────────────────────
ch_dyn <- tibble::tibble(
  channel   = CHANNELS,
  rms0      = c(0.72, 0.78, 0.65),
  drms      = c(0.26, 0.19, 0.31),
  mdf0      = c(122,  118,  115),
  dmdf      = c(-26,  -21,  -19),
  sd_rms    = c(0.018, 0.022, 0.015),
  sd_mdf    = c(3.0,   2.5,   2.8)
)

ex_dyn_mid1s_feature_table_norm <-
  tidyr::expand_grid(VC_count = seq_len(N_DYN), channel = CHANNELS) |>
  dplyr::left_join(ch_dyn, by = "channel") |>
  dplyr::mutate(
    SEQ           = "EX_DYN",
    t             = (VC_count - 1L) / (N_DYN - 1L),
    torque_norm   = 0.87 - 0.39 * t + rnorm(n(), 0, 0.025),
    rms_norm      = rms0  + drms  * t + rnorm(n(), 0, sd_rms),
    mdf_mid_1s_hz = mdf0  + dmdf  * t + rnorm(n(), 0, sd_mdf),
    rms_mvc_ref_mid1s_baseline = RMS_MVC[channel],
    torque_mean_Nm = torque_norm * TQ_MVC,
    rms_mid_1s     = rms_norm * rms_mvc_ref_mid1s_baseline,
    n_samples      = 2148L
  ) |>
  dplyr::select(-t, -rms0, -drms, -mdf0, -dmdf, -sd_rms, -sd_mdf)

# ── 1f. EX_STA feature table (norm) ──────────────────────────────────
ex_sta_window_table <- tibble::tibble(
  window_id      = seq_len(N_STA),
  window_start_s = seq_len(N_STA) - 1L,
  window_end_s   = seq_len(N_STA),
  window_mid_s   = seq_len(N_STA) - 0.5
)

ch_sta <- tibble::tibble(
  channel = CHANNELS,
  rms0    = c(0.68, 0.74, 0.62),
  drms    = c(0.22, 0.16, 0.24),
  mdf0    = c(118,  115,  112),
  dmdf    = c(-16,  -13,  -15),
  sd_rms  = c(0.015, 0.018, 0.013),
  sd_mdf  = c(2.5, 2.0, 2.2)
)

ex_sta_mid1s_feature_table_norm <-
  tidyr::expand_grid(window_id = seq_len(N_STA), channel = CHANNELS) |>
  dplyr::left_join(ex_sta_window_table, by = "window_id") |>
  dplyr::left_join(ch_sta, by = "channel") |>
  dplyr::mutate(
    SEQ           = "EX_STA",
    t             = (window_id - 1L) / (N_STA - 1L),
    torque_norm   = 0.70 + rnorm(n(), 0, 0.022),
    rms_norm      = rms0 + drms * t + rnorm(n(), 0, sd_rms),
    mdf_mid_1s_hz = mdf0 + dmdf * t + rnorm(n(), 0, sd_mdf),
    rms_mvc_ref_mid1s_baseline = RMS_MVC[channel],
    torque_mean_Nm = torque_norm * TQ_MVC,
    rms_mid_1s     = rms_norm * rms_mvc_ref_mid1s_baseline,
    n_samples      = 2148L
  ) |>
  dplyr::select(-t, -rms0, -drms, -mdf0, -dmdf, -sd_rms, -sd_mdf)

# ── 1g. Reference comparison / residual tables ────────────────────────
ex_dyn_reference_rms_fit_table_norm <- reference_calibration_table_norm |>
  dplyr::select(channel, response_variable, predictor_variable, fit_type, slope, intercept, r_squared)

ex_dyn_reference_rms_fit_table <- reference_calibration_table |>
  dplyr::filter(response_variable == "rms_mid_1s") |>
  dplyr::select(channel, response_variable, predictor_variable, fit_type, slope, intercept, r_squared)

ex_dyn_reference_mdf_fit_table <- reference_calibration_table |>
  dplyr::filter(response_variable == "mdf_mid_1s_hz") |>
  dplyr::select(channel, response_variable, predictor_variable, fit_type, slope, intercept, r_squared)

ex_dyn_reference_rms_comparison_table_norm <-
  ex_dyn_reference_projection_table_norm <-
  ex_dyn_mid1s_feature_table_norm |>
  dplyr::select(SEQ, VC_count, channel, torque_norm, rms_norm) |>
  dplyr::left_join(ref_norm_coefs, by = "channel") |>
  dplyr::mutate(
    observed_rms_norm          = rms_norm,
    expected_rms_norm          = intercept_ref + slope_ref * torque_norm,
    residual_rms_norm          = observed_rms_norm - expected_rms_norm,
    relative_residual_rms_norm = residual_rms_norm / expected_rms_norm,
    response_variable  = "rms_norm",
    predictor_variable = "torque_norm",
    fit_type           = "linear",
    r_squared          = 0.97
  ) |>
  dplyr::select(-slope_ref, -intercept_ref, -rms_norm)

ex_dyn_reference_mdf_comparison_table <- ex_dyn_mid1s_feature_table_norm |>
  dplyr::select(SEQ, VC_count, channel, torque_mean_Nm, mdf_mid_1s_hz) |>
  dplyr::left_join(
    reference_calibration_table |>
      dplyr::filter(response_variable == "mdf_mid_1s_hz") |>
      dplyr::select(channel, slope, intercept),
    by = "channel"
  ) |>
  dplyr::mutate(
    observed_mdf_mid_1s_hz = mdf_mid_1s_hz,
    expected_mdf_mid_1s_hz = intercept + slope * torque_mean_Nm,
    residual_mdf_mid_1s_hz = observed_mdf_mid_1s_hz - expected_mdf_mid_1s_hz,
    relative_residual_mdf  = residual_mdf_mid_1s_hz / expected_mdf_mid_1s_hz,
    response_variable = "mdf_mid_1s_hz", predictor_variable = "torque_mean_Nm",
    fit_type = "linear", r_squared = 0.95
  ) |>
  dplyr::select(-slope, -intercept)

# ── 1h. EMG/torque ratio tables ───────────────────────────────────────
ex_dyn_emg_torque_ratio_table <- ex_dyn_mid1s_feature_table_norm |>
  dplyr::mutate(emg_torque_ratio = rms_norm / torque_norm) |>
  dplyr::select(SEQ, VC_count, channel, rms_norm, torque_norm, emg_torque_ratio) |>
  dplyr::left_join(ref_norm_coefs, by = "channel") |>
  dplyr::mutate(expected_ratio = (intercept_ref + slope_ref * torque_norm) / torque_norm) |>
  dplyr::select(-slope_ref, -intercept_ref)

ex_sta_emg_torque_ratio_table <- ex_sta_mid1s_feature_table_norm |>
  dplyr::mutate(emg_torque_ratio = rms_norm / torque_norm) |>
  dplyr::select(SEQ, window_id, window_mid_s, channel, rms_norm, torque_norm, emg_torque_ratio) |>
  dplyr::left_join(ref_norm_coefs, by = "channel") |>
  dplyr::mutate(expected_ratio = (intercept_ref + slope_ref * torque_norm) / torque_norm) |>
  dplyr::select(-slope_ref, -intercept_ref)

# ── 1i. Helper dfs for Group A ────────────────────────────────────────
mvc_ref_mid1s_rms_baseline_table <- tibble::tibble(
  channel = CHANNELS, rms_mvc_ref_mid1s_baseline = unname(RMS_MVC)
)
svc_df <- reference_level_feature_table |>
  dplyr::mutate(intensity = factor(reference_level, levels = ref_labels))

svc_pct_df <- reference_level_feature_table |>
  dplyr::left_join(mvc_ref_mid1s_rms_baseline_table, by = "channel") |>
  dplyr::mutate(
    torque_pct = 100 * torque_mean_Nm / TQ_MVC,
    emg_pct    = 100 * rms_mid_1s / rms_mvc_ref_mid1s_baseline,
    intensity  = factor(reference_level, levels = ref_labels)
  )

# ══════════════════════════════════════════════════════════════════════
# 2.  GROUP A — Reference calibration
# ══════════════════════════════════════════════════════════════════════

# A1. Reference MDF across intensities
p_mdf <- ggplot(svc_df, aes(x = intensity, y = mdf_mid_1s_hz, group = 1)) +
  geom_line(linewidth = 0.9, colour = PAL$ref_line) +
  geom_point(size = 3.5, colour = PAL$ref_line) +
  facet_wrap(~ channel, scales = "free_y") +
  labs(
    title = "Reference calibration — Median frequency",
    x     = "Intensity  (% MVC)",
    y     = "MDF  (Hz)"
  )
print(p_mdf)

# A2. Reference RMS across intensities
p_rms <- ggplot(svc_df, aes(x = intensity, y = rms_mid_1s, group = 1)) +
  geom_line(linewidth = 0.9, colour = PAL$ref_line) +
  geom_point(size = 3.5, colour = PAL$ref_line) +
  facet_wrap(~ channel, scales = "free_y") +
  scale_y_continuous(labels = label_scientific()) +
  labs(
    title = "Reference calibration — RMS amplitude",
    x     = "Intensity  (% MVC)",
    y     = "RMS  (a.u.)"
  )
print(p_rms)

# A3. %EMG vs %Force
p_pct_emg_force <- ggplot(svc_pct_df, aes(x = torque_pct, y = emg_pct)) +
  geom_line(aes(group = 1), linewidth = 0.9, colour = PAL$ref_line) +
  geom_point(aes(shape = intensity), size = 3.5, colour = PAL$ref_line,
             fill = "white", stroke = 1.2) +
  scale_shape_manual(values = c(21, 22, 23, 24), name = "Intensity") +
  facet_wrap(~ channel, scales = "free_y") +
  labs(
    title = "Reference calibration — %EMG vs %Force",
    x     = "% Force  (torque / MVC torque \u00d7 100)",
    y     = "% EMG  (RMS / MVC RMS \u00d7 100)"
  )
print(p_pct_emg_force)

# ══════════════════════════════════════════════════════════════════════
# 3.  GROUP B — EX_DYN vs reference fit
# ══════════════════════════════════════════════════════════════════════

ref_pts_rms_nm   <- dplyr::select(reference_level_feature_table, channel, torque_mean_Nm, rms_mid_1s)
ref_pts_rms_norm <- dplyr::select(reference_level_feature_table_norm, channel, torque_norm, rms_norm)
ref_pts_mdf_nm   <- dplyr::select(reference_level_feature_table, channel, torque_mean_Nm, mdf_mid_1s_hz)

# B1. EX_DYN RMS overlay — Nm space
ex_dyn_reference_rms_overlay_plot <- ggplot() +
  geom_abline(
    data = ex_dyn_reference_rms_fit_table,
    aes(intercept = intercept, slope = slope),
    colour = PAL$ref_line, linewidth = 1.0, alpha = 0.9
  ) +
  geom_point(data = ref_pts_rms_nm, aes(x = torque_mean_Nm, y = rms_mid_1s),
             colour = PAL$ref_pt, size = 3.5, alpha = 0.95) +
  geom_point(data = ex_dyn_mid1s_feature_table_norm,
             aes(x = torque_mean_Nm, y = rms_mid_1s, colour = VC_count),
             size = 1.8, alpha = 0.75) +
  facet_wrap(~ channel, scales = "free_y") +
  grad_scale() +
  scale_y_continuous(labels = label_scientific()) +
  labs(
    title    = "EX_DYN — RMS vs reference fit  (Nm space)",
    subtitle = "Grey: reference levels & linear fit.  Colour: repetition order (early \u2192 late).",
    x        = "Torque  (Nm)",
    y        = "RMS  (a.u.)"
  )
print(ex_dyn_reference_rms_overlay_plot)

# B2. EX_DYN RMS overlay — normalized space + recovery star
ex_dyn_reference_rms_overlay_plot_norm <- ggplot() +
  geom_ribbon(
    data = ref_rms_ci_df,
    aes(x = torque_norm, ymin = lwr, ymax = upr),
    fill = PAL$ref_line, alpha = 0.12
  ) +
  geom_line(
    data = ref_rms_ci_df,
    aes(x = torque_norm, y = fit),
    colour = PAL$ref_line, linewidth = 1.0, alpha = 0.9
  ) +
  geom_point(data = ref_pts_rms_norm, aes(x = torque_norm, y = rms_norm),
             colour = PAL$ref_pt, size = 3.5, alpha = 0.95) +
  geom_point(data = ex_dyn_reference_rms_comparison_table_norm,
             aes(x = torque_norm, y = observed_rms_norm, colour = VC_count),
             size = 1.8, alpha = 0.75) +
  geom_point(data = mvc_recov_dyn_mid1s_feature_table_norm,
             aes(x = torque_norm, y = rms_norm),
             shape = 8, size = 3.0, stroke = 1.1, colour = PAL$recov,
             inherit.aes = FALSE) +
  facet_wrap(~ channel, scales = "free_y") +
  grad_scale() +
  scale_x_continuous(labels = label_percent(accuracy = 1, scale = 100)) +
  scale_y_continuous(labels = label_percent(accuracy = 1, scale = 100)) +
  labs(
    title    = "EX_DYN — Normalized RMS vs reference fit",
    subtitle = "Grey: reference levels, fit, and 95% CI.  Colour: rep order.  * = MVC_RECOV_DYN.",
    x        = "Normalized torque  (% MVC)",
    y        = "Normalized RMS  (% MVC)"
  )
print(ex_dyn_reference_rms_overlay_plot_norm)

# B3. EX_DYN RMS residual over reps
ex_dyn_reference_residual_plot_norm <- ggplot(
  ex_dyn_reference_projection_table_norm,
  aes(x = VC_count, y = residual_rms_norm)
) +
  geom_hline(yintercept = 0, linewidth = 0.8, colour = "#bbbbbb", linetype = "dashed") +
  geom_line(colour = PAL$obs, linewidth = 0.6, alpha = 0.45) +
  geom_point(aes(colour = VC_count), size = 1.7, alpha = 0.85) +
  facet_wrap(~ channel, scales = "free_y") +
  grad_scale(name = "Repetition") +
  guides(colour = "none") +
  scale_y_continuous(labels = label_percent(accuracy = 0.1, scale = 100)) +
  labs(
    title    = "EX_DYN — Normalized RMS residual over repetitions",
    subtitle = "Residual = observed RMS\u2099\u2092\u02b3\u1d39 \u2212 expected from fixed reference fit",
    x        = "Repetition",
    y        = "Residual RMS  (% MVC)"
  )
print(ex_dyn_reference_residual_plot_norm)

# B4. EX_DYN observed vs expected RMS over reps
obs_exp_rms_long <- ex_dyn_reference_projection_table_norm |>
  dplyr::select(channel, VC_count, observed_rms_norm, expected_rms_norm) |>
  tidyr::pivot_longer(c(observed_rms_norm, expected_rms_norm),
                      names_to = "series", values_to = "val") |>
  dplyr::mutate(
    series = factor(
      dplyr::recode(series,
        observed_rms_norm = "Observed",
        expected_rms_norm = "Expected (reference)"
      ),
      levels = c("Observed", "Expected (reference)")
    )
  )

ex_dyn_reference_observed_expected_over_repetition_plot_norm <- ggplot(
  obs_exp_rms_long, aes(x = VC_count, y = val, colour = series)
) +
  geom_line(linewidth = 0.85) +
  geom_point(size = 1.5, alpha = 0.8) +
  facet_wrap(~ channel, scales = "free_y") +
  scale_colour_manual(
    values = c(Observed = PAL$obs, `Expected (reference)` = PAL$expected), name = NULL
  ) +
  scale_y_continuous(labels = label_percent(accuracy = 1, scale = 100)) +
  labs(
    title    = "EX_DYN — Observed vs expected normalized RMS",
    subtitle = "Expected: projection from fixed pre-fatigue reference fit (no re-fitting on fatigue data)",
    x        = "Repetition",
    y        = "Normalized RMS  (% MVC)"
  )
print(ex_dyn_reference_observed_expected_over_repetition_plot_norm)

# B5. EX_DYN MDF overlay — Nm space
ex_dyn_reference_mdf_overlay_plot <- ggplot() +
  geom_abline(
    data = ex_dyn_reference_mdf_fit_table,
    aes(intercept = intercept, slope = slope),
    colour = PAL$ref_line, linewidth = 1.0, alpha = 0.9
  ) +
  geom_point(data = ref_pts_mdf_nm, aes(x = torque_mean_Nm, y = mdf_mid_1s_hz),
             colour = PAL$ref_pt, size = 3.5, alpha = 0.95) +
  geom_point(data = ex_dyn_reference_mdf_comparison_table,
             aes(x = torque_mean_Nm, y = observed_mdf_mid_1s_hz, colour = VC_count),
             size = 1.8, alpha = 0.75) +
  facet_wrap(~ channel, scales = "free_y") +
  grad_scale() +
  labs(
    title    = "EX_DYN — MDF vs reference fit  (Nm space)",
    subtitle = "Grey: reference levels & linear fit.  Colour: repetition order (early \u2192 late).",
    x        = "Torque  (Nm)",
    y        = "MDF  (Hz)"
  )
print(ex_dyn_reference_mdf_overlay_plot)

# B6. EX_DYN MDF residual over reps
ex_dyn_reference_mdf_residual_plot <- ggplot(
  ex_dyn_reference_mdf_comparison_table,
  aes(x = VC_count, y = residual_mdf_mid_1s_hz)
) +
  geom_hline(yintercept = 0, linewidth = 0.8, colour = "#bbbbbb", linetype = "dashed") +
  geom_line(colour = PAL$mdf, linewidth = 0.6, alpha = 0.45) +
  geom_point(aes(colour = VC_count), size = 1.7, alpha = 0.85) +
  facet_wrap(~ channel, scales = "free_y") +
  grad_scale() +
  guides(colour = "none") +
  labs(
    title    = "EX_DYN — MDF residual over repetitions",
    subtitle = "Residual = observed MDF \u2212 expected from fixed reference fit",
    x        = "Repetition",
    y        = "Residual MDF  (Hz)"
  )
print(ex_dyn_reference_mdf_residual_plot)

# B7. EX_DYN observed vs expected MDF over reps
obs_exp_mdf_long <- ex_dyn_reference_mdf_comparison_table |>
  dplyr::select(channel, VC_count, observed_mdf_mid_1s_hz, expected_mdf_mid_1s_hz) |>
  tidyr::pivot_longer(c(observed_mdf_mid_1s_hz, expected_mdf_mid_1s_hz),
                      names_to = "series", values_to = "val") |>
  dplyr::mutate(
    series = factor(
      dplyr::recode(series,
        observed_mdf_mid_1s_hz = "Observed",
        expected_mdf_mid_1s_hz = "Expected (reference)"
      ),
      levels = c("Observed", "Expected (reference)")
    )
  )

ex_dyn_reference_mdf_observed_expected_plot <- ggplot(
  obs_exp_mdf_long, aes(x = VC_count, y = val, colour = series)
) +
  geom_line(linewidth = 0.85) +
  geom_point(size = 1.5, alpha = 0.8) +
  facet_wrap(~ channel, scales = "free_y") +
  scale_colour_manual(
    values = c(Observed = PAL$mdf, `Expected (reference)` = PAL$expected), name = NULL
  ) +
  labs(
    title    = "EX_DYN — Observed vs expected MDF",
    subtitle = "Expected: projection from fixed pre-fatigue reference fit",
    x        = "Repetition",
    y        = "MDF  (Hz)"
  )
print(ex_dyn_reference_mdf_observed_expected_plot)

# ══════════════════════════════════════════════════════════════════════
# 4.  GROUP C — MVC recovery summary
# ══════════════════════════════════════════════════════════════════════

mvc_xlabels <- c(MVC_REF = "MVC ref", MVC_RECOV_DYN = "Recov dyn", MVC_RECOV_STA = "Recov sta")

# C1. Torque
mvc_torque_plot_summary <- ggplot(
  mvc_level_summary |> dplyr::distinct(SEQ, torque_mean_Nm),
  aes(x = SEQ, y = torque_mean_Nm, group = 1)
) +
  geom_line(linewidth = 1.1, colour = PAL$ref_line) +
  geom_point(size = 4.5, colour = PAL$ref_line) +
  scale_x_discrete(labels = mvc_xlabels) +
  labs(title = "MVC — Torque across recovery checkpoints", x = NULL, y = "Torque  (Nm)")
grid::grid.newpage()
print(mvc_torque_plot_summary)

# C2. RMS
mvc_recov_plot_summary <- ggplot(
  mvc_level_summary, aes(x = SEQ, y = rms_mid_1s_mean, group = channel)
) +
  geom_line(linewidth = 0.9, colour = PAL$ref_line) +
  geom_point(size = 3, colour = PAL$ref_line) +
  facet_wrap(~ channel, scales = "free_y") +
  scale_x_discrete(labels = mvc_xlabels) +
  scale_y_continuous(labels = label_scientific()) +
  labs(title = "MVC — RMS across recovery checkpoints", x = NULL, y = "RMS  (a.u.)") +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))
grid::grid.newpage()
print(mvc_recov_plot_summary)

# C3. MDF
mvc_mdf_plot_summary <- ggplot(
  mvc_level_summary, aes(x = SEQ, y = mdf_mid_1s_hz_mean, group = channel)
) +
  geom_line(linewidth = 0.9, colour = PAL$ref_line) +
  geom_point(size = 3, colour = PAL$ref_line) +
  facet_wrap(~ channel, scales = "free_y") +
  scale_x_discrete(labels = mvc_xlabels) +
  labs(title = "MVC — MDF across recovery checkpoints", x = NULL, y = "MDF  (Hz)") +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))
grid::grid.newpage()
print(mvc_mdf_plot_summary)

# C4. Triple panel 3x3
mvc_long <- mvc_level_summary |>
  dplyr::group_by(SEQ) |>
  dplyr::mutate(torque_mean_Nm = mean(torque_mean_Nm)) |>
  dplyr::ungroup() |>
  tidyr::pivot_longer(c(torque_mean_Nm, rms_mid_1s_mean, mdf_mid_1s_hz_mean),
                      names_to = "metric", values_to = "value") |>
  dplyr::mutate(
    metric = factor(
      dplyr::recode(metric,
        torque_mean_Nm     = "Torque  (Nm)",
        rms_mid_1s_mean    = "RMS  (a.u.)",
        mdf_mid_1s_hz_mean = "MDF  (Hz)"
      ),
      levels = c("Torque  (Nm)", "RMS  (a.u.)", "MDF  (Hz)")
    )
  )

mvc_dyn_recovery_plot <- ggplot(mvc_long, aes(x = SEQ, y = value, group = channel)) +
  geom_line(linewidth = 0.9, colour = PAL$ref_line) +
  geom_point(size = 2.8, colour = PAL$ref_line) +
  facet_grid(metric ~ channel, scales = "free_y") +
  scale_x_discrete(labels = mvc_xlabels) +
  labs(title = "MVC — Recovery profile  (torque · RMS · MDF)", x = NULL, y = NULL) +
  theme(
    axis.text.x  = element_text(angle = 30, hjust = 1),
    strip.text.y = element_text(size = rel(0.78))
  )
print(mvc_dyn_recovery_plot)

# ══════════════════════════════════════════════════════════════════════
# 5.  GROUP D — EX_DYN individual trend plots
# ══════════════════════════════════════════════════════════════════════

# D1. Torque vs repetition
ex_dyn_torque_vs_rep <- ggplot(
  ex_dyn_mid1s_feature_table_norm, aes(x = VC_count, y = torque_norm, colour = VC_count)
) +
  geom_line(aes(group = channel), colour = PAL$ref_line, linewidth = 0.5, alpha = 0.35) +
  geom_point(size = 1.8, alpha = 0.8) +
  geom_smooth(method = "lm", se = TRUE, colour = PAL$obs,
              fill = PAL$ci, alpha = 0.15, linewidth = 1.0) +
  facet_wrap(~ channel) +
  grad_scale() +
  guides(colour = "none") +
  scale_y_continuous(labels = label_percent(accuracy = 1, scale = 100)) +
  labs(title = "EX_DYN — Torque fatigue trajectory",
       x = "Repetition", y = "Normalized torque  (% MVC)")
print(ex_dyn_torque_vs_rep)

# D2. RMS vs torque
ex_dyn_rms_vs_torque <- ggplot(
  ex_dyn_mid1s_feature_table_norm, aes(x = torque_norm, y = rms_norm)
) +
  geom_point(colour = PAL$obs, size = 1.8, alpha = 0.65) +
  geom_smooth(method = "lm", se = TRUE, colour = PAL$obs,
              fill = PAL$ci, alpha = 0.15, linewidth = 1.0) +
  facet_wrap(~ channel) +
  scale_x_continuous(labels = label_percent(accuracy = 1, scale = 100)) +
  scale_y_continuous(labels = label_percent(accuracy = 1, scale = 100)) +
  labs(title = "EX_DYN — RMS activation vs torque output",
       x = "Normalized torque  (% MVC)", y = "Normalized RMS  (% MVC)")
print(ex_dyn_rms_vs_torque)

# D3. MDF vs repetition
ex_dyn_mdf_vs_rep <- ggplot(
  ex_dyn_mid1s_feature_table_norm, aes(x = VC_count, y = mdf_mid_1s_hz)
) +
  geom_point(colour = PAL$mdf, size = 1.8, alpha = 0.65) +
  geom_smooth(method = "lm", se = TRUE, colour = PAL$mdf,
              fill = PAL$mdf, alpha = 0.15, linewidth = 1.0) +
  facet_wrap(~ channel) +
  labs(title = "EX_DYN — Spectral fatigue (MDF) over repetitions",
       x = "Repetition", y = "MDF  (Hz)")
print(ex_dyn_mdf_vs_rep)

# D4. MDF vs torque
ex_dyn_mdf_vs_torque <- ggplot(
  ex_dyn_mid1s_feature_table_norm, aes(x = torque_norm, y = mdf_mid_1s_hz)
) +
  geom_point(colour = PAL$mdf, size = 1.8, alpha = 0.65) +
  geom_smooth(method = "lm", se = TRUE, colour = PAL$mdf,
              fill = PAL$mdf, alpha = 0.15, linewidth = 1.0) +
  facet_wrap(~ channel) +
  scale_x_continuous(labels = label_percent(accuracy = 1, scale = 100)) +
  labs(title = "EX_DYN — Spectral fatigue (MDF) vs torque",
       x = "Normalized torque  (% MVC)", y = "MDF  (Hz)")
print(ex_dyn_mdf_vs_torque)

# D5. Dual-axis RMS + MDF vs torque (one plot per channel)
dax_params <- ex_dyn_mid1s_feature_table_norm |>
  dplyr::group_by(channel) |>
  dplyr::summarise(
    rms_min = min(rms_norm), rms_max = max(rms_norm),
    mdf_min = min(mdf_mid_1s_hz), mdf_max = max(mdf_mid_1s_hz),
    .groups = "drop"
  ) |>
  dplyr::mutate(
    sf = (rms_max - rms_min) / (mdf_max - mdf_min)
  )

dax_data <- ex_dyn_mid1s_feature_table_norm |>
  dplyr::left_join(dax_params, by = "channel") |>
  dplyr::mutate(mdf_scaled = (mdf_mid_1s_hz - mdf_min) * sf + rms_min)

ex_dyn_dual_axis_plots <- lapply(CHANNELS, function(ch) {
  df <- dplyr::filter(dax_data, channel == ch)
  p <- dax_params |> dplyr::filter(channel == ch)
  ggplot(df, aes(x = torque_norm)) +
    geom_point(aes(y = rms_norm,   colour = "RMS"), size = 2.0, alpha = 0.75) +
    geom_smooth(aes(y = rms_norm,   colour = "RMS"), method = "lm", se = FALSE, linewidth = 1.0) +
    geom_point(aes(y = mdf_scaled, colour = "MDF"), shape = 17, size = 2.0, alpha = 0.75) +
    geom_smooth(aes(y = mdf_scaled, colour = "MDF"), method = "lm", se = FALSE, linewidth = 1.0) +
    scale_y_continuous(
      name   = "Normalized RMS  (% MVC)",
      labels = label_percent(accuracy = 1, scale = 100),
      sec.axis = sec_axis(
        transform = ~ (. - p$rms_min) / p$sf + p$mdf_min,
        name = "MDF  (Hz)"
      )
    ) +
    scale_x_continuous(labels = label_percent(accuracy = 1, scale = 100)) +
    scale_colour_manual(values = c(RMS = PAL$obs, MDF = PAL$mdf), name = NULL) +
    coord_cartesian(ylim = c(p$rms_min, p$rms_max)) +
    labs(title = paste0("EX_DYN — RMS & MDF vs torque  [", ch, "]"),
         x = "Normalized torque  (% MVC)")
})
names(ex_dyn_dual_axis_plots) <- CHANNELS
for (p in ex_dyn_dual_axis_plots) print(p)

# D6. RMS vs repetition
ex_dyn_rms_vs_repetition <- ggplot(
  ex_dyn_mid1s_feature_table_norm, aes(x = VC_count, y = rms_norm)
) +
  geom_point(colour = PAL$obs, size = 1.8, alpha = 0.65) +
  geom_smooth(method = "lm", se = TRUE, colour = PAL$obs,
              fill = PAL$ci, alpha = 0.15, linewidth = 1.0) +
  facet_wrap(~ channel) +
  scale_y_continuous(labels = label_percent(accuracy = 1, scale = 100)) +
  labs(title = "EX_DYN — RMS activation over repetitions",
       x = "Repetition", y = "Normalized RMS  (% MVC)")
print(ex_dyn_rms_vs_repetition)

# D7. EMG/torque ratio vs repetition
dyn_ratio_long <- ex_dyn_emg_torque_ratio_table |>
  tidyr::pivot_longer(c(emg_torque_ratio, expected_ratio),
                      names_to = "series", values_to = "val") |>
  dplyr::mutate(
    series = factor(
      dplyr::recode(series,
        emg_torque_ratio = "Observed",
        expected_ratio   = "Expected (reference)"
      ),
      levels = c("Observed", "Expected (reference)")
    )
  )

ex_dyn_emg_torque_ratio_trend_plot <- ggplot(
  dyn_ratio_long, aes(x = VC_count, y = val, colour = series)
) +
  geom_line(linewidth = 0.85) +
  geom_point(data = dplyr::filter(dyn_ratio_long, series == "Observed"),
             size = 1.4, alpha = 0.75) +
  facet_wrap(~ channel, scales = "free_y") +
  scale_colour_manual(
    values = c(Observed = PAL$obs, `Expected (reference)` = PAL$expected), name = NULL
  ) +
  labs(
    title    = "EX_DYN — EMG/torque efficiency ratio over repetitions",
    subtitle = "Rising observed ratio \u2192 more EMG per unit torque = neuromuscular fatigue",
    x        = "Repetition",
    y        = "RMS\u2099\u2092\u02b3\u1d39 / torque\u2099\u2092\u02b3\u1d39"
  )
print(ex_dyn_emg_torque_ratio_trend_plot)

# D8. Combined fatigue triple panel — main EX_DYN summary
dyn_metric_levels <- c("Norm. torque  (% MVC)", "Norm. RMS  (% MVC)", "MDF  (Hz)")

ex_dyn_plot_long <- ex_dyn_mid1s_feature_table_norm |>
  dplyr::select(channel, VC_count, torque_norm, rms_norm, mdf_mid_1s_hz) |>
  tidyr::pivot_longer(c(torque_norm, rms_norm, mdf_mid_1s_hz),
                      names_to = "metric", values_to = "value") |>
  dplyr::mutate(
    metric = factor(
      dplyr::recode(metric,
        torque_norm   = dyn_metric_levels[[1]],
        rms_norm      = dyn_metric_levels[[2]],
        mdf_mid_1s_hz = dyn_metric_levels[[3]]
      ),
      levels = dyn_metric_levels
    )
  )

ex_dyn_recov_star_long <- mvc_recov_dyn_mid1s_feature_table_norm |>
  dplyr::group_by(channel) |>
  dplyr::summarise(across(c(torque_norm, rms_norm, mdf_mid_1s_hz), mean), .groups = "drop") |>
  tidyr::pivot_longer(c(torque_norm, rms_norm, mdf_mid_1s_hz),
                      names_to = "metric", values_to = "value") |>
  dplyr::mutate(
    metric = factor(
      dplyr::recode(metric,
        torque_norm   = dyn_metric_levels[[1]],
        rms_norm      = dyn_metric_levels[[2]],
        mdf_mid_1s_hz = dyn_metric_levels[[3]]
      ),
      levels = dyn_metric_levels
    ),
    x_star = max(ex_dyn_mid1s_feature_table_norm$VC_count) + 4L
  )

ex_dyn_combined_fatigue_plot <- ggplot(
  ex_dyn_plot_long, aes(x = VC_count, y = value, colour = VC_count)
) +
  geom_point(alpha = 0.65, size = 1.0) +
  geom_point(data = ex_dyn_recov_star_long, aes(x = x_star, y = value),
             shape = 8, size = 2.8, stroke = 1.1, colour = PAL$recov,
             inherit.aes = FALSE) +
  facet_grid(metric ~ channel, scales = "free_y") +
  grad_scale() +
  labs(
    title    = "EX_DYN — Fatigue progression summary",
    subtitle = "\u2605 = MVC_RECOV_DYN mean  (offset for readability)",
    x        = "Repetition",
    y        = NULL
  ) +
  theme(strip.text.y = element_text(size = rel(0.78)))
print(ex_dyn_combined_fatigue_plot)

# ══════════════════════════════════════════════════════════════════════
# 6.  GROUP E — EX_STA trends
# ══════════════════════════════════════════════════════════════════════

# E1. EMG/torque ratio vs time
sta_ratio_long <- ex_sta_emg_torque_ratio_table |>
  tidyr::pivot_longer(c(emg_torque_ratio, expected_ratio),
                      names_to = "series", values_to = "val") |>
  dplyr::mutate(
    series = factor(
      dplyr::recode(series,
        emg_torque_ratio = "Observed",
        expected_ratio   = "Expected (reference)"
      ),
      levels = c("Observed", "Expected (reference)")
    )
  )

ex_sta_emg_torque_ratio_trend_plot <- ggplot(
  sta_ratio_long, aes(x = window_mid_s, y = val, colour = series)
) +
  geom_line(linewidth = 0.85) +
  geom_point(data = dplyr::filter(sta_ratio_long, series == "Observed"),
             size = 1.4, alpha = 0.75) +
  facet_wrap(~ channel, scales = "free_y") +
  scale_colour_manual(
    values = c(Observed = PAL$obs, `Expected (reference)` = PAL$expected), name = NULL
  ) +
  labs(
    title    = "EX_STA — EMG/torque efficiency ratio over time",
    subtitle = "Rising observed ratio \u2192 increasing neuromuscular demand at constant torque",
    x        = "Time  (s)",
    y        = "RMS\u2099\u2092\u02b3\u1d39 / torque\u2099\u2092\u02b3\u1d39"
  )
print(ex_sta_emg_torque_ratio_trend_plot)

# E2. Combined fatigue triple panel — main EX_STA summary
sta_metric_levels <- c("Norm. torque  (% MVC)", "Norm. RMS  (% MVC)", "MDF  (Hz)")

ex_sta_plot_long <- ex_sta_mid1s_feature_table_norm |>
  dplyr::select(channel, window_id, window_mid_s, torque_norm, rms_norm, mdf_mid_1s_hz) |>
  tidyr::pivot_longer(c(torque_norm, rms_norm, mdf_mid_1s_hz),
                      names_to = "metric", values_to = "value") |>
  dplyr::mutate(
    metric = factor(
      dplyr::recode(metric,
        torque_norm   = sta_metric_levels[[1]],
        rms_norm      = sta_metric_levels[[2]],
        mdf_mid_1s_hz = sta_metric_levels[[3]]
      ),
      levels = sta_metric_levels
    )
  )

ex_sta_recov_star_long <- mvc_recov_sta_mid1s_feature_table_norm |>
  dplyr::group_by(channel) |>
  dplyr::summarise(across(c(torque_norm, rms_norm, mdf_mid_1s_hz), mean), .groups = "drop") |>
  tidyr::pivot_longer(c(torque_norm, rms_norm, mdf_mid_1s_hz),
                      names_to = "metric", values_to = "value") |>
  dplyr::mutate(
    metric = factor(
      dplyr::recode(metric,
        torque_norm   = sta_metric_levels[[1]],
        rms_norm      = sta_metric_levels[[2]],
        mdf_mid_1s_hz = sta_metric_levels[[3]]
      ),
      levels = sta_metric_levels
    ),
    x_star = max(ex_sta_plot_long$window_mid_s) + 5
  )

ex_sta_triple_panel_fatigue <- ggplot(
  ex_sta_plot_long, aes(x = window_mid_s, y = value, colour = window_mid_s)
) +
  geom_point(alpha = 0.65, size = 1.0) +
  geom_point(data = ex_sta_recov_star_long, aes(x = x_star, y = value),
             shape = 8, size = 2.8, stroke = 1.1, colour = PAL$recov,
             inherit.aes = FALSE) +
  facet_grid(metric ~ channel, scales = "free_y") +
  scale_color_viridis_c(option = "turbo", begin = 0.1, end = 0.85, name = "Time  (s)") +
  labs(
    title    = "EX_STA — Fatigue progression summary",
    subtitle = "\u2605 = MVC_RECOV_STA mean  (offset for readability)",
    x        = "Time  (s)",
    y        = NULL
  ) +
  theme(strip.text.y = element_text(size = rel(0.78)))
print(ex_sta_triple_panel_fatigue)

# ══════════════════════════════════════════════════════════════════════
# 7.  GROUP F — EMG10 recap
# ══════════════════════════════════════════════════════════════════════

target_channel <- "emg10"

emg10_ref_rms_overlay_plot <- ggplot() +
  geom_ribbon(
    data = dplyr::filter(ref_rms_ci_df, channel == target_channel),
    aes(x = torque_norm, ymin = lwr, ymax = upr),
    fill = PAL$ref_line, alpha = 0.12
  ) +
  geom_line(
    data = dplyr::filter(ref_rms_ci_df, channel == target_channel),
    aes(x = torque_norm, y = fit),
    colour = PAL$ref_line, linewidth = 1.0, alpha = 0.9
  ) +
  geom_point(
    data = dplyr::filter(reference_level_feature_table_norm, channel == target_channel),
    aes(x = torque_norm, y = rms_norm),
    colour = PAL$ref_pt, size = 3.5, alpha = 0.95
  ) +
  geom_point(
    data = dplyr::filter(ex_dyn_reference_rms_comparison_table_norm, channel == target_channel),
    aes(x = torque_norm, y = observed_rms_norm, colour = VC_count),
    size = 1.8, alpha = 0.75
  ) +
  geom_point(
    data = dplyr::filter(mvc_recov_dyn_mid1s_feature_table_norm, channel == target_channel),
    aes(x = torque_norm, y = rms_norm),
    shape = 8, size = 3.0, stroke = 1.1, colour = PAL$recov,
    inherit.aes = FALSE
  ) +
  grad_scale() +
  labs(
    title = "EMG10 recap — Reference RMS fit vs EX_DYN",
    subtitle = "Grey = reference fit, 95% CI, and points; colour = EX_DYN repetitions; * = MVC_RECOV_DYN",
    x = "Normalized torque  (% MVC)",
    y = "Normalized RMS  (% MVC)"
  ) +
  scale_x_continuous(labels = label_percent(accuracy = 1, scale = 100)) +
  scale_y_continuous(labels = label_percent(accuracy = 1, scale = 100))
print(emg10_ref_rms_overlay_plot)

emg10_dyn_ratio_plot <- ggplot(
  dplyr::filter(dyn_ratio_long, channel == target_channel),
  aes(x = VC_count, y = val, colour = series)
) +
  geom_line(linewidth = 0.85) +
  geom_point(
    data = dplyr::filter(dyn_ratio_long, channel == target_channel, series == "Observed"),
    size = 1.5, alpha = 0.8
  ) +
  scale_colour_manual(
    values = c(Observed = PAL$obs, `Expected (reference)` = PAL$expected), name = NULL
  ) +
  labs(
    title = "EMG10 recap — EX_DYN EMG/torque ratio",
    subtitle = "Observed versus expected ratio from reference calibration",
    x = "Repetition",
    y = "RMSnorm / torquenorm"
  )
print(emg10_dyn_ratio_plot)

emg10_dyn_summary_plot <- ggplot(
  dplyr::filter(ex_dyn_plot_long, channel == target_channel),
  aes(x = VC_count, y = value, colour = VC_count)
) +
  geom_point(alpha = 0.7, size = 1.2) +
  geom_point(
    data = dplyr::filter(ex_dyn_recov_star_long, channel == target_channel),
    aes(x = x_star, y = value),
    shape = 8, size = 3.0, stroke = 1.1, colour = PAL$recov,
    inherit.aes = FALSE
  ) +
  facet_wrap(~ metric, scales = "free_y", ncol = 1) +
  grad_scale() +
  labs(
    title = "EMG10 recap — EX_DYN fatigue summary",
    subtitle = "\u2605 = MVC_RECOV_DYN mean",
    x = "Repetition",
    y = NULL
  )
print(emg10_dyn_summary_plot)

emg10_sta_ratio_plot <- ggplot(
  dplyr::filter(sta_ratio_long, channel == target_channel),
  aes(x = window_mid_s, y = val, colour = series)
) +
  geom_line(linewidth = 0.85) +
  geom_point(
    data = dplyr::filter(sta_ratio_long, channel == target_channel, series == "Observed"),
    size = 1.5, alpha = 0.8
  ) +
  scale_colour_manual(
    values = c(Observed = PAL$obs, `Expected (reference)` = PAL$expected), name = NULL
  ) +
  labs(
    title = "EMG10 recap — EX_STA EMG/torque ratio",
    subtitle = "Observed versus expected ratio from reference calibration",
    x = "Time  (s)",
    y = "RMSnorm / torquenorm"
  )
print(emg10_sta_ratio_plot)

emg10_sta_summary_plot <- ggplot(
  dplyr::filter(ex_sta_plot_long, channel == target_channel),
  aes(x = window_mid_s, y = value, colour = window_mid_s)
) +
  geom_point(alpha = 0.7, size = 1.2) +
  geom_point(
    data = dplyr::filter(ex_sta_recov_star_long, channel == target_channel),
    aes(x = x_star, y = value),
    shape = 8, size = 3.0, stroke = 1.1, colour = PAL$recov,
    inherit.aes = FALSE
  ) +
  facet_wrap(~ metric, scales = "free_y", ncol = 1) +
  scale_color_viridis_c(option = "turbo", begin = 0.1, end = 0.85, name = "Time  (s)") +
  labs(
    title = "EMG10 recap — EX_STA fatigue summary",
    subtitle = "\u2605 = MVC_RECOV_STA mean",
    x = "Time  (s)",
    y = NULL
  )
print(emg10_sta_summary_plot)

# Optional redesign export: selected EMG-focused plots only.
# Keeps the same RUN_ID-based naming idea as the Python QC export, but
# writes a separate redesign file so it does not interfere with pipeline outputs.
EXPORT_EMG_PLOTS <- TRUE
EXPORT_RUN_ID <- "684LuSh_20251013"
EXPORT_EMG10_REF_RMS <- TRUE
EXPORT_EMG10_DYN_RATIO <- TRUE
EXPORT_EMG10_DYN_SUMMARY <- TRUE
EXPORT_EMG10_STA_RATIO <- TRUE
EXPORT_EMG10_STA_SUMMARY <- TRUE

plot_export_flags <- c(
  emg10_ref_rms_overlay_plot = EXPORT_EMG10_REF_RMS,
  emg10_dyn_ratio_plot = EXPORT_EMG10_DYN_RATIO,
  emg10_dyn_summary_plot = EXPORT_EMG10_DYN_SUMMARY,
  emg10_sta_ratio_plot = EXPORT_EMG10_STA_RATIO,
  emg10_sta_summary_plot = EXPORT_EMG10_STA_SUMMARY
)

plot_export_map <- list(
  emg10_ref_rms_overlay_plot = emg10_ref_rms_overlay_plot,
  emg10_dyn_ratio_plot = emg10_dyn_ratio_plot,
  emg10_dyn_summary_plot = emg10_dyn_summary_plot,
  emg10_sta_ratio_plot = emg10_sta_ratio_plot,
  emg10_sta_summary_plot = emg10_sta_summary_plot
)

if (isTRUE(EXPORT_EMG_PLOTS)) {
  selected_plot_names <- names(plot_export_flags)[plot_export_flags]
  if (length(selected_plot_names) == 0L) {
    stop("EXPORT_EMG_PLOTS=TRUE but no EMG export plot selector is set to TRUE.")
  }

  qc_export_root <- file.path("results", "QC_EXPORT")
  dir.create(qc_export_root, recursive = TRUE, showWarnings = FALSE)

  qc_emg_export_path <- file.path(
    qc_export_root,
    paste0("qc_emg_export_", EXPORT_RUN_ID, ".pdf")
  )

  grDevices::pdf(
    file = qc_emg_export_path,
    width = 11,
    height = 8.5,
    onefile = TRUE
  )

  for (plot_name in selected_plot_names) {
    print(plot_export_map[[plot_name]])
  }

  grDevices::dev.off()

  message("\n[QC EMG EXPORT] PDF created -> ", normalizePath(qc_emg_export_path, winslash = "/", mustWork = FALSE))
}

message("\nAll 27 plots rendered. Edit theme_amstrom() or PAL to iterate.")
