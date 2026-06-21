# Shared plotting style for the AMSTROM R analysis: the modality colour palette, the house
# ggplot theme (theme_amstrom_hgrid), and small label helpers. Sourced by the notebooks and
# stats scripts so every figure looks the same -- not meant to be run on its own.

# Per-modality colours, kept consistent across every figure.
col_rms          <- "#3d6b99"  # EMG amplitude (nRMS) - blue
col_mdf          <- "#a0682a"  # EMG spectral (nMDF) - brown
col_bia          <- "#7b5ea7"  # EIM / bio-impedance - purple
col_nirs         <- "#2a8f6a"  # NIRS (ΔHHb) - green
col_myoton       <- "#b0503a"  # Myoton - terracotta
col_expected     <- "#555555"  # model-expected / reference series
col_ref_line     <- "#666666"  # calibration fit line
col_ref_fill     <- "#888888"  # calibration ribbon (CI / SD band)
col_ref_pts      <- "#aaaaaa"  # calibration reference points
col_neutral_data <- "#4d4d4d"  # neutral data (torque, single-series plots)

# Internal channel IDs -> muscle labels, for plot facets only (exports keep emg6/emg8/emg10).
channel_display_labels <- c(
  emg6 = "RF",
  emg8 = "VM",
  emg10 = "VL"
)

# House theme: half-open frame (left + bottom axis lines only), horizontal gridlines, outward ticks.
theme_amstrom_hgrid <- ggplot2::theme_bw(base_size = 13) +
  ggplot2::theme(
    panel.grid.major.x         = ggplot2::element_blank(),
    panel.grid.minor           = ggplot2::element_blank(),
    panel.grid.major.y         = ggplot2::element_line(colour = "grey91", linewidth = 0.35),
    panel.border               = ggplot2::element_blank(),
    axis.line.x.bottom         = ggplot2::element_line(colour = "grey55", linewidth = 0.4),
    axis.line.y.left           = ggplot2::element_line(colour = "grey55", linewidth = 0.4),
    strip.background           = ggplot2::element_blank(),
    strip.text                 = ggplot2::element_text(face = "bold", size = ggplot2::rel(0.88)),
    legend.position            = "top",
    legend.key.size            = grid::unit(0.8, "lines"),
    legend.text                = ggplot2::element_text(size = ggplot2::rel(0.88)),
    axis.ticks                 = ggplot2::element_line(colour = "grey55", linewidth = 0.4),
    axis.ticks.length.x.bottom = grid::unit(3.5, "pt"),
    axis.ticks.length.y.left   = grid::unit(3.5, "pt"),
    panel.spacing              = grid::unit(3, "pt")
  )

# Axis-label helper: show a normalized value (MVC_REF = 1) as a whole-number percent.
label_pct_100 <- function(x) return(round(x * 100))

# Format a BIA frequency in Hz as a "kHz" label (e.g. 48800 -> "48.8 kHz").
label_bia_frequency <- function(freq_hz) {
  return(paste0(format(freq_hz / 1000, trim = TRUE), " kHz"))
}
