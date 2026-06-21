# slope_summary.R - Per-participant fatigue-rate (slope) by modality (EX_DYN, 60 reps, N=16)
# Produces the values in article Table 3 ("Differentiated fatigue kinetics"): for each
# continuous signal, the OLS slope against repetition number is fitted per participant, and
# the cohort mean +/- SD and sign consistency are reported. Signals: nRMS, nMDF, normalized
# torque (EMG table), ΔHHb (NIRS, hhb_from_seq_rest_umol), EIM R and PhA RelDev at 48.8 kHz.
# Run: Rscript --vanilla slope_summary.R

suppressPackageStartupMessages({library(arrow); library(dplyr); library(tidyr); library(purrr)})
reg <- read_parquet(here::here("results/R_ANALYSIS/subset_registry/multimodal_subset_registry.parquet"))
ids <- reg |> filter(analysis_block == "cohort_final", included == 1L) |> pull(RUN_ID)
pool <- function(ids, root, f) return(map_dfr(ids, function(x) {
  p <- file.path(root, x, f); if (file.exists(p)) read_parquet(p) else NULL
}))

emg      <- pool(ids, here::here("results/R_ANALYSIS"),  "ex_dyn_mid1s_feature_table_norm.parquet") |> filter(channel == "emg10")  # VL EX_DYN features
nirs_dyn <- pool(ids, here::here("results/DATA_EXPORT"), "nirs_ex_dyn_feature_table.parquet")  # NIRS EX_DYN features
bia_dyn  <- pool(ids, here::here("results/R_ANALYSIS"),  "bia_ex_dyn_feature_table.parquet")   # EIM EX_DYN features

# per-participant OLS slope of `yvar` against repetition number
sl <- function(df, yvar) return(df |> group_by(RUN_ID) |>
  summarise(s = coef(lm(reformulate("VC_count", yvar), na.action = na.omit))[["VC_count"]], .groups = "drop"))

# per-run EMG slopes vs rep (one fit per metric)
emg_s <- emg |> group_by(RUN_ID) |> summarise(
  nRMS   = coef(lm(rms_norm    ~ VC_count))[["VC_count"]],   # nRMS rate
  nMDF   = coef(lm(mdf_norm    ~ VC_count))[["VC_count"]],   # nMDF rate
  Torque = coef(lm(torque_norm ~ VC_count))[["VC_count"]], .groups = "drop")  # nTorque rate
hhb_s <- sl(nirs_dyn, "hhb_from_seq_rest_umol") |> rename(dHHb = s)                            # NIRS ΔHHb rate
R_s   <- sl(bia_dyn |> filter(freq_hz == 48800L, summary_variable == "R"),   "reldev_pct") |> rename(R_reldev   = s)  # EIM R rate at 48.8 kHz
Pha_s <- sl(bia_dyn |> filter(freq_hz == 48800L, summary_variable == "PhA"), "reldev_pct") |> rename(PhA_reldev = s)  # EIM PhA rate at 48.8 kHz

D <- emg_s |> left_join(hhb_s, by = "RUN_ID") |> left_join(R_s, by = "RUN_ID") |> left_join(Pha_s, by = "RUN_ID")  # one row per run, all six slopes

rpt <- function(v, unit) { x <- D[[v]]; x <- x[is.finite(x)]
  cat(sprintf("%-12s mean %+.4f +/- %.4f  [%s]  sign: %d up / %d down  (n=%d)\n",
              v, mean(x), sd(x), unit, sum(x > 0), sum(x < 0), length(x)))
  return(invisible(NULL)) }
cat("=== Per-participant slope vs repetition (EX_DYN, N=16) ===\n")
rpt("nRMS","/rep"); rpt("nMDF","/rep"); rpt("Torque","/rep")
rpt("dHHb","umol/L/rep"); rpt("R_reldev","%/rep"); rpt("PhA_reldev","%/rep")

# ── Linear R2 of each EX_DYN trajectory (Table 3 R2 column) + fresh calibration linearity ──
ref <- pool(ids, here::here("results/R_ANALYSIS"), "reference_level_feature_table_norm.parquet") |> filter(channel == "emg10")
r2_traj <- function(df, yvar) return(df |> group_by(RUN_ID) |>
  summarise(r2 = summary(lm(reformulate("VC_count", yvar), na.action = na.omit))$r.squared, .groups = "drop") |> pull(r2))
r2rpt <- function(lbl, v) { v <- v[is.finite(v)]; cat(sprintf("%-18s R2 = %.2f +/- %.2f\n", lbl, mean(v), sd(v))); return(invisible(NULL)) }

cat("\n=== EX_DYN trajectory linearity: R2 of feature ~ repetition (per-run, cohort mean) ===\n")
r2rpt("nRMS",  r2_traj(emg,"rms_norm"));     r2rpt("nMDF",  r2_traj(emg,"mdf_norm"))
r2rpt("Torque",r2_traj(emg,"torque_norm")); r2rpt("dHHb",  r2_traj(nirs_dyn,"hhb_from_seq_rest_umol"))
r2rpt("R_reldev",   r2_traj(bia_dyn |> filter(freq_hz==48800L, summary_variable=="R"),  "reldev_pct"))
r2rpt("PhA_reldev", r2_traj(bia_dyn |> filter(freq_hz==48800L, summary_variable=="PhA"),"reldev_pct"))

cat("\n=== Fresh reference calibration nRMS ~ nTorque linearity (Fig 8; per-run R2) ===\n")
ref_r2 <- ref |> filter(SEQ %in% c("SVC_REF","MVC_REF")) |> group_by(RUN_ID) |>
  summarise(r2 = summary(lm(rms_norm ~ torque_norm))$r.squared, .groups = "drop") |> pull(r2)
r2rpt("ref nRMS~nTorque", ref_r2)

# ── Shape check: does a quadratic (curvature) term beat the linear fit? ──────────
# Tests whether the low linear R2 is unmodelled shape (two-regime / fast-then-slow) or noise.
# A small consistent gain (here ~0.06-0.13, largest for dHHb) indicates mild curvature on top
# of substantial scatter -> trajectories summarized by the linear rate (no per-subject exp/segmented fit).
r2_quad <- function(df, y) return(df |> group_by(RUN_ID) |>
  summarise(r = summary(lm(reformulate("poly(VC_count, 2)", y)))$r.squared, .groups = "drop") |> pull(r))
shape <- function(lbl, df, y) { l <- r2_traj(df, y); q <- r2_quad(df, y)
  l <- l[is.finite(l)]; q <- q[is.finite(q)]
  cat(sprintf("%-12s linear R2 %.2f -> quadratic R2 %.2f  (gain %+.2f)\n", lbl, mean(l), mean(q), mean(q)-mean(l)))
  return(invisible(NULL)) }
cat("\n=== EX_DYN trajectory shape: linear vs quadratic R2 (per-run, cohort mean) ===\n")
shape("nRMS",emg,"rms_norm"); shape("nMDF",emg,"mdf_norm"); shape("Torque",emg,"torque_norm")
shape("dHHb",nirs_dyn,"hhb_from_seq_rest_umol")
shape("R_reldev",  bia_dyn |> filter(freq_hz==48800L, summary_variable=="R"),  "reldev_pct")
shape("PhA_reldev",bia_dyn |> filter(freq_hz==48800L, summary_variable=="PhA"),"reldev_pct")
