# Build a clean, canonical participant characteristics table for the N=16 cohort.
# Per-participant rows are retained (for an optional detailed table); a cohort
# summary is printed. Cleaning applied:
#   - sex normalized to F / M (source uses F / H / Masculin / Homme)
#   - MVC recomputed from MEASURED torque (MVC_REF grand mean, Nm), not the
#     hand-logged xlsx field (which has missing values and is approximate)
#   - DOB corrections for known source-form entry errors (see DOB_CORRECTIONS)
# Outputs: results/R_ANALYSIS/participant_characteristics_clean.{parquet,csv}
suppressMessages({library(arrow); library(dplyr)})

# Known DOB entry errors in the source *_Infos.xlsx forms. 802BlHu's form held
# Excel serial 44876 (= 2022-11-11), giving an impossible age of ~3 yr; corrected
# to 2002 per participant confirmation. Add further corrections here if found.
DOB_CORRECTIONS <- c("802BlHu_20251028" = "2002-11-11")

# Physical-activity classification (manual, from the free-text `practice_level` field).
# Threshold for "active": currently engaged in regular structured exercise (>= 2 sessions/week).
# The cohort is heavily imbalanced (~13 active / 3 inactive), so this supports description only,
# NOT a subgroup analysis. Borderline cases (only light/occasional or past-only activity):
# 151FuCy (daily 20-min walk only), 802BlHu (past school sport only); 011BeSa self-reported
# "actuellement non sportive". A validated instrument (e.g. IPAQ) is recommended prospectively.
INACTIVE_RUN_IDS <- c("011BeSa_20251023", "151FuCy_20251105", "802BlHu_20251028")

reg <- read_parquet(here::here("results/R_ANALYSIS/subset_registry/multimodal_subset_registry.parquet"))
runs <- reg |>
  filter(analysis_block == "cohort_final", included == 1L) |>
  distinct(RUN_ID) |>
  pull(RUN_ID)
stopifnot(length(runs) == 16)

norm_sex <- function(s) {
  s <- trimws(toupper(as.character(s)))
  return(ifelse(s %in% c("F", "FEMININ", "FEMME", "FÉMININ"), "F", "M"))
}

# Measured MVC per run = mean of MVC_REF torque_mean_Nm
mvc_from_torque <- function(r) {
  f <- file.path(here::here("results/R_ANALYSIS"), r, "reference_level_feature_table_norm.parquet")
  if (!file.exists(f)) return(NA_real_)
  t <- read_parquet(f)
  if (!all(c("SEQ", "torque_mean_Nm") %in% names(t))) return(NA_real_)
  return(mean(t$torque_mean_Nm[t$SEQ == "MVC_REF"], na.rm = TRUE))
}

rows <- lapply(runs, function(r) {
  p <- read_parquet(file.path(here::here("results/DATA_EXPORT"), r, "participant.parquet"))
  sess <- as.Date(sub(".*_([0-9]{8})$", "\\1", r), format = "%Y%m%d")
  dob  <- if (r %in% names(DOB_CORRECTIONS)) as.Date(DOB_CORRECTIONS[[r]]) else as.Date(p$dob)
  dob_corrected <- r %in% names(DOB_CORRECTIONS)
  age  <- as.numeric(sess - dob) / 365.25
  age_flag <- is.na(age) || age < 15 || age > 70   # residual implausible DOB
  data.frame(
    run_id         = r,
    participant_id = p$participant_id,
    sex            = norm_sex(p$sex),
    dob            = as.character(dob),
    dob_corrected  = dob_corrected,
    age_yr         = ifelse(age_flag, NA_real_, age),
    age_flag       = age_flag,
    height_cm      = as.numeric(p$height_cm),
    mass_kg        = as.numeric(p$mass_kg),
    mvc_ref_nm     = mvc_from_torque(r),
    practice_level    = gsub("[\r\n]+", " ", trimws(as.character(p$practice_level))),
    physical_activity = ifelse(r %in% INACTIVE_RUN_IDS, "inactive", "active"),
    stringsAsFactors = FALSE
  )
})
d <- bind_rows(rows)  # one row per participant

dir.create(here::here("results/R_ANALYSIS"), showWarnings = FALSE, recursive = TRUE)
write_parquet(d, here::here("results/R_ANALYSIS/participant_characteristics_clean.parquet"))
write.csv(d, here::here("results/R_ANALYSIS/participant_characteristics_clean.csv"), row.names = FALSE)

print(d, row.names = FALSE)

if (any(d$age_flag))
  cat("\n[FLAG] residual DOB entry error (age NA):",
      paste(d$run_id[d$age_flag], collapse = ", "), "\n")

# format a numeric vector as "mean +/- SD (range, n)" for the cohort summary lines.
f2 <- function(x) {
  x <- x[is.finite(x)]
  return(sprintf("%.1f +/- %.1f (range %.1f-%.1f, n=%d)",
          mean(x), sd(x), min(x), max(x), length(x)))
}
cat("\n--- COHORT SUMMARY (N=", nrow(d), ") ---\n", sep = "")
cat(sprintf("Sex:       %d M / %d F\n", sum(d$sex == "M"), sum(d$sex == "F")))
cat("Age (yr):  ", f2(d$age_yr), "\n")
cat("Height(cm):", f2(d$height_cm), "\n")
cat("Mass (kg): ", f2(d$mass_kg), "\n")
cat("MVC (Nm):  ", f2(d$mvc_ref_nm), "\n")
cat(sprintf("Activity:   %d active / %d inactive (manual, from practice_level; imbalanced)\n",
            sum(d$physical_activity == "active"), sum(d$physical_activity == "inactive")))
