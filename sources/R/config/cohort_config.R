# Shared EX_STA harmonization config — sourced by all four cohort notebooks.
#
# Approach: absolute torque corridor [STA_TORQUE_LOWER, 0.70] expressed in MVC_REF units.
# A window is valid if torque_norm >= STA_TORQUE_LOWER. Runs with fewer than
# STA_MIN_VALID_WINDOWS valid windows are excluded from the EX_STA cohort (insufficient data).
# After exclusion, GLOBAL_STA_N_MIN is the min(last_valid_window) across included runs.
#
# To recompute: run _tmp_compute_sta_n_min.R (or the equivalent inline in §6.4.6.5),
# then update GLOBAL_STA_N_MIN and STA_EXCLUDE_RUN_IDS below.

# Corridor lower bound: 60% MVC_REF (absolute). torque_norm is normalized to pipeline MVC_REF,
# not the on-the-fly experimental MVC; some subjects' SVC_70% target sits at ~62% MVC_REF due to
# ~10% underestimation at collection.
STA_TORQUE_LOWER      <- 0.60

# Minimum valid windows (seconds) for EX_STA cohort inclusion.
# Historical EX_STA setting; EX_STA is currently parked for the M2 memoir.
STA_MIN_VALID_WINDOWS <- 25L

# Common EX_STA truncation point (min last_valid_window across included runs).
# NOTE: 48 is the pre-memoir value (computed with onset window_id=1). With onset window_id=2,
# the actual binding run is 890PeTh_20251023 (last_valid_window=28). EX_STA is parked for the
# M2 memoir — recompute this when EX_STA is un-parked and onset policy is finalised.
GLOBAL_STA_N_MIN      <- 48L

# Runs excluded from the EX_STA cohort (last_valid_window < STA_MIN_VALID_WINDOWS).
# 284ChGe: 21 valid windows
# 092MaLe: 23 valid windows
# Note: 011BeSa (13 valid windows) was here but re-enters cohort_final because EX_STA is
# parked for the M2 memoir. Restore if EX_STA comes back into scope.
STA_EXCLUDE_RUN_IDS   <- c("284ChGe_20251030", "092MaLe_20251031")

# Runs excluded from cohort_final for signal quality reasons (not EX_STA duration).
# These are excluded from ALL cohort notebooks to keep a consistent N everywhere.
# Individual pipelines continue to export data so the exclusion reason can be documented.
#
# 261PoLe_20251128: NIRS session-wide probe drift. ΔHHb decreases at 100% MVC
#   (EX_DYN mean = −5.0 µmol/L — physiologically impossible). Probe movement confirmed
#   during contractions. Per-sequence baseline subtraction partially corrects absolute values
#   but the fatigue trajectory is unreliable. NIRS data excluded; other modalities excluded
#   for N consistency.
COHORT_SIGNAL_EXCLUDE_RUN_IDS <- c("261PoLe_20251128")

# Runs excluded from cohort_final because their EX_DYN ΔHHb direction is inverted
# (ΔHHb < 0, i.e. HHb decreases during 100% MVC contractions). Mixing positive and
# negative direction runs in a cohort mean would cause cancellation artefacts.
# Hypothesis: tHb-dominated vascular response rather than true deoxygenation.
# Individual NIRS pipelines continue to run for documentation.
#
# 622PiSt_20251104: mean EX_DYN hhb_from_seq_rest_umol = −1.82 µmol/L, 87% of reps negative
# 222SoMa_20251113: slope −0.070 µmol/L per rep, 57% of reps negative
NIRS_DIRECTION_EXCLUDE_RUN_IDS <- c("622PiSt_20251104", "222SoMa_20251113")
