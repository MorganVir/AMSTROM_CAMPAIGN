# _01_participant.py - PARTICIPANT METADATA IMPORT
# First step of the preprocessing pipeline.
# Loads and cleans the participant Excel template for one RUN_ID and caches the result.
#
# Inputs
#   ctx keys  : RUN_ID
#   parameter : CACHE_DIR (Path to the current participant's cache folder)
#   files     : data/participants/<subject_key>*.xlsx
#               where subject_key = first 7 characters of RUN_ID (e.g. "011BeSa")
#
# Outputs
#   ctx keys set : participants_df (single-row DataFrame with cleaned metadata)
#   cache file   : 01_participants.parquet
#
# Notes
#   - The Excel template uses French labels as row indices; LABEL_TO_FIELD maps them to field names.
#   - Raw cell values like "85 kg" or "1,75" are cleaned with regex before numeric parsing.
#   - Some fields (dominant_side, practice_level, patho_history, medication, comments) are not used
#     in early pipeline steps but are kept in the export for later R analysis (e.g. clustering
#     participants by activity level or filtering by pathology history).
#   - The subject key (first 7 chars of RUN_ID) is used for file lookup only.
#     The full RUN_ID is preserved everywhere else to retain session identity.

from pathlib import Path
import glob
import re
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

# Number of characters in RUN_ID that identify the subject (as opposed to the session date).
# Example: RUN_ID = "011BeSa_20251023" → subject_key = "011BeSa" (first 7 chars).
# Used only for locating the Excel file, which is named by subject, not by session.
SUBJECT_KEY_LEN = 7

# Maps French Excel row labels (column 0) to Python field names (column 1).
# All entries are extracted and written to the cache parquet, even if not used in early steps.
# Fields currently reserved for later R analysis: dominant_side, practice_level,
# patho_history, medication, comments.
LABEL_TO_FIELD = {
    "Identifiant  (format: XXXNnPp)":                                        "participant_id",
    "Date de naissance":                                                      "dob",
    "Sexe":                                                                   "sex",
    "Taille":                                                                 "height_cm",
    "Poids":                                                                  "mass_kg",
    "Côté dominant (D/G)":                                                   "dominant_side",
    "Niveau de pratique physique":                                            "practice_level",
    "Historique de pathologies ostéoarticulaires":                           "patho_history",
    "Prise de médications pouvant affecter les réponses physiologiques":     "medication",
    "run number":                                                             "run_number",
    "MVC_REF (moyenne des 3/4 mesures calculée dans le fichier Excel)":      "mvc_ref",
    "DELSYS Timestamp (format YYY/MM/DD hh:mm:ss.ms)":                      "delsys_ts_init",
    "3. Commentaires / Observations pendant l'expérimentation":              "comments",
}

# Fields that must be present and non-null for the pipeline to proceed.
# If any of these are missing from the Excel, extraction will fail earlier (KeyError on label lookup).
REQUIRED_FIELDS = ["participant_id", "dob", "sex", "height_cm", "mass_kg", "delsys_ts_init"]


# ── Main function ──────────────────────────────────────────────────────────────

def run_participant(
    *,
    ctx: dict,
    participants_dir: Path | None = None,
    force_recompute: bool = False,
    CACHE_DIR: Path | None = None,
) -> pd.DataFrame:
    """
    Load participant metadata from the Excel template for the current RUN_ID.
    Returns a single-row DataFrame and writes it to ctx['participants_df'] and cache.

    The Excel file is located by matching the subject key (first 7 chars of RUN_ID)
    against files in participants_dir. If multiple matches exist, the first sorted match is used.

    Set force_recompute=True to bypass the cache and re-read from Excel.
    """

    # --- Resolve run identifiers ---
    run_id = str(ctx["RUN_ID"])
    # Subject key: the first SUBJECT_KEY_LEN characters identify the person across sessions.
    # The rest of RUN_ID (e.g. "_20251023") is the session date and is not used for file lookup.
    subject_key = run_id[:SUBJECT_KEY_LEN]

    if participants_dir is None:
        participants_dir = Path("data/participants")

    # --- Cache check ---
    # If the parquet cache already exists and recompute is not requested, load and return early.
    cache_path = CACHE_DIR / "01_participants.parquet"
    if cache_path.exists() and not force_recompute:
        participants_df = pd.read_parquet(cache_path)
        ctx["participants_df"] = participants_df
        print(f"[01_participants] Loaded from cache: {cache_path}. Set force_recompute=True to reprocess.")
        return participants_df

    print(f"[01_participants] Loading participant Excel for RUN_ID={run_id} (subject_key={subject_key})")

    # --- Define parsing helpers ---

    # Regex that strips everything except digits, dots, commas, and minus signs.
    # Used to clean raw cell values like "85 kg" or "1,75 m" before converting to float.
    _num_re = re.compile(r"[^0-9\.,\-]+")

    def parse_float(x) -> float:
        """Strip non-numeric characters and parse a float, handling comma-as-decimal."""
        s = str(x).strip().lower().replace(",", ".")
        s = _num_re.sub("", s)
        return float(s)

    def extract_one_template(path: Path) -> dict:
        """
        Read one participant Excel template and return a flat dict of cleaned field values.
        The Excel layout has labels in column 0 and values in column 1 (no header row).
        All fields from LABEL_TO_FIELD are extracted; types are cast here, not downstream.
        """
        # Read with no header: col 0 = label, col 1 = value. Set col 0 as index for label lookup.
        df = pd.read_excel(path, header=None)
        s = df.set_index(0)[1]

        # Extract every field defined in LABEL_TO_FIELD. Missing labels raise KeyError here.
        out = {"file": path.name}
        for label, field in LABEL_TO_FIELD.items():
            out[field] = s[label]

        # --- Type casting ---
        # String fields: strip whitespace to avoid invisible trailing spaces causing mismatches.
        out["participant_id"] = str(out["participant_id"]).strip()
        out["sex"] = str(out["sex"]).strip()

        # Date fields: normalize() removes the time component (sets to midnight) for clean date comparisons.
        out["dob"] = pd.to_datetime(out["dob"]).normalize()
        # delsys_ts_init keeps full datetime precision — it is the reference clock anchor for the session.
        out["delsys_ts_init"] = pd.to_datetime(out["delsys_ts_init"])

        # Numeric fields with unit strings (e.g. "175 cm", "85,5 kg").
        out["height_cm"] = parse_float(out["height_cm"])
        out["mass_kg"] = parse_float(out["mass_kg"])

        # Numeric fields that may be empty in the Excel (errors="coerce" → NaN instead of crash).
        out["run_number"] = pd.to_numeric(out["run_number"], errors="coerce")
        out["mvc_ref"] = pd.to_numeric(out["mvc_ref"], errors="coerce")

        return out

    # --- Find and load Excel template ---
    # Glob for any .xls or .xlsx file whose name starts with the subject key.
    # Sorted for determinism in case multiple files match (e.g. a backup copy).
    participants_excel_files = sorted(
        glob.glob(str(Path(participants_dir) / f"{subject_key}*.xls*"))
    )
    if len(participants_excel_files) == 0:
        raise FileNotFoundError(
            f"No Excel template found for subject_key={subject_key!r} in: {participants_dir}"
        )

    template_path = Path(participants_excel_files[0])
    participants_df = pd.DataFrame([extract_one_template(template_path)])

    # --- Validate and write cache ---
    missing = [c for c in REQUIRED_FIELDS if c not in participants_df.columns]
    if missing:
        raise AssertionError(f"Missing required fields after extraction: {missing}")

    ctx["participants_df"] = participants_df
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    participants_df.to_parquet(cache_path, index=False)

    print(f"[01_participants] Done — loaded: {template_path.name}")
    return participants_df
