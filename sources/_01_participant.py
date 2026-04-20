# _01_participant.py - PARTICIPANT METADATA IMPORT
# Load and clean the participant Excel template for one RUN_ID.
#
# Inputs
#   ctx keys:  RUN_ID, CACHE_DIR
#   files:     data/participants/<RUN_ID[:7]>*.xlsx
#
# Outputs (ctx keys set)
#   - participants_df  (single-row DataFrame with cleaned metadata)
#
# Cache
#   - 01_participants.parquet
#
# Notes
#   - Non-standard cell values (e.g. "85 kg") are cleaned with regex before parsing.
#   - Participant identity uses only the first 7 characters of RUN_ID as lookup key.

from pathlib import Path
import glob
import re
import pandas as pd


LABEL_TO_FIELD = {
    "Identifiant  (format: XXXNnPp)": "participant_id",
    "Date de naissance": "dob",
    "Sexe": "sex",
    "Taille": "height_cm",
    "Poids": "mass_kg",
    "Côté dominant (D/G)": "dominant_side",
    "Niveau de pratique physique": "practice_level",
    "Historique de pathologies ostéoarticulaires": "patho_history",
    "Prise de médications pouvant affecter les réponses physiologiques": "medication",
    "run number": "run_number",
    "MVC_REF (moyenne des 3/4 mesures calculée dans le fichier Excel)": "mvc_ref",
    "DELSYS Timestamp (format YYY/MM/DD hh:mm:ss.ms)": "delsys_ts_init",
    "3. Commentaires / Observations pendant l'expérimentation": "comments",
}

REQUIRED_FIELDS = ["participant_id", "dob", "sex", "height_cm", "mass_kg", "delsys_ts_init"]

def run_participant(
    *,  # force keyword-only args
    ctx: dict,
    participants_dir: Path | None = None,
    force_recompute: bool = False,
    CACHE_DIR: Path | None = None,
) -> pd.DataFrame:
    """
    01 — Participant import from ONE Excel template.
    Output: participants_df (single-row)

    Requires ctx keys:
      RUN_ID, CACHE_DIR
    """

    if "CACHE_DIR" not in ctx or ctx["CACHE_DIR"] is None:
        raise KeyError("[01_participants] missing ctx['CACHE_DIR']")
    cache_dir: Path = ctx["CACHE_DIR"]
    if not isinstance(cache_dir, Path):
        raise TypeError(f"[01_participants] ctx['CACHE_DIR'] must be pathlib.Path, got {type(cache_dir)}")

    if "RUN_ID" not in ctx or ctx["RUN_ID"] is None:
        raise KeyError("[01_participants] missing ctx['RUN_ID']")
    run_id = str(ctx["RUN_ID"])
    subj_key = run_id[:7]

    # default path
    if participants_dir is None:
        participants_dir = Path("data/participants")

    cache_path = CACHE_DIR / "01_participants.parquet"

    # cache hit
    if cache_path.exists() and (not force_recompute):
        participants_df = pd.read_parquet(cache_path)
        ctx["participants_df"] = participants_df
        print(f"[01_participants] Loaded participants_df from cache: {cache_path}.\n Set force_recompute=True to reprocess or manually delete the cache.")
        return participants_df

    print(f"[01_participants] Loading participant Excel for RUN_ID={run_id} (key={subj_key})")

    _num_re = re.compile(r"[^0-9\.,\-]+")

    def parse_float(x) -> float:
        s = str(x).strip().lower().replace(",", ".")
        s = _num_re.sub("", s)
        return float(s)

    def extract_one_template(path: Path) -> dict:
        df = pd.read_excel(path, header=None)
        s = df.set_index(0)[1]

        out = {"file": path.name}
        for label, field in LABEL_TO_FIELD.items():
            out[field] = s[label]

        out["participant_id"] = str(out["participant_id"]).strip()
        out["sex"] = str(out["sex"]).strip()

        out["dob"] = pd.to_datetime(out["dob"]).normalize()
        out["delsys_ts_init"] = pd.to_datetime(out["delsys_ts_init"])

        out["height_cm"] = parse_float(out["height_cm"])
        out["mass_kg"] = parse_float(out["mass_kg"])

        out["run_number"] = pd.to_numeric(out["run_number"], errors="coerce")
        out["mvc_ref"] = pd.to_numeric(out["mvc_ref"], errors="coerce")

        return out

    # only grab the one linked to RUN_ID (participant key prefix)
    participants_excel_files = sorted(glob.glob(str(Path(participants_dir) / f"{subj_key}*.xls*")))
    if len(participants_excel_files) == 0:
        raise FileNotFoundError(f"No Excel template found for participant key {subj_key} in: {participants_dir}")

    # keep it deterministic: pick the first match
    template_path = Path(participants_excel_files[0])

    participants_df = pd.DataFrame([extract_one_template(template_path)])

    missing = [c for c in REQUIRED_FIELDS if c not in participants_df.columns]
    if len(missing) != 0:
        raise AssertionError(f"Missing required fields: {missing}")

    ctx["participants_df"] = participants_df
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    participants_df.to_parquet(cache_path, index=False)

    print(f"[01_participants] Successfully loaded participant sheet: {template_path.name}")

    return participants_df