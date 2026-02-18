# PARTICIPANT MODULE
# Output : participants_df


from importlib.resources import path
from pathlib import Path
import glob
import re
import pandas as pd


PART_DIR = Path("data/participants")

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

_num_re = re.compile(r"[^0-9\.,\-]+") # clean data - Remove all characters except digits, dot, comma, and minus sign

# function to turn ",." into "." and remove any other characters (python don't like comma as decimal separator) could be leftover when typing in french
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

participants_excel_files = sorted(glob.glob(str(PART_DIR / "*.xls*")))
assert len(participants_excel_files) > 0, f"No Excel templates found in: {PART_DIR}"

participants_df = pd.DataFrame([extract_one_template(Path(f)) for f in participants_excel_files])

missing = [c for c in REQUIRED_FIELDS if c not in participants_df.columns]
assert len(missing) == 0, f"Missing required fields: {missing}"


# Workspace cleanup (free ram, clean namespace)
# Keep only: participants_df

del PART_DIR, LABEL_TO_FIELD, REQUIRED_FIELDS
del _num_re, parse_float, extract_one_template
del participants_excel_files, missing
del Path, glob, re


participants_df
