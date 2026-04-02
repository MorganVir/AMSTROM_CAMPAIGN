"""
One-off utility to export the full standardized participant table.

This script is intentionally separate from the main pipeline participant step.
It copies the current participant loading/cleaning logic from sources/_01_participant.py
and removes the RUN_ID-based file filtering so every participant template is included.
"""

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

REQUIRED_FIELDS = ["RUN_ID", "dob", "sex", "height_cm", "mass_kg", "delsys_ts_init"]


def export_all_participants(
    *,
    participants_dir: Path = Path("data/participants"),
    output_path: Path = Path("results/REFERENCE_EXPORT/participants_all.parquet"),
) -> Path:
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

        run_id = path.stem
        if run_id.endswith("_Infos"):
            run_id = run_id[: -len("_Infos")]

        out["participant_id"] = str(out["participant_id"]).strip()
        out["RUN_ID"] = run_id.strip()
        out["sex"] = str(out["sex"]).strip()

        out["dob"] = pd.to_datetime(out["dob"]).normalize()
        out["delsys_ts_init"] = pd.to_datetime(out["delsys_ts_init"])

        out["height_cm"] = parse_float(out["height_cm"])
        out["mass_kg"] = parse_float(out["mass_kg"])

        out["run_number"] = pd.to_numeric(out["run_number"], errors="coerce")
        out["mvc_ref"] = pd.to_numeric(out["mvc_ref"], errors="coerce")

        return out

    participants_excel_files = sorted(glob.glob(str(participants_dir / "*.xls*")))
    if len(participants_excel_files) == 0:
        raise FileNotFoundError(f"No participant Excel templates found in: {participants_dir}")

    participant_rows = [extract_one_template(Path(path)) for path in participants_excel_files]
    participants_df = pd.DataFrame(participant_rows)

    missing = [c for c in REQUIRED_FIELDS if c not in participants_df.columns]
    if len(missing) != 0:
        raise AssertionError(f"Missing required fields: {missing}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    participants_df.to_parquet(output_path, index=False)

    print(f"[REFERENCE EXPORT] Wrote full participant table -> {output_path.resolve()}")
    return output_path


if __name__ == "__main__":
    export_all_participants()
