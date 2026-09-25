"""Synthetic EUCAIM CDM dataset generator for tests.

Writes the directory layout eucaim_cdm_reader.CdmReader reads (mandatory views
+ the mandatory normalized tables under clinical_data/), generated from the
reader's own schema: every required/conditional column is present, typed as
the schema declares, and coded columns hold "EUCAIM:<code>" values from the
column's valid code set -- so CdmReader.validate() passes. IDs and foreign keys
are consistent across tables, and patient_deceased depends on age at
diagnosis so models have a real signal to learn.
"""
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from eucaim_cdm_reader import schema
from eucaim_cdm_reader.ontology import get_ontology

TRAIN_LABELS = [
    "cancer_condition_age_at_diagnosis",
    "patient_birth_sex",
    "cancer_condition_type",
    "first_treatment_offset_from_diagnosis",
]


def valid_codes(field):
    """Bare codes a coded column accepts: the explicit code list when the
    schema gives one, else every ontology subclass of its concept range."""
    if field.code_list:
        return sorted(field.code_list)
    if field.concept_range:
        ontology = get_ontology()
        return sorted(set().union(*(ontology.descendants_of(c) for c in field.concept_range)))
    return None


def _default_value(field, key):
    codes = valid_codes(field)
    if codes:
        return f"EUCAIM:{codes[0]}"
    if field.is_date:
        return "2020-01-01"
    if field.xsd_range == "boolean":
        return "false"
    if field.pandas_dtype == "Float64":
        return "1.0"
    if field.pandas_dtype == "Int64":
        return "0"
    return f"{field.name}_{key}"


def _table(fields, rows, key_field):
    """One DataFrame with every non-optional schema column (plus any optional
    column a row sets), filling unset columns with a schema-valid default."""
    names = [f.name for f in fields if f.status != "optional" or any(f.name in r for r in rows)]
    by_name = {f.name: f for f in fields}
    records = []
    for row in rows:
        records.append({n: row[n] if n in row else _default_value(by_name[n], row[key_field]) for n in names})
    return pd.DataFrame(records, columns=names)


def make_eucaim_fixture(out_dir: Path, task: str = "classification", n_patients: int = 80, seed: int = 0) -> dict:
    """Write a synthetic EUCAIM CDM dataset into out_dir.

    Returns {"data_dir", "train_labels", "target_labels", "time_col",
    "event_col"} ready to merge into a client config with data_source="eucaim".
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    (out_dir / "clinical_data").mkdir(parents=True, exist_ok=True)

    def pick(field_list, name, k=3):
        field = next(f for f in field_list if f.name == name)
        codes = valid_codes(field)[:k]
        return f"EUCAIM:{codes[rng.integers(len(codes))]}"

    patients, conditions, procedures, episodes, events = [], [], [], [], []
    studies, series, clinical_view, imaging_view = [], [], [], []

    for i in range(n_patients):
        pid = f"PAT{i:04d}"
        age = float(np.clip(round(rng.normal(62, 12), 1), 20, 95))
        deceased = rng.random() < 1 / (1 + np.exp(-(age - 62) / 6))
        diagnosis = date(2015, 1, 1) + timedelta(days=int(rng.integers(0, 2000)))
        follow_up = int(rng.integers(30, 600 if deceased else 2500))
        treatment_offset = int(rng.integers(5, 90))
        proc_id, study_uid, series_uid = f"PROC{i:04d}", f"1.2.826.0.{i}", f"1.2.826.0.{i}.1"

        common = {
            "dataset_id": "DS_SYNTH",
            "patient_id": pid,
            "patient_birth_sex": pick(schema.PATIENT_FIELDS, "patient_birth_sex"),
            "patient_diagnostic_category": pick(schema.PATIENT_FIELDS, "patient_diagnostic_category"),
            "cancer_condition_age_at_diagnosis": str(age),
            "cancer_condition_age_unit": pick(schema.CANCER_CONDITION_FIELDS, "cancer_condition_age_unit", k=1),
            "cancer_condition_code": pick(schema.CANCER_CONDITION_FIELDS, "cancer_condition_code"),
            "cancer_condition_asserted_date": diagnosis.isoformat(),
            "cancer_condition_type": pick(schema.CANCER_CONDITION_FIELDS, "cancer_condition_type"),
            "cancer_condition_topography": pick(schema.CANCER_CONDITION_FIELDS, "cancer_condition_topography"),
        }
        procedure_code = pick(schema.PROCEDURE_FIELDS, "procedure_code")

        patients.append({
            **common,
            "patient_managing_organization_id": "ORG1",
            "patient_deceased": "true" if deceased else "false",
            "patient_date_of_last_contact": (diagnosis + timedelta(days=follow_up)).isoformat(),
        })
        conditions.append({**common, "cancer_condition_id": f"CC{i:04d}", "procedure_id": proc_id})
        procedures.append({"procedure_id": proc_id, "patient_id": pid, "procedure_code": procedure_code})
        episodes.append({"episode_id": f"EP{i:04d}", "patient_id": pid})
        events.append({"episode_id": f"EP{i:04d}", "event_table_id": str(i)})
        studies.append({"study_uid": study_uid, "patient_id": pid, "procedure_id": proc_id})
        series.append({"series_uid": series_uid, "study_uid": study_uid, "series_manufacturer_name": "Siemens"})
        clinical_view.append({
            **common,
            "cancer_condition_histology_morphology": "ICDO3:8500/3",
            "evidence_procedure_code": procedure_code,
            "first_treatment_type": "SNOMEDCT:387713003",
            "first_treatment_offset_from_diagnosis": str(treatment_offset),
            "first_treatment_offset_unit": pick(schema.EPISODE_FIELDS, "start_offset_unit", k=1),
        })
        imaging_view.append({
            "dataset_id": "DS_SYNTH",
            "patient_id": pid,
            "image_procedure_code": procedure_code,
            "study_uid": study_uid,
            "series_uid": series_uid,
            "series_manufacturer": "Siemens",
            "series_body_site_code": common["cancer_condition_topography"],
        })

    tables = {
        "clinical_mandatory_view.csv": (schema.CLINICAL_MANDATORY_FIELDS, clinical_view, "patient_id"),
        "imaging_mandatory_view.csv": (schema.IMAGING_MANDATORY_FIELDS, imaging_view, "series_uid"),
        "clinical_data/patient.csv": (schema.PATIENT_FIELDS, patients, "patient_id"),
        "clinical_data/cancer_condition.csv": (schema.CANCER_CONDITION_FIELDS, conditions, "cancer_condition_id"),
        "clinical_data/procedure.csv": (schema.PROCEDURE_FIELDS, procedures, "procedure_id"),
        "clinical_data/episode.csv": (schema.EPISODE_FIELDS, episodes, "episode_id"),
        "clinical_data/episode_event.csv": (schema.EPISODE_EVENT_FIELDS, events, "episode_id"),
        "clinical_data/image_study.csv": (schema.IMAGE_STUDY_FIELDS, studies, "study_uid"),
        "clinical_data/image_series.csv": (schema.IMAGE_SERIES_FIELDS, series, "series_uid"),
    }
    for relative_path, (fields, rows, key) in tables.items():
        _table(fields, rows, key).to_csv(out_dir / relative_path, index=False)

    if task == "classification":
        train_labels, target_labels = TRAIN_LABELS, ["patient_deceased"]
    elif task == "regression":
        train_labels = [c for c in TRAIN_LABELS if c != "cancer_condition_age_at_diagnosis"]
        target_labels = ["cancer_condition_age_at_diagnosis"]
    elif task == "survival":
        train_labels, target_labels = TRAIN_LABELS, []
    else:
        raise ValueError(f"Unknown task: {task}")

    return {
        "data_dir": str(out_dir),
        "train_labels": train_labels,
        "target_labels": target_labels,
        "time_col": "survival_time_days",
        "event_col": "patient_deceased",
    }
