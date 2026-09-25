"""EUCAIM Common Data Model format, read through eucaim_cdm_reader.CdmReader.

Dataset root layout (as materialized at a EUCAIM node):
    clinical_mandatory_view.csv   one row per patient
    imaging_mandatory_view.csv    one row per image series
    clinical_data/*.csv           normalized tables (patient.csv, ...)
    imaging_data/...              DICOM files (not used yet)

Row grain is one patient: clinical_mandatory_view joined with
clinical_data/patient.csv (which carries outcome-like columns such as
patient_deceased and patient_date_of_last_contact). One derived column is
added: survival_time_days = patient_date_of_last_contact -
cancer_condition_asserted_date, meant to be used with patient_deceased as
--time_col/--event_col.

Column types come from the CDM schema instead of a metadata file:
- xsd:boolean columns -> BOOLEAN ("true"/"false"/"1"/"0" parsed to bool)
- coded columns (explicit code list or ontology concept range) -> NOMINAL,
  with valueSet = the column's full valid code set, sorted. Every node gets
  the same category -> integer mapping without sharing any data. The
  "EUCAIM:"-style system prefix is stripped from values first.
- Float64/Int64 columns -> NUMERIC
- identifiers, dates and free-text columns aren't usable as model inputs and
  get no spec.

EUCAIM ships no precomputed statistics, so NUMERIC normalization stats
(q1/q2/q3, min/max) are computed from each node's local data by default: each
center normalizes slightly differently. When partners agree on shared values,
--stats_file points to a JSON file overriding them per column, either
{"<column>": {"q1": ..., ...}} or {"featureStats": {...}, "outcomeStats": {...}}.
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

from flcore.data_sources.base import (
    BOOLEAN,
    FEATURE,
    NOMINAL,
    NUMERIC,
    OUTCOME,
    ColumnSpec,
    DataSource,
)

CLINICAL_VIEW = "clinical_mandatory_view.csv"
PATIENT_TABLE = Path("clinical_data") / "patient.csv"

SURVIVAL_TIME_COLUMN = "survival_time_days"
OUTCOME_COLUMNS = {"patient_deceased", "patient_cause_of_death", SURVIVAL_TIME_COLUMN}

_TRUE = {"true", "1"}
_FALSE = {"false", "0"}

# load_table is called several times per client (config validation, stats,
# loading); reading + validating the CSVs is the slow part, so cache per root.
_TABLE_CACHE: Dict[str, pd.DataFrame] = {}


def _schema_fields():
    """The CDM FieldSpecs for the patient-level table's columns: the clinical
    mandatory view's, then patient.csv's columns the view doesn't have."""
    from eucaim_cdm_reader import schema

    fields = list(schema.CLINICAL_MANDATORY_FIELDS)
    seen = {f.name for f in fields}
    fields += [f for f in schema.PATIENT_FIELDS if f.name not in seen]
    return fields


def _valid_codes(field):
    """Bare codes a coded column accepts, mirroring the reader's own
    validator: every ontology subclass of the concept range when there is one,
    else the explicit code list; None for uncoded columns."""
    if field.concept_range:
        from eucaim_cdm_reader.ontology import get_ontology

        ontology = get_ontology()
        return sorted(set().union(*(ontology.descendants_of(c) for c in field.concept_range)))
    if field.code_list:
        return sorted(field.code_list)
    return None


def _strip_prefix(value):
    if isinstance(value, str) and ":" in value:
        return value.split(":", 1)[1]
    return value


def _to_bool(value):
    if pd.isna(value):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"not a boolean: {value!r}")


def _numeric_stats(series: pd.Series) -> Dict:
    values = series.dropna().astype(float)
    if values.empty:
        return {"numOfNotNull": 0}
    return {
        "numOfNotNull": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "q1": float(values.quantile(0.25)),
        "q2": float(values.quantile(0.50)),
        "q3": float(values.quantile(0.75)),
    }


class EucaimSource(DataSource):
    # load_survival one-hot encodes raw categorical columns per node, which
    # gives nodes with different local categories different feature columns.
    # EUCAIM columns have federation-wide category sets, so encode them first.
    encode_for_survival = True

    def resolve(self, config: dict) -> None:
        root = self._find_root(config)
        missing = [p for p in (CLINICAL_VIEW, PATIENT_TABLE) if not (root / p).exists()]
        if missing:
            print(f"Not a EUCAIM CDM dataset root: {root} (missing {', '.join(map(str, missing))})")
            sys.exit(1)
        config["data_file"] = str(root)

        stats_file = config.get("stats_file")
        if stats_file and not Path(stats_file).exists():
            print(f"--stats_file {stats_file} does not exist")
            sys.exit(1)

    @staticmethod
    def _find_root(config: dict) -> Path:
        """--data_id as given if it exists; else --data_id under $DATA_PATH
        (production) or --data_path; else $DATA_PATH itself, when the node
        mounts a single dataset root there."""
        data_id = Path(config["data_id"])
        if data_id.exists():
            return data_id
        bases = []
        if not config.get("testing_mode") and os.getenv("DATA_PATH"):
            bases.append(Path(os.getenv("DATA_PATH")))
        if config.get("data_path"):
            bases.append(Path(config["data_path"]))
        for base in bases:
            if (base / data_id).exists():
                return base / data_id
        for base in bases:
            if (base / CLINICAL_VIEW).exists():
                return base
        return data_id

    def load_table(self, config: dict) -> pd.DataFrame:
        root = config["data_file"]
        if root not in _TABLE_CACHE:
            _TABLE_CACHE[root] = self._read_patient_table(root)
        return _TABLE_CACHE[root].copy()

    def _read_patient_table(self, root: str) -> pd.DataFrame:
        from eucaim_cdm_reader import CdmReader

        reader = CdmReader(root)
        view = reader.mandatory_clinical_data
        duplicated = view["patient_id"][view["patient_id"].duplicated()].unique()
        if len(duplicated) > 0:
            raise ValueError(
                f"{CLINICAL_VIEW} must have one row per patient, but {len(duplicated)} patient_id(s) "
                f"repeat (e.g. {list(duplicated[:5])})"
            )

        patient = reader.patient_data
        extra = [c for c in patient.columns if c not in view.columns]
        table = view.merge(patient[["patient_id", *extra]], on="patient_id", how="left")

        for field in _schema_fields():
            name = field.name
            if name not in table.columns:
                continue
            if field.xsd_range == "boolean":
                table[name] = table[name].map(_to_bool).astype(object)
            elif _valid_codes(field) is not None:
                table[name] = table[name].map(_strip_prefix).astype(object)
            elif field.pandas_dtype in ("Float64", "Int64"):
                table[name] = table[name].astype("float64")

        if {"patient_date_of_last_contact", "cancer_condition_asserted_date"} <= set(table.columns):
            elapsed = table["patient_date_of_last_contact"] - table["cancer_condition_asserted_date"]
            table[SURVIVAL_TIME_COLUMN] = elapsed.dt.days.astype("float64")

        return table

    def column_specs(self, config: dict) -> List[ColumnSpec]:
        table = self.load_table(config)
        overrides = self._stats_overrides(config)

        specs = []
        for field in _schema_fields() + [None]:
            if field is None:
                name, dtype, codes = SURVIVAL_TIME_COLUMN, NUMERIC, None
            else:
                name, codes = field.name, _valid_codes(field)
                if field.xsd_range == "boolean":
                    dtype = BOOLEAN
                elif codes is not None:
                    dtype = NOMINAL
                elif field.pandas_dtype in ("Float64", "Int64"):
                    dtype = NUMERIC
                else:
                    continue
            if name not in table.columns:
                continue

            column = table[name]
            if dtype == NUMERIC:
                stats = _numeric_stats(column)
            else:
                stats = {"numOfNotNull": int(column.notna().sum())}
                if dtype == NOMINAL:
                    self._check_codes(name, column, codes)
                    stats["valueSet"] = codes
            stats.update(overrides.get(name, {}))

            specs.append(
                ColumnSpec(
                    name=name,
                    dtype=dtype,
                    role=OUTCOME if name in OUTCOME_COLUMNS else FEATURE,
                    stats=stats,
                )
            )
        return specs

    @staticmethod
    def _check_codes(name, column, codes):
        unknown = sorted(set(column.dropna()) - set(codes))
        if unknown:
            raise ValueError(
                f"Column {name!r} has values outside its EUCAIM CDM code set: {unknown[:10]} -- the "
                f"dataset isn't CDM-conformant (check it with eucaim_cdm_reader's CdmReader.validate())"
            )

    @staticmethod
    def _stats_overrides(config: dict) -> Dict[str, Dict]:
        stats_file = config.get("stats_file")
        if not stats_file:
            return {}
        with open(stats_file) as f:
            content = json.load(f)
        if "featureStats" in content or "outcomeStats" in content:
            return {**content.get("featureStats", {}), **content.get("outcomeStats", {})}
        return content
