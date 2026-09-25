"""Tests for flcore/data_sources/eucaim.py::EucaimSource against the synthetic
EUCAIM CDM fixture: column typing from the CDM schema, federation-wide
category sets, local vs --stats_file normalization stats, dataset-root
resolution, and the patient-grain / CDM-conformance checks.
"""
import json

import pandas as pd
import pytest

from flcore.data_sources import BOOLEAN, NOMINAL, NUMERIC, OUTCOME, get_data_source
from flcore.data_sources import eucaim
from flcore.datasets import load_survival, load_tabular

from fixtures.synthetic_eucaim import make_eucaim_fixture


@pytest.fixture(autouse=True)
def _clear_table_cache():
    eucaim._TABLE_CACHE.clear()
    yield
    eucaim._TABLE_CACHE.clear()


def _config(data_dir, **overrides):
    config = {"data_source": "eucaim", "data_id": str(data_dir), "data_path": "", "testing_mode": True}
    config.update(overrides)
    get_data_source(config).resolve(config)
    return config


def _specs(config):
    return {s.name: s for s in get_data_source(config).column_specs(config)}


def test_column_types_come_from_cdm_schema(tmp_path):
    info = make_eucaim_fixture(tmp_path / "ds")
    specs = _specs(_config(info["data_dir"]))

    assert specs["cancer_condition_age_at_diagnosis"].dtype == NUMERIC
    assert specs["first_treatment_offset_from_diagnosis"].dtype == NUMERIC
    assert specs["patient_birth_sex"].dtype == NOMINAL
    assert specs["patient_deceased"].dtype == BOOLEAN
    assert specs["patient_deceased"].role == OUTCOME
    assert specs["survival_time_days"].dtype == NUMERIC
    # identifiers, dates and free text are not model inputs
    for name in ("patient_id", "dataset_id", "cancer_condition_asserted_date", "first_treatment_type"):
        assert name not in specs


def test_category_sets_are_identical_across_nodes(tmp_path):
    # Two centers with different local data must get the same code -> int mapping.
    node_a = _specs(_config(make_eucaim_fixture(tmp_path / "a", seed=1)["data_dir"]))
    eucaim._TABLE_CACHE.clear()
    node_b = _specs(_config(make_eucaim_fixture(tmp_path / "b", seed=2, n_patients=30)["data_dir"]))

    for name, spec in node_a.items():
        if spec.dtype == NOMINAL:
            assert spec.stats["valueSet"] == node_b[name].stats["valueSet"], name
            assert len(spec.stats["valueSet"]) > 0


def test_values_are_stripped_and_typed(tmp_path):
    info = make_eucaim_fixture(tmp_path / "ds")
    config = _config(info["data_dir"])
    table = get_data_source(config).load_table(config)

    assert table["patient_id"].is_unique
    assert not table["patient_birth_sex"].str.contains(":").any()
    assert set(table["patient_deceased"].dropna()) <= {True, False}
    assert (table["survival_time_days"] > 0).all()


def test_numeric_stats_are_local_unless_stats_file_given(tmp_path):
    info = make_eucaim_fixture(tmp_path / "ds")
    local = _specs(_config(info["data_dir"]))["cancer_condition_age_at_diagnosis"].stats
    ages = pd.read_csv(tmp_path / "ds" / "clinical_mandatory_view.csv")["cancer_condition_age_at_diagnosis"]
    assert local["q2"] == pytest.approx(ages.median())

    stats_file = tmp_path / "agreed_stats.json"
    stats_file.write_text(json.dumps({"featureStats": {"cancer_condition_age_at_diagnosis": {"q2": 50.0}}}))
    agreed = _specs(_config(info["data_dir"], stats_file=str(stats_file)))["cancer_condition_age_at_diagnosis"].stats
    assert agreed["q2"] == 50.0
    assert agreed["q1"] == local["q1"]  # keys the file doesn't set stay local


def test_root_resolves_under_data_path_env_in_production(tmp_path, monkeypatch):
    make_eucaim_fixture(tmp_path / "mount" / "ds1")
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "mount"))
    config = _config("ds1", testing_mode=False)
    assert config["data_file"] == str(tmp_path / "mount" / "ds1")


def test_root_can_be_data_path_itself(tmp_path, monkeypatch):
    make_eucaim_fixture(tmp_path / "mount")
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "mount"))
    config = _config("data_id.parquet", testing_mode=False)  # --data_id left at its default
    assert config["data_file"] == str(tmp_path / "mount")


def test_missing_dataset_exits(tmp_path):
    with pytest.raises(SystemExit):
        _config(tmp_path / "nothing_here")


def test_duplicate_patients_are_rejected(tmp_path):
    info = make_eucaim_fixture(tmp_path / "ds")
    view_path = tmp_path / "ds" / "clinical_mandatory_view.csv"
    view = pd.read_csv(view_path, dtype=str, keep_default_na=False)
    pd.concat([view, view.iloc[:1]]).to_csv(view_path, index=False)

    config = _config(info["data_dir"])
    with pytest.raises(ValueError, match="one row per patient"):
        get_data_source(config).load_table(config)


def test_non_conformant_codes_are_rejected(tmp_path):
    info = make_eucaim_fixture(tmp_path / "ds")
    view_path = tmp_path / "ds" / "clinical_mandatory_view.csv"
    view = pd.read_csv(view_path, dtype=str, keep_default_na=False)
    view.loc[0, "patient_birth_sex"] = "Female"  # a label, not a CDM code
    view.to_csv(view_path, index=False)

    with pytest.raises(ValueError, match="outside its EUCAIM CDM code set"):
        _specs(_config(info["data_dir"]))


def test_load_tabular_encodes_everything_numerically(tmp_path):
    info = make_eucaim_fixture(tmp_path / "ds")
    config = _config(
        info["data_dir"],
        train_labels=info["train_labels"],
        target_labels=info["target_labels"],
        normalization_method="IQR",
        train_size=0.7,
        task="classification",
    )
    (X_train, y_train), (X_test, y_test) = load_tabular(config)

    assert len(X_train) + len(X_test) == 80
    assert all(pd.api.types.is_numeric_dtype(X_train[c]) for c in X_train.columns)
    assert set(y_train.unique()) <= {0, 1}


def test_load_survival_uses_shared_encoding(tmp_path):
    info = make_eucaim_fixture(tmp_path / "ds", task="survival")
    config = _config(
        info["data_dir"],
        train_labels=info["train_labels"],
        target_labels=[],
        time_col=info["time_col"],
        event_col=info["event_col"],
        accumulative_pattern_col=None,
        negative_duration_strategy="clip",
        normalization_method="IQR",
        train_size=0.7,
        seed=42,
    )
    (X_train, y_train), _, time_col, event_col = load_survival(config)

    # Coded columns arrive already integer-encoded with the federation-wide
    # code set, so get_dummies has nothing to expand: same columns on every node.
    assert list(X_train.columns) == info["train_labels"]
    assert y_train[event_col].dtype == bool
