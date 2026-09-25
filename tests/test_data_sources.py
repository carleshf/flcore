"""Tests for flcore/data_sources: the DT4H source must describe columns exactly
as the metadata.json parsing it replaced did, in both metadata.json shapes, and
checkpoint metadata must keep the same features_meta/outcomes_meta content.
"""
import json

import pytest

from flcore.data_sources import (
    FEATURE,
    OUTCOME,
    build_checkpoint_metadata,
    find_spec,
    get_data_source,
)

from fixtures.synthetic_dt4h import make_dt4h_fixture


def _flatten_metadata(data_dir):
    """Rewrite a fixture's metadata.json from the {"entries": [...]} shape to
    the flat shape (same keys at the top level) DT4HSource also accepts."""
    path = data_dir / "metadata.json"
    entry = json.loads(path.read_text())["entries"][0]
    path.write_text(json.dumps(entry))


def _config(data_info, **overrides):
    config = {
        "data_source": "dt4h",
        "data_id": data_info["data_dir"],
        "train_labels": data_info["train_labels"],
        "target_labels": data_info["target_labels"],
        "node_name": "client_0",
        "task": "classification",
        "n_out": 1,
        "n_feats": len(data_info["train_labels"]),
        "model": "random_forest",
    }
    config.update(overrides)
    get_data_source(config).resolve(config)
    return config


def _legacy_checkpoint_meta(config):
    """The features_meta/outcomes_meta computation every model client's
    save_model() used to inline, kept verbatim as the compatibility reference."""
    data_metadata = json.load(open(config["metadata_file"], "r"))
    entity = data_metadata.get("entries", [])[0]
    features_list = entity.get("features", [])
    outcomes_list = entity.get("outcomes", [])
    dataset_stats = entity.get("datasetStats", {})
    feature_stats = dataset_stats.get("featureStats", {})
    outcome_stats = dataset_stats.get("outcomeStats", {})
    all_features_meta = {f["name"]: f for f in features_list}
    all_outcomes_meta = {o["name"]: o for o in outcomes_list}
    for f_name, f_meta in all_features_meta.items():
        f_meta["stats"] = feature_stats.get(f_name, {})
    for o_name, o_meta in all_outcomes_meta.items():
        o_meta["stats"] = outcome_stats.get(o_name, {})
    features_meta = {}
    for label in config["train_labels"]:
        if label in all_features_meta:
            features_meta[label] = all_features_meta[label]
        elif label in all_outcomes_meta:
            features_meta[label] = all_outcomes_meta[label]
    outcomes_meta = {}
    for label in config["target_labels"]:
        if label in all_outcomes_meta:
            outcomes_meta[label] = all_outcomes_meta[label]
        elif label in all_features_meta:
            outcomes_meta[label] = all_features_meta[label]
    return features_meta, outcomes_meta


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_checkpoint_metadata_matches_legacy_inline_version(task, tmp_path):
    data_info = make_dt4h_fixture(tmp_path / "data", task=task)
    config = _config(data_info, task=task)

    metadata = build_checkpoint_metadata(config, metrics={"accuracy": 0.5})
    legacy_features_meta, legacy_outcomes_meta = _legacy_checkpoint_meta(config)

    assert metadata["features_meta"] == legacy_features_meta
    assert metadata["outcomes_meta"] == legacy_outcomes_meta
    assert metadata["n_out"] == config["n_out"]
    assert metadata["n_feats"] == config["n_feats"]
    assert metadata["metrics"] == {"accuracy": 0.5}


def test_flat_metadata_shape_gives_same_specs(tmp_path):
    data_info = make_dt4h_fixture(tmp_path / "data")
    entries_specs = get_data_source(_config(data_info)).column_specs(_config(data_info))

    _flatten_metadata(tmp_path / "data")
    flat_specs = get_data_source(_config(data_info)).column_specs(_config(data_info))

    assert flat_specs == entries_specs
    assert {s.role for s in flat_specs} == {FEATURE, OUTCOME}


def test_checkpoint_metadata_works_with_flat_metadata_shape(tmp_path):
    # The legacy inline code indexed entries[0] and crashed on this shape.
    data_info = make_dt4h_fixture(tmp_path / "data")
    _flatten_metadata(tmp_path / "data")
    metadata = build_checkpoint_metadata(_config(data_info), metrics=None)
    assert set(metadata["features_meta"]) == set(data_info["train_labels"])


def test_find_spec_prefers_requested_role(tmp_path):
    data_info = make_dt4h_fixture(tmp_path / "data")
    config = _config(data_info)
    specs = get_data_source(config).column_specs(config)
    target = data_info["target_labels"][0]
    assert find_spec(specs, target, prefer=OUTCOME).role == OUTCOME
    assert find_spec(specs, "no_such_column", prefer=OUTCOME) is None


def test_unknown_data_source_is_rejected():
    with pytest.raises(ValueError, match="Unknown data source"):
        get_data_source({"data_source": "nope"})
