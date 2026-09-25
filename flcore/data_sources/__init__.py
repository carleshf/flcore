"""Pluggable on-disk dataset formats, selected with --data_source.

Everything that used to parse DT4H's metadata.json directly (config
validation, dataset loading, checkpoint metadata) goes through
get_data_source(config) instead, so supporting another project's format means
adding one DataSource subclass here, not touching models or preprocessing.
"""
from typing import Dict, Optional

from flcore.data_sources.base import (  # noqa: F401
    BOOLEAN,
    FEATURE,
    NOMINAL,
    NUMERIC,
    OUTCOME,
    ColumnSpec,
    DataSource,
    find_spec,
)
from flcore.data_sources.dt4h import DT4HSource
from flcore.data_sources.eucaim import EucaimSource

DATA_SOURCES = {
    "dt4h": DT4HSource,
    "eucaim": EucaimSource,
}


def get_data_source(config: dict) -> DataSource:
    name = config.get("data_source", "dt4h")
    if name not in DATA_SOURCES:
        raise ValueError(f"Unknown data source: {name!r} (available: {sorted(DATA_SOURCES)})")
    return DATA_SOURCES[name]()


def build_checkpoint_metadata(config: dict, metrics: Optional[Dict]) -> Dict:
    """The *_model_metadata.json every model client writes next to a
    checkpoint: training schema (with the stats used to normalize each column)
    plus the round's metrics, enough to reuse the model for inference."""
    specs = get_data_source(config).column_specs(config)

    features_meta = {}
    for label in config["train_labels"]:
        spec = find_spec(specs, label, prefer=FEATURE)
        if spec is not None:
            features_meta[label] = spec.describe()

    outcomes_meta = {}
    for label in config["target_labels"]:
        spec = find_spec(specs, label, prefer=OUTCOME)
        if spec is not None:
            outcomes_meta[label] = spec.describe()

    return {
        "node_name": config["node_name"],
        "task": config["task"],
        "n_out": config["n_out"],
        "n_feats": config["n_feats"],
        "model_type": config["model"],
        "feature_names": config["train_labels"],
        "target_names": config["target_labels"],
        "metrics": metrics,
        "features_meta": features_meta,
        "outcomes_meta": outcomes_meta,
    }
