"""DataTools4Heart / AI4HF format: --data_id is a directory holding one or
more *.parquet files plus a metadata.json describing every column.

metadata.json comes in two shapes, both supported:
- {"entries": [{"features": [...], "outcomes": [...], "datasetStats":
  {"featureStats": {...}, "outcomeStats": {...}}}]} (only entries[0] is read)
- the same keys at the top level, without "entries".
Each feature/outcome is {"name": ..., "dataType": NUMERIC|NOMINAL|BOOLEAN, ...};
stats are per column name: q1/q2/q3, min/max, valueSet, numOfNotNull.
"""
import glob
import json
import os
import sys
from typing import List

import pandas as pd

from flcore.data_sources.base import FEATURE, OUTCOME, ColumnSpec, DataSource


class DT4HSource(DataSource):
    def resolve(self, config: dict) -> None:
        data_dir = config["data_id"]
        config["metadata_file"] = os.path.join(data_dir, "metadata.json")

        parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
        if len(parquet_files) == 0:
            print("No parquet files found in ", data_dir)
            sys.exit(1)
        # Several matches: the last one glob returns is used (glob order is
        # filesystem-dependent, not sorted).
        config["data_file"] = parquet_files[-1]

    def load_table(self, config: dict) -> pd.DataFrame:
        return pd.read_parquet(config["data_file"])

    def column_specs(self, config: dict) -> List[ColumnSpec]:
        with open(config["metadata_file"]) as f:
            meta = json.load(f)

        entries = meta.get("entries", [])
        if entries:
            entry = entries[0]
            features = entry["features"]
            outcomes = entry["outcomes"]
            feature_stats = entry["datasetStats"]["featureStats"]
            outcome_stats = entry["datasetStats"]["outcomeStats"]
        else:
            dataset_stats = meta.get("datasetStats", {})
            features = meta.get("features", [])
            outcomes = meta.get("outcomes", [])
            feature_stats = dataset_stats.get("featureStats", {})
            outcome_stats = dataset_stats.get("outcomeStats", {})

        specs = []
        for columns, stats, role in ((features, feature_stats, FEATURE), (outcomes, outcome_stats, OUTCOME)):
            for column in columns:
                specs.append(
                    ColumnSpec(
                        name=column["name"],
                        dtype=column["dataType"],
                        role=role,
                        stats=stats.get(column["name"], {}),
                        extra=column,
                    )
                )
        return specs
