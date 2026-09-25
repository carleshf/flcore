"""Interface every data source (a project's on-disk dataset format) implements.

A data source answers three questions for the rest of flcore, so that config
validation, preprocessing and checkpoint metadata never parse a specific
project's files directly:

- resolve(config): where is the data? Validates it exists and records the
  resolved paths in `config`.
- load_table(config): the raw table, one row per sample, before any encoding
  or normalization.
- column_specs(config): each usable column's type (NUMERIC / NOMINAL /
  BOOLEAN) and the statistics preprocessing needs (q1/q2/q3, min/max,
  valueSet, numOfNotNull).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

NUMERIC = "NUMERIC"
NOMINAL = "NOMINAL"
BOOLEAN = "BOOLEAN"

FEATURE = "feature"
OUTCOME = "outcome"


@dataclass
class ColumnSpec:
    name: str
    dtype: str  # NUMERIC | NOMINAL | BOOLEAN
    role: str = FEATURE  # FEATURE | OUTCOME: where the source declares it, not how it's used
    stats: Dict = field(default_factory=dict)
    # Source-specific description written into checkpoint metadata as-is
    # (for DT4H: the column's original metadata.json entry).
    extra: Dict = field(default_factory=dict)

    def describe(self) -> Dict:
        """This column's entry in a checkpoint's features_meta/outcomes_meta."""
        description = dict(self.extra) if self.extra else {"name": self.name, "dataType": self.dtype}
        description["stats"] = self.stats
        return description


def find_spec(specs: List[ColumnSpec], name: str, prefer: str) -> Optional[ColumnSpec]:
    """The spec for `name`, preferring the one with role `prefer` when a
    source declares the same column both as a feature and as an outcome."""
    matches = [s for s in specs if s.name == name]
    for spec in matches:
        if spec.role == prefer:
            return spec
    return matches[0] if matches else None


class DataSource(ABC):
    # Whether load_survival should encode/normalize feature columns with
    # column_specs before its own per-node one-hot encoding (see
    # EucaimSource). Off for DT4H to keep its survival path unchanged.
    encode_for_survival = False

    @abstractmethod
    def resolve(self, config: dict) -> None:
        """Locate the dataset from config (e.g. --data_id), exit with a clear
        message if it's missing, and store resolved paths in config."""

    @abstractmethod
    def load_table(self, config: dict) -> pd.DataFrame:
        """The raw sample table (one row per sample), columns named as the
        user refers to them in --train_labels / --target_labels / --time_col."""

    @abstractmethod
    def column_specs(self, config: dict) -> List[ColumnSpec]:
        """Typed column descriptions, features first, then outcomes."""
