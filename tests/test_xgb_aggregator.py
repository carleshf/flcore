"""Regression test for the one behavior change flagged while migrating xgb onto
BaseFLStrategy: pre-migration, aggregate_fit hard-stopped
the whole round (returned None, {}) if every client submitted empty-bytes params
in the same round. XGBAggregator now falls back to keeping current_model
unchanged instead.

Empty bytes are genuinely reachable, not just a theoretical guard:
flcore/models/xgb/client.py::XGBoostClient.get_parameters returns
[np.array([], dtype=np.uint8)] whenever self.bst is None (no local model
trained yet) -- e.g. a client whose fit() was never successfully called.
"""
from flcore.models.xgb.aggregator import XGBAggregator


def test_all_empty_models_falls_back_to_current_model_bagging():
    aggregator = XGBAggregator(
        models=[b"", b"", b""],
        weights=[1, 1, 1],
        train_method="bagging",
        current_model=b"previous-round-bytes",
    )
    result = aggregator.aggregate()

    assert result == b"previous-round-bytes"
    assert aggregator.updated_current_model == b"previous-round-bytes"


def test_all_empty_models_falls_back_to_current_model_cyclic():
    aggregator = XGBAggregator(
        models=[b"", b""],
        weights=[1, 1],
        train_method="cyclic",
        current_model=b"previous-round-bytes",
    )
    result = aggregator.aggregate()

    assert result == b"previous-round-bytes"


def test_all_empty_models_on_round_one_stays_empty():
    """Round 1, no prior model, every client empty -- current_model's own
    default (b"") carries through rather than crashing."""
    aggregator = XGBAggregator(
        models=[b"", b""],
        weights=[1, 1],
        train_method="bagging",
        current_model=b"",
    )
    result = aggregator.aggregate()

    assert result == b""


def test_partial_empty_models_are_filtered_not_fatal():
    """A mix of one empty and one real client shouldn't be blocked by the
    empty one -- cyclic just needs *a* valid model to pick from."""
    aggregator = XGBAggregator(
        models=[b"", b"real-model-bytes"],
        weights=[1, 1],
        train_method="cyclic",
        current_model=b"",
    )
    result = aggregator.aggregate()

    assert result == b"real-model-bytes"
