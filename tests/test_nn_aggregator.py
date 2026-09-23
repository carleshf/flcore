"""Deterministic regression test for flcore/models/nn/aggregator.py::NNAggregator.

nn's end-to-end smoke test can't reliably golden-diff pre/post migration --
torch's model init and dataloader shuffling aren't seeded, so even two runs of
the *same* code produce different losses (confirmed manually while migrating
nn onto BaseFLStrategy, see CLAUDE.md Sec 5.11). This test instead verifies
the merge formula itself against fixed inputs, reproducing the exact inline
weighted-average UncertaintyWeightedFedAvg.aggregate_fit used to do by hand
before the migration (num_examples / (epsilon + entropy) per client, then a
normalized weighted sum of ndarray layers) -- bit-for-bit, not just "close".
"""
import numpy as np

from flcore.models.nn.aggregator import NNAggregator


def test_nn_aggregator_matches_pre_migration_inline_formula():
    models = [
        [np.array([1.0, 2.0]), np.array([[1.0, 1.0], [1.0, 1.0]])],
        [np.array([3.0, 4.0]), np.array([[2.0, 2.0], [2.0, 2.0]])],
        [np.array([5.0, 6.0]), np.array([[3.0, 3.0], [3.0, 3.0]])],
    ]
    num_examples = [40, 15, 25]
    entropies = [0.3, 1.2, 0.7]
    epsilon = 1e-3

    # Pre-migration inline formula (flcore/models/nn/FedCustomAggregator.py,
    # UncertaintyWeightedFedAvg.aggregate_fit, before it extended BaseFLStrategy).
    weights_results = []
    agg_weights = []
    for m, n, e in zip(models, num_examples, entropies):
        w = n / (epsilon + e)
        weights_results.append((m, w))
        agg_weights.append(w)
    wsum = np.sum(agg_weights) + 1e-12
    scaled = [(params, w / wsum) for params, w in weights_results]
    expected = None
    for params, alpha in scaled:
        if expected is None:
            expected = [alpha * p for p in params]
        else:
            expected = [np.add(acc, alpha * p) for acc, p in zip(expected, params)]

    # Post-migration: BaseFLStrategy._compute_weights' formula + NNAggregator.
    weights = [n / (epsilon + e) for n, e in zip(num_examples, entropies)]
    actual = NNAggregator(models=models, weights=weights).aggregate()

    for expected_layer, actual_layer in zip(expected, actual):
        assert np.array_equal(expected_layer, actual_layer)
