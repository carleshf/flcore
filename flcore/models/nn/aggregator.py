"""Aggregator for the nn model: weighted average of raw ndarray layers.

Generalizes what UncertaintyWeightedFedAvg.aggregate_fit used to do by hand
(flcore/models/nn/FedCustomAggregator.py, pre-Phase-3): weight each client's
params and sum them layer by layer. Divides by the total weight itself
(matching flcore/models/cox/aggregator.py::CoxAggregator's defensive style),
so it doesn't assume the weights it's given are already normalized to sum to
1 -- BaseFLStrategy._compute_weights overrides (like nn's entropy-based one)
don't have to pre-normalize.
"""
import numpy as np

from flcore.base_aggregator import BaseAggregator


class NNAggregator(BaseAggregator):
    def aggregate(self):
        total_weight = sum(self.weights) + 1e-12
        aggregated = None
        for params, weight in zip(self.models, self.weights):
            alpha = weight / total_weight
            if aggregated is None:
                aggregated = [alpha * layer for layer in params]
            else:
                aggregated = [np.add(acc, alpha * layer) for acc, layer in zip(aggregated, params)]
        return aggregated
