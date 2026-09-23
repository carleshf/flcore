"""Aggregator for the linear_models family: weighted average of raw ndarray
layers -- identical shape to flcore/models/nn/aggregator.py::NNAggregator.

Generalizes what FedCustomAggregator.aggregate_fit used to do by calling
flcore/smoothWeights.py::smooth_aggregate directly. That function both computes
the per-client weights (via computeSmoothedWeights) *and* does the weighted
sum in one call; BaseFLStrategy.aggregate_fit already computes weights via
_compute_weights (computeSmoothedWeights, unmodified default) before handing
them to this aggregator, so this only needs the second half -- the weighted
sum -- not a full re-implementation of the first half too.
"""
import numpy as np

from flcore.base_aggregator import BaseAggregator


class LinearModelAggregator(BaseAggregator):
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
