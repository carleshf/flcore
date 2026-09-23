from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
import flwr as fl
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
import numpy as np
import flwr.server.strategy.fedavg as fedav
import time
import joblib

from flcore.base_strategy import BaseFLStrategy
from flcore.models.nn.aggregator import NNAggregator


class UncertaintyWeightedFedAvg(BaseFLStrategy):
    """Weights each client's params by num_examples / (epsilon + entropy) --
    more data and lower prediction entropy means more confidence in that
    client's update -- then does a weighted average of the raw ndarray layers
    (NNAggregator). Overrides _compute_weights instead of using
    BaseFLStrategy's default computeSmoothedWeights, since this weighting is
    metrics-derived (each client's reported "entropy"), not num_examples/
    smoothing_strenght-derived."""

    def __init__(self, config: dict, epsilon: float = 1e-3, **kwargs):
        super().__init__(
            config=config,
            aggregator_cls=NNAggregator,
            serialize_fn=ndarrays_to_parameters,
            deserialize_fn=parameters_to_ndarrays,
            **kwargs,
        )
        self.epsilon = epsilon

    def _compute_weights(self, deserialized: list, results) -> List[float]:
        weights = []
        for _, fit_res in results:
            entropy = fit_res.metrics.get("entropy", 1.0)
            # more data and lower entropy => more confidence
            weights.append(fit_res.num_examples / (self.epsilon + entropy))
        return weights

