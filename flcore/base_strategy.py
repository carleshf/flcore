"""Shared Flower Strategy base for the flcore model packages (Phase 3, TODO item 3).

Factors out the two pieces of configure_fit/aggregate_fit boilerplate that used to
be hand-duplicated (with drift) across each model's own FedCustomAggregator/
CustomStrategy: client dropout selection, and the
deserialize -> weight -> merge (via a model's flcore.base_aggregator.BaseAggregator
subclass) -> re-serialize sequence.

aggregate_evaluate is deliberately NOT unified here. Most models can just rely on
flwr's own FedAvg.aggregate_evaluate (num_examples-weighted loss +
evaluate_metrics_aggregation_fn) by passing evaluate_metrics_aggregation_fn to
this class's constructor and not overriding anything. A few (cox/rsf/gbs) have
their own different-on-purpose behavior (an unweighted mean, a results_history.json
side effect) that predates this class and is exercised by tests/golden/ -- folding
that into one shared method would be a real behavior change, not a refactor, so
those keep their own aggregate_evaluate override, now built on top of this class
for configure_fit/aggregate_fit instead of duplicating those too.
"""
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import FitIns, FitRes, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flcore.dropout import select_clients
from flcore.smoothWeights import computeSmoothedWeights


class BaseFLStrategy(fl.server.strategy.FedAvg):
    """Shared configure_fit (dropout) + aggregate_fit (weight -> per-model
    Aggregator.aggregate() -> re-serialize) for any model whose aggregation reduces
    to "merge N clients' deserialized params into one weighted-average result".

    aggregator_cls: a flcore.base_aggregator.BaseAggregator subclass.
    serialize_fn/deserialize_fn: the model's existing Parameters codec (e.g.
        flcore.serialization_funs.serialize_RF/deserialize_RF for tree models,
        flwr.common.ndarrays_to_parameters/parameters_to_ndarrays for numeric-
        tensor models).
    """

    def __init__(
        self,
        config: dict,
        aggregator_cls,
        serialize_fn,
        deserialize_fn,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.aggregator_cls = aggregator_cls
        self.serialize_fn = serialize_fn
        self.deserialize_fn = deserialize_fn
        self.dropout_method = config.get("dropout_method", "None")
        self.dropout_percentage = config.get("dropout_percentage", 0.0)
        self.smooth_method = config.get("smooth_method", "None")
        self.smoothing_strenght = config.get("smoothing_strenght", 0.0)
        self.clients_first_round_time: Dict[str, float] = {}
        self.clients_num_examples: Dict[str, int] = {}

    def _compute_weights(self, deserialized: list, results) -> List[float]:
        """Per-client weight for the merge, one entry per `results`/`deserialized`
        item, in order. Default: computeSmoothedWeights (num_examples + smoothing).
        Override for a model-specific weighting scheme (e.g. nn's entropy-based
        confidence weights) instead of hardcoding one scheme for every model."""
        num_examples = [fit_res.num_examples for _, fit_res in results]
        return computeSmoothedWeights(
            list(zip(deserialized, num_examples)), self.smooth_method, self.smoothing_strenght
        )

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        if self.dropout_method != "None" and server_round > 1:
            clients = select_clients(
                self.dropout_method,
                self.dropout_percentage,
                clients,
                self.clients_first_round_time,
                server_round,
                self.clients_num_examples,
            )

        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Deserialize each client's params, weight them, merge via aggregator_cls,
        re-serialize. Round-1 bookkeeping feeds dropout's later-round selection."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        deserialized = [self.deserialize_fn(fit_res.parameters) for _, fit_res in results]
        weights = self._compute_weights(deserialized, results)

        aggregator = self.aggregator_cls(models=deserialized, weights=weights)
        aggregated_params = aggregator.aggregate()
        parameters_aggregated = self.serialize_fn(aggregated_params)

        if server_round == 1:
            for client, res in results:
                self.clients_first_round_time[client.cid] = res.metrics.get("running_time", 0.0)
                self.clients_num_examples[client.cid] = res.num_examples

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters_aggregated, metrics_aggregated
