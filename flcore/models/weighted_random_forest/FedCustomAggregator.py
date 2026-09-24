import time

from flcore.base_strategy import BaseFLStrategy
from flcore.models.random_forest.aggregator import RandomForestAggregator
from flcore.models.weighted_random_forest.aggregator import (
    WeightedRandomForestBroadcastAggregator,
    broadcast_deserialize,
    broadcast_serialize,
)
from flcore.models.weighted_random_forest.utils import get_model
from flcore.serialization_funs import serialize_RF, deserialize_RF


class FedCustom(BaseFLStrategy):
    """Reuses BaseFLStrategy's shared configure_fit (dropout) and aggregate_fit.
    Which aggregator_cls/serialize_fn/deserialize_fn get wired in depends on
    config["wrf_aggregation_mode"] (default "server_merge", see
    flcore/cli_args.py and flcore/models/weighted_random_forest/aggregator.py
    for what each mode does):

    - server_merge: same shape as random_forest's own migration -- merged
      model, server_estimators/server_estimators_weights carried across
      rounds via _aggregator_kwargs/_after_aggregate.
    - client_ensemble: no server-side merge; num_examples threaded into
      _aggregator_kwargs via `results` for WeightedRandomForestBroadcastAggregator,
      no round-to-round state on the Strategy at all (client.py accumulates
      the ensemble across rounds on its own side instead).

    Always tolerates client failures (accept_failures=True, matching the
    pre-migration behavior) -- this model has never refused to aggregate a
    partial round.
    """

    def __init__(self, config: dict, **kwargs):
        self.wrf_aggregation_mode = config.get("wrf_aggregation_mode", "server_merge")
        if self.wrf_aggregation_mode == "server_merge":
            aggregator_cls = RandomForestAggregator
            serialize_fn = serialize_RF
        else:
            aggregator_cls = WeightedRandomForestBroadcastAggregator
            serialize_fn = broadcast_serialize

        kwargs["accept_failures"] = True
        super().__init__(
            config=config,
            aggregator_cls=aggregator_cls,
            serialize_fn=serialize_fn,
            # Clients always send their own single model the standard way
            # regardless of mode -- only the broadcast/aggregated result's
            # codec differs (serialize_fn above).
            deserialize_fn=deserialize_RF,
            **kwargs,
        )
        self.server_estimators = None
        self.server_estimators_weights = None
        self.time_server_round = time.time()

    def evaluate(self, server_round: int, parameters):
        """Overrides FedAvg's own evaluate(): its stock parameters_to_ndarrays
        assumes flwr's plain tensor wire format, but this model's parameters
        are whole pickled model objects (or, in client_ensemble mode, a
        pickled tuple list) -- not currently reachable in practice (no
        evaluate_fn is wired for this model, see server.py), kept for parity
        with the pre-migration code in case one ever is."""
        if self.evaluate_fn is None:
            return None
        deserialize = deserialize_RF if self.wrf_aggregation_mode == "server_merge" else broadcast_deserialize
        parameters_ndarrays = deserialize(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def _aggregator_kwargs(self, results) -> dict:
        if self.wrf_aggregation_mode == "server_merge":
            return {
                "model_factory": lambda: get_model(self.config["balanced"], random_state=self.config.get("seed", 42)),
                "previous_estimators": self.server_estimators,
                "previous_estimator_weights": self.server_estimators_weights,
            }
        return {"num_examples": [res.num_examples for _, res in results]}

    def _after_aggregate(self, aggregator) -> None:
        if self.wrf_aggregation_mode == "server_merge":
            self.server_estimators = aggregator.updated_estimators
            self.server_estimators_weights = aggregator.updated_estimator_weights

    def aggregate_fit(self, server_round: int, results, failures):
        parameters, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        if parameters is None:
            return None, {}

        elapsed_time = time.time() - self.time_server_round
        self.time_server_round = time.time()
        print(f"Elapsed time: {elapsed_time} for round {server_round}")

        return parameters, metrics_aggregated
