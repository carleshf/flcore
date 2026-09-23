import time

from flcore.base_strategy import BaseFLStrategy
from flcore.models.random_forest.aggregator import RandomForestAggregator
from flcore.serialization_funs import serialize_RF, deserialize_RF


class FedCustom(BaseFLStrategy):
    """Reuses BaseFLStrategy's shared configure_fit (dropout) and aggregate_fit
    (deserialize -> weight -> RandomForestAggregator -> re-serialize, carrying
    server_estimators/server_estimators_weights across rounds via
    _aggregator_kwargs/_after_aggregate); adds this model's own per-round side
    effect (cumulative training-time tracking) on top of the post-merge params
    BaseFLStrategy already computed.

    Always tolerates client failures (accept_failures=True, matching the
    pre-migration behavior) -- this model has never refused to aggregate a
    partial round.
    """

    def __init__(self, config: dict, **kwargs):
        kwargs["accept_failures"] = True
        super().__init__(
            config=config,
            aggregator_cls=RandomForestAggregator,
            serialize_fn=serialize_RF,
            deserialize_fn=deserialize_RF,
            **kwargs,
        )
        self.server_estimators = None
        self.server_estimators_weights = None
        self.time_server_round = time.time()
        self.accum_time = 0

    def evaluate(self, server_round: int, parameters):
        """Overrides FedAvg's own evaluate(): its stock parameters_to_ndarrays
        assumes flwr's plain tensor wire format, but this model's parameters
        are whole pickled model objects (serialize_RF/deserialize_RF) -- not
        currently reachable in practice (no evaluate_fn is wired for this model,
        see server.py), kept for parity with the pre-migration code in case one
        ever is."""
        if self.evaluate_fn is None:
            return None
        parameters_ndarrays = deserialize_RF(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def _aggregator_kwargs(self) -> dict:
        return {
            "config": self.config,
            "previous_estimators": self.server_estimators,
            "previous_estimator_weights": self.server_estimators_weights,
        }

    def _after_aggregate(self, aggregator) -> None:
        self.server_estimators = aggregator.updated_estimators
        self.server_estimators_weights = aggregator.updated_estimator_weights

    def aggregate_fit(self, server_round: int, results, failures):
        parameters, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        if parameters is None:
            return None, {}

        elapsed_time = time.time() - self.time_server_round
        self.accum_time += elapsed_time
        self.time_server_round = time.time()
        print(f"Elapsed time: {elapsed_time} for round {server_round}")
        metrics_aggregated["training_time [s]"] = self.accum_time

        with open("server_results.txt", "a") as f:
            f.write(f"Accumulated Time: {self.accum_time} for round {server_round}\n")

        return parameters, metrics_aggregated
