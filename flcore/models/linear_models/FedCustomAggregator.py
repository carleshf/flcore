import time

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
import joblib

from flcore.base_strategy import BaseFLStrategy
from flcore.models.linear_models.aggregator import LinearModelAggregator


class FedCustom(BaseFLStrategy):
    """Reuses BaseFLStrategy's shared configure_fit (dropout) and aggregate_fit
    (deserialize -> weight via computeSmoothedWeights -> LinearModelAggregator ->
    re-serialize); adds this model's own per-round side effects (checkpoint
    dump, cumulative training-time tracking) on top of the post-merge params
    BaseFLStrategy already computed."""

    def __init__(self, config: dict, checkpoint_dir=None, **kwargs):
        super().__init__(
            config=config,
            aggregator_cls=LinearModelAggregator,
            serialize_fn=ndarrays_to_parameters,
            deserialize_fn=parameters_to_ndarrays,
            **kwargs,
        )
        self.checkpoint_dir = checkpoint_dir
        self.time_server_round = time.time()
        self.accum_time = 0

    def aggregate_fit(self, server_round: int, results, failures):
        parameters, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        if parameters is None:
            return None, {}

        weights_aggregated = parameters_to_ndarrays(parameters)
        joblib.dump(weights_aggregated, f"{self.checkpoint_dir}/round_{server_round}_weights.joblib")

        elapsed_time = time.time() - self.time_server_round
        self.accum_time += elapsed_time
        self.time_server_round = time.time()
        print(f"Elapsed time: {elapsed_time} for round {server_round}")
        metrics_aggregated["training_time [s]"] = self.accum_time

        with open("server_results.txt", "a") as f:
            f.write(f"Accumulated Time: {self.accum_time} for round {server_round}\n")

        return parameters, metrics_aggregated
