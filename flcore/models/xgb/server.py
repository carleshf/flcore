"""
Fully Federated XGBoost - Flower Message-Based Server
"""

import json
import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Callable, Iterable, Any, cast

import numpy as np
import xgboost as xgb

from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MetricRecord,
    RecordDict,
    Parameters,
    FitRes,
    EvaluateRes,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server import Grid
from flwr.server.client_proxy import ClientProxy

from flcore.base_strategy import BaseFLStrategy
from flcore.models.xgb.aggregator import XGBAggregator


def aggregate_metricrecords(
    records: list[RecordDict], weighting_metric_name: str
) -> MetricRecord:
    """Perform weighted aggregation all MetricRecords using a specific key."""
    # Retrieve weighting factor from MetricRecord
    weights: list[float] = []
    for record in records:
        # Get the first (and only) MetricRecord in the record
        metricrecord = next(iter(record.metric_records.values()))
        # Because replies have been checked for consistency,
        # we can safely cast the weighting factor to float
        w = cast(float, metricrecord[weighting_metric_name])
        weights.append(w)

    # Average
    total_weight = sum(weights)
    weight_factors = [w / total_weight for w in weights]

    aggregated_metrics = MetricRecord()
    for record, weight in zip(records, weight_factors, strict=True):
        for record_item in record.metric_records.values():
            # aggregate in-place
            for key, value in record_item.items():
                if key == weighting_metric_name:
                    # We exclude the weighting key from the aggregated MetricRecord
                    continue
                if key not in aggregated_metrics:
                    if isinstance(value, list):
                        aggregated_metrics[key] = [v * weight for v in value]
                    else:
                        aggregated_metrics[key] = value * weight
                else:
                    if isinstance(value, list):
                        current_list = cast(list[float], aggregated_metrics[key])
                        aggregated_metrics[key] = [
                            curr + val * weight
                            for curr, val in zip(current_list, value, strict=True)
                        ]
                    else:
                        current_value = cast(float, aggregated_metrics[key])
                        aggregated_metrics[key] = current_value + value * weight

    return aggregated_metrics

# ==========================================================
# STRATEGY
# ==========================================================

def _xgb_deserialize(parameters) -> bytes:
    """xgb's own quirk: a client's params are wire-encoded as one ndarray of
    raw booster bytes (uint8), not a numeric tensor list -- unwrap straight to
    bytes so XGBAggregator can operate on them directly."""
    ndarrays = parameters_to_ndarrays(parameters)
    return ndarrays[0].tobytes()


def _xgb_serialize(model_bytes: bytes) -> Parameters:
    return ndarrays_to_parameters([np.frombuffer(model_bytes, dtype=np.uint8)])


class FedXgbFullyFederated(BaseFLStrategy):
    """Fully federated XGBoost strategy (bagging or cyclic).

    Reuses BaseFLStrategy's shared configure_fit (dropout -- newly gained, this
    model never had it wired in before) and aggregate_fit (deserialize -> weight
    -- computed but unused, see XGBAggregator -> XGBAggregator.aggregate() ->
    re-serialize, carrying current_model across rounds via
    _aggregator_kwargs/_after_aggregate, same pattern as random_forest's
    server_estimators). aggregate_evaluate is NOT inherited: this model never
    wired evaluate_metrics_aggregation_fn in, so its own hand-rolled proportional
    metrics average is genuinely different behavior, not a redundant
    reimplementation of the FedAvg default -- kept as its own override.
    """

    def __init__(
        self,
        config: dict,
        num_local_rounds: int = 5,
        xgb_params: Dict = None,
        saving_path: str = "./sandbox",
        train_method: str = "bagging",
        fraction_train=1.0,
        fraction_evaluate=1.0,
        **kwargs,
    ):
        # fraction_train/fraction_evaluate accepted-and-discarded for parity
        # with the pre-migration signature (they were never real FedAvg kwargs
        # -- fraction_fit/fraction_evaluate are -- so forwarding them via
        # **kwargs would raise; keep them out of kwargs instead of wiring them
        # in, since that'd be a real behavior change, not a refactor).
        super().__init__(
            config=config,
            aggregator_cls=XGBAggregator,
            serialize_fn=_xgb_serialize,
            deserialize_fn=_xgb_deserialize,
            **kwargs,
        )

        self.train_method = train_method
        self.xgb_params = xgb_params or {}
        self.saving_path = Path(saving_path)
        self.saving_path.mkdir(parents=True, exist_ok=True)

        self.current_model: bytes = b""

        print(f"[FedXgb] Training method: {train_method}")
        print(f"[FedXgb] XGBoost params: {self.xgb_params}")

    # ------------------------------------------------------
    # INITIALIZE
    # ------------------------------------------------------

    def initialize_parameters(self, client_manager):
        """Start with empty model."""
        empty = np.frombuffer(b"", dtype=np.uint8)
        return ndarrays_to_parameters([empty])

    # ------------------------------------------------------
    # AGGREGATE FIT
    # ------------------------------------------------------

    def _aggregator_kwargs(self) -> dict:
        return {"train_method": self.train_method, "current_model": self.current_model}

    def _after_aggregate(self, aggregator) -> None:
        self.current_model = aggregator.updated_current_model

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        print(f"\n[Round {server_round}] Aggregating {len(results)} clients")

        parameters, _ = super().aggregate_fit(server_round, results, failures)
        if parameters is None:
            return None, {}

        self._save_checkpoint(self.current_model, server_round)

        # xgb's own weighted metric aggregation -- not fit_metrics_aggregation_fn
        # based like other models (never wired in for this one), kept as-is.
        metrics_aggregated: Dict[str, Scalar] = {}
        total_examples = sum(fit_res.num_examples for _, fit_res in results)

        for _, fit_res in results:
            if "n_out" in fit_res.metrics:
                self.xgb_params["num_class"] = int(fit_res.metrics["n_out"])

            for key, value in fit_res.metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                metrics_aggregated.setdefault(key, 0.0)
                metrics_aggregated[key] += (
                    value * fit_res.num_examples / total_examples
                )

        print(f"[Round {server_round}] Aggregation done.")

        return parameters, metrics_aggregated

    # ------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        if not results:
            return None, {}

        total_examples = sum(eval_res.num_examples for _, eval_res in results)

        total_loss = sum(
            eval_res.loss * eval_res.num_examples
            for _, eval_res in results
        )

        if total_examples == 0:
            return None, {}

        avg_loss = total_loss / total_examples

        metrics_aggregated: Dict[str, Scalar] = {}

        for _, eval_res in results:
            for key, value in eval_res.metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                metrics_aggregated.setdefault(key, 0.0)
                metrics_aggregated[key] += (
                    value * eval_res.num_examples / total_examples
                )

        print(f"[Round {server_round}] Eval loss: {avg_loss:.4f}")

        return avg_loss, metrics_aggregated

    # ------------------------------------------------------
    # CHECKPOINT
    # ------------------------------------------------------

    def _save_checkpoint(self, model_bytes: bytes, round_num: int):

        if not model_bytes:
            return

        checkpoint_dir = self.saving_path / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        bst = xgb.Booster(params=self.xgb_params)
        bst.load_model(bytearray(model_bytes))

        model_path = checkpoint_dir / f"xgboost_round_{round_num}.json"
        bst.save_model(str(model_path))

        print(f"[Checkpoint] Saved {model_path}")


def get_fit_config_fn(
    num_local_rounds: int,
    train_method: str,
    xgb_params: Dict,
) -> Callable[[int], Dict[str, Any]]:
    """Return a function that returns training configuration."""
    
    def fit_config(server_round: int) -> Dict[str, Any]:
        config = {
            "server_round": server_round,
            "num_local_rounds": num_local_rounds,
            "train_method": train_method,
        }
        # Add XGBoost parameters
        config.update(xgb_params)
        return config
    
    return fit_config

def get_evaluate_config_fn(xgb_params: Dict) -> Callable[[int], Dict[str, Any]]:
    """Return a function that returns evaluation configuration."""
    
    def evaluate_config(server_round: int) -> Dict[str, Any]:
        config = {
            "server_round": server_round,
        }
        config.update(xgb_params)
        return config
    
    return evaluate_config

# ==========================================================
# SERVER FACTORY
# ==========================================================

def get_server_and_strategy(config: dict) -> FedXgbFullyFederated:
    """Create strategy from config dictionary."""

    os.makedirs(config["experiment_dir"], exist_ok=True)

    task = config.get("task", "binary").lower()
    xgb_config = config.get("xgb", {})

    xgb_params = {
        "eta": xgb_config.get("learning_rate", 0.1),
        "max_depth": xgb_config.get("max_depth", 6),
        "tree_method": "hist",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": config.get("seed", 42),
    }

    if task == "binary":
        xgb_params["objective"] = "binary:logistic"
        xgb_params["eval_metric"] = "auc"

    elif task == "multiclass":
        n_out = config.get("n_out")
        xgb_params["objective"] = "multi:softmax"
        xgb_params["eval_metric"] = "mlogloss"
        if n_out is not None and n_out >= 2:
            xgb_params["num_class"] = n_out

    elif task == "regression":
        xgb_params["objective"] = "reg:squarederror"
        xgb_params["eval_metric"] = "rmse"

    train_method = xgb_config.get("train_method", "bagging")  # 'bagging' or 'cyclic'
    num_local_rounds = xgb_config.get("tree_num", 100) // config.get("num_rounds", 10)  # Trees per round
    

    print("\n" + "=" * 60)
    print("Federated XGBoost Configuration")
    print("=" * 60)
    print("Task:", task.upper())
    print("Train method:", xgb_config.get("train_method", "bagging"))
    print("Rounds:", config.get("num_rounds"))
    print("Clients:", config.get("num_clients"))
    print("XGBoost params:", xgb_params)
    print("=" * 60 + "\n")

    strategy = FedXgbFullyFederated(
        config=config,
        train_method=train_method,
        num_local_rounds=num_local_rounds,
        xgb_params=xgb_params,
        saving_path=config['experiment_dir'],
        min_fit_clients=config.get('min_fit_clients', config['num_clients']),
        min_evaluate_clients=config.get('min_evaluate_clients', config['num_clients']),
        min_available_clients=config.get('min_available_clients', config['num_clients']),
        on_fit_config_fn=get_fit_config_fn(num_local_rounds, train_method, xgb_params),
        on_evaluate_config_fn=get_evaluate_config_fn(xgb_params),
        fraction_train=1.0,
        fraction_evaluate=1.0
    )

    return None, strategy