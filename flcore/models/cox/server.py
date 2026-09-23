# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *
# Survival model
# Author: Iratxe Moya
# Date: January 2026
# Project: AI4HF
# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *

# src/server.py
from logging import WARNING
import argparse
import sys, os
import logging
import hashlib
import flwr as fl
from flwr.common.logger import log
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from typing import List, Optional, Tuple, Union, Dict
# from flwr import weighted_loss_avg

import numpy as np
import pickle, json

from flcore.base_strategy import BaseFLStrategy
from flcore.models.cox.model import CoxPHModel
from flcore.models.cox.aggregator import CoxAggregator


logger = logging.getLogger(__name__)

# -------------------------------
# Custom FedAvg Strategy
# -------------------------------

class CustomStrategy(BaseFLStrategy):
    def __init__(self, l1_penalty: float, rounds: int, saving_path :str = '/sandbox/', **kwargs):
        super().__init__(
            aggregator_cls=CoxAggregator,
            serialize_fn=ndarrays_to_parameters,
            deserialize_fn=parameters_to_ndarrays,
            **kwargs,
        )
        self.rounds = rounds
        self.results_history = {}
        self.saving_path = saving_path
        self.l1_penalty = l1_penalty

    def _save_results_history(self):
        """Save the results history to a file."""
        with open(f"{self.saving_path}/history.json", "w") as f:
            json.dump(self.results_history, f)

    def aggregate_fit(self, server_round: int, results, failures):
        """Reuses BaseFLStrategy's merge (deserialize -> weight -> CoxAggregator ->
        re-serialize); adds cox's own "save the global model on the last round"
        side effect, which needs the post-merge params BaseFLStrategy computed."""
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if parameters is None:
            return None, {}

        # --- SAVE GLOBAL MODEL AFTER LAST ROUND ---
        if server_round == self.rounds:
            aggregated_params = parameters_to_ndarrays(parameters)
            print(aggregated_params)
            model = CoxPHModel()
            model.set_parameters(aggregated_params)
            os.makedirs(f"{self.saving_path}/models/", exist_ok=True)
            with open(f"{self.saving_path}/models/cox.pkl", "wb") as f:
                pickle.dump(model, f)

            model_bytes = pickle.dumps(CoxPHModel)
            model_md5 = hashlib.md5(model_bytes).hexdigest()
            self.results_history['MODEL_MD5'] = model_md5

        return parameters, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list,
        failures: list,
    ) -> tuple:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        round_results = {'CLIENTS': {}, 'ROUND_INFO': {}}
        for _, res in results:
            round_results['CLIENTS'][res.metrics['client_id']] = {key: value for key, value in res.metrics.items() if key != 'client_id'}
            round_results['CLIENTS'][res.metrics['client_id']]['num_examples'] = res.num_examples
            round_results['CLIENTS'][res.metrics['client_id']]['1-c_index(loss)'] = res.loss
        

        # Aggregate loss
        loss_aggregated = np.mean([evaluate_res.loss for _, evaluate_res in results])
        round_results['ROUND_INFO']['aggregated_loss'] = loss_aggregated

        # Aggregate custom metrics if aggregation fn was provided

        metrics_aggregated = {}
        for _, res in results:
            for key, value in res.metrics.items():
                if key == 'client_id':
                    continue
                if key not in metrics_aggregated:
                    metrics_aggregated[key] = []
                metrics_aggregated[key].append(value)
        for key in metrics_aggregated:
            metrics_aggregated[key] = np.mean(metrics_aggregated[key])

        round_results['ROUND_INFO']['aggregated_metrics'] = metrics_aggregated
        
        self.results_history[f"ROUND {server_round}"] = round_results
        self.results_history['MODEL_TYPE'] = 'cox'
        self._save_results_history()

        return loss_aggregated, metrics_aggregated

# -------------------------------
# Fit config function
# -------------------------------

def get_fit_config_fn(l1_penalty: float = 0.0):
    def fit_config(rnd: int):
        conf = {"model_type": 'cox', "l1_penalty": l1_penalty}
        return conf
    return fit_config

# -------------------------------
# Get server helper
# -------------------------------

def get_server_and_strategy(
    config
) -> Tuple[fl.server.Server, CustomStrategy]:

    os.makedirs(f"{config['experiment_dir']}", exist_ok=True)

    server = fl.server.Server
    strategy = CustomStrategy(
        config=config,
        on_fit_config_fn=get_fit_config_fn(config['l1_penalty']),
        rounds = config['num_rounds'],
        min_fit_clients = config["min_fit_clients"],
        min_evaluate_clients = config["min_evaluate_clients"],
        min_available_clients=config['num_clients'],
        saving_path=config['experiment_dir'],
        l1_penalty=config['l1_penalty']
    )

    return None, strategy
