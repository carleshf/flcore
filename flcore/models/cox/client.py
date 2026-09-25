# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *
# Survival model
# Author: Iratxe Moya
# Date: January 2026
# Project: AI4HF
# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *

# src/client/client.py
"""
Federated Survival Analysis Flower client.
Supports multiple model types (Cox PH, RSF, GBS) via external model factory.

Usage:
    python client.py
"""

import os
import sys
import json
import time
import argparse
import flwr as fl
from typing import Dict
from pathlib import Path

from flcore.models.cox.model import CoxPHModel
from flcore.models.cox.data_formatter import get_numpy
from flcore.data_sources import build_checkpoint_metadata


# -------------------------------
# Flower client definition
# -------------------------------

class FLClient(fl.client.NumPyClient):
    def __init__(self, local_data, config):
        self.config = config
        self.model_wrapper = None  # will be set later
        self.local_data = local_data
        self.id = config["node_name"]
        self.saving_path = config["experiment_dir"]
        self.round = 0
        os.makedirs(f"{self.saving_path}", exist_ok=True)
        os.makedirs(f"{self.saving_path}/models/", exist_ok=True)

    def get_parameters(self, config=None):
        if self.model_wrapper is None:
            return []
        return self.model_wrapper.get_parameters()

    def fit(self, parameters, config):
        try:
            # Get model type from server
            start_time = time.time()
            model_kwargs = {k: v for k, v in config.items() if k != "model_type"}
            if self.model_wrapper is None:
                self.model_wrapper = CoxPHModel(**model_kwargs)
                print(f"[Client] Initialized model type from server: cox")

            if parameters:
                self.model_wrapper.set_parameters(parameters)

            data = self.local_data
            self.model_wrapper.fit(data)

            params = self.get_parameters()
            num_examples = data.get("num_examples", len(data.get("X", [])) if "X" in data else len(data.get("df")))

            if self.round % self.config["save_every_n_rounds"] == 0:
                self.save_model()

            elapsed_time = (time.time() - start_time)
            metrics = {"running_time": elapsed_time}

            print(f"num_client {self.id} has an elapsed time {elapsed_time}")
            print(f"Training finished for round {self.round}")
            self.round += 1
            return params, num_examples, metrics

        except Exception as e:
            from flcore.utils import log_detailed_error
            log_detailed_error(
                "Model Fitting (Local Training)",
                e,
                config=getattr(self, "config", None),
                X=self.local_data.get("X") if isinstance(self.local_data, dict) else None,
                y=self.local_data.get("y") if isinstance(self.local_data, dict) else None
            )
            raise e

    def evaluate(self, parameters, config):
        try:
            model_kwargs = {k: v for k, v in config.items() if k != "model_type"}
            if self.model_wrapper is None:
                self.model_wrapper = CoxPHModel(**model_kwargs)
                print(f"[Client] Initialized model type from server (evaluate): cox")

            if parameters:
                self.model_wrapper.set_parameters(parameters)

            data = self.local_data
            metrics = self.model_wrapper.evaluate(data)
            metrics['client_id'] = self.id

            num_examples = data.get("num_examples", len(data.get("X", [])) if "X" in data else len(data.get("df")))
            return 1 - metrics['c_index'], num_examples, metrics
        except Exception as e:
            from flcore.utils import log_detailed_error
            log_detailed_error(
                "Model Evaluation (Local Validation)",
                e,
                config=getattr(self, "config", None),
                X=self.local_data.get("X_test") if isinstance(self.local_data, dict) else None,
                y=self.local_data.get("y_test") if isinstance(self.local_data, dict) else None
            )
            raise e

    def save_model(self):
        save_path = Path(self.config["experiment_dir"])/"models"
        save_path.mkdir(parents=True, exist_ok=True)
        model_name = self.config["model"]+"_"+self.config["task"]+"_round_"+str(self.round)
        model_path = save_path / f"{model_name}_model.pkl"        
        self.model_wrapper.save_model(model_path)

        metadata = build_checkpoint_metadata(self.config, getattr(self, "last_metrics", None))

        metadata_path = save_path / f"{model_name}_model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        #print(f"Model and metadata saved for inference at {save_path}")

def get_client(config, data) -> fl.client.Client:
    (X_train, y_train), (X_test, y_test), time, event = data
    local_data = get_numpy(X_train, y_train, X_test, y_test, time, event)
    return FLClient(local_data, config)
