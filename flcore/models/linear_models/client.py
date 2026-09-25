from scipy.special import softmax
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
import time
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import KFold, StratifiedShuffleSplit, train_test_split
import warnings
import flcore.models.linear_models.utils as utils
import flwr as fl
from sklearn.metrics import log_loss
from flcore.performance import measurements_metrics, get_metrics
from flcore.metrics import calculate_metrics
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import json
import joblib
from pathlib import Path
from flcore.data_sources import build_checkpoint_metadata

# Define Flower client
class MnistClient(fl.client.NumPyClient):
    def __init__(self, data,config):
        self.config = config
        self.node_name = config["node_name"]
        # Load data
        (self.X_train, self.y_train), (self.X_test, self.y_test) = data

        if self.config["task"] == "classification":
            stratify=self.y_train
        else:
            stratify = None

        # Create train and validation split
#        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
#                self.X_train,
#                self.y_train,
#                test_size=config["test_size"],
#                random_state=config["seed"],
#                stratify=stratify
#                )
#                stratify=self.y_train)

        # #Only use the standardScaler to the continous variables
        # scaled_features_train = StandardScaler().fit_transform(self.X_train.values)
        # scaled_features_train = pd.DataFrame(scaled_features_train, index=self.X_train.index, columns=self.X_train.columns)
        # self.X_train = scaled_features_train

        # #Only use the standardScaler to the continous variables. 
        # scaled_features = StandardScaler().fit_transform(self.X_test.values)
        # scaled_features_df = pd.DataFrame(scaled_features, index=self.X_test.index, columns=self.X_test.columns)
        # self.X_test = scaled_features_df

        self.model = utils.get_model(config)
        self.round_time = 0
        self.first_round = True
        self.round = 0
        self.personalize = True
        # Setting initial parameters, akin to model.compile for keras models
        utils.set_initial_params(self.model, config)

    def get_parameters(self, config):  # type: ignore
        #compute the feature selection
        #We perform it from the one called by the server
        #at the begining to start the parameters
        # if(bool(config) == False):
        #         fs = SelectKBest(f_classif, k= self.n_features).fit(self.X_train, self.y_train)
        #         index_features = fs.get_support()
        #         self.model.features = index_features
        return utils.get_model_parameters(self.model)

    def fit(self, parameters, config):  # type: ignore
        try:
            utils.set_model_params(self.model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                #To implement the center dropout, we need the execution time
                start_time = time.time()
                self.model.fit(self.X_train, self.y_train)
                # self.model.fit(self.X_train.loc[:, parameters[2].astype(bool)], self.y_train)
                # y_pred = self.model.predict(self.X_test.loc[:, parameters[2].astype(bool)])
                y_pred = self.model.predict(self.X_test)

                metrics = calculate_metrics(self.y_test, y_pred,self.config)
                # Add 'personalized' to the metrics to identify them
                metrics = {f"personalized {key}": metrics[key] for key in metrics}
                self.round_time = (time.time() - start_time)
                metrics["running_time"] = self.round_time

            if self.first_round:
                local_model = utils.get_model(self.config)
                utils.set_initial_params(self.model, self.config)
                local_model.fit(self.X_train, self.y_train)
                y_pred = local_model.predict(self.X_test)
                local_metrics = calculate_metrics(self.y_test, y_pred,self.config)
                #Add 'local' to the metrics to identify them
                local_metrics = {f"local {key}": local_metrics[key] for key in local_metrics}
                metrics.update(local_metrics)
                self.first_round = False

            if self.round % self.config["save_every_n_rounds"] == 0:
                self.save_model()

            elapsed_time = (time.time() - start_time)
            metrics["running_time"] = elapsed_time

            print(f"num_client {self.node_name} has an elapsed time {elapsed_time}")
            print(f"Training finished for round {self.round}")

            self.round += 1
            return utils.get_model_parameters(self.model), len(self.X_train), metrics
        except Exception as e:
            from flcore.utils import log_detailed_error
            log_detailed_error("Model Fitting (Local Training)", e, config=getattr(self, "config", None), X=getattr(self, "X_train", None), y=getattr(self, "y_train", None))
            raise e

    def evaluate(self, parameters, config):
        try:
            utils.set_model_params(self.model, parameters)
            # Calculate validation set metrics
            pred = self.model.predict(self.X_test)
            y_pred = pred
            metrics = calculate_metrics(self.y_test, y_pred, self.config)
            if self.config["task"] == "classification":
                if self.config["n_out"] > 1: # Multivariable
                    if hasattr(self.model, "predict_proba"):
                        y_score = self.model.predict_proba(self.X_test)
                        loss = log_loss(self.y_test,y_score,labels=np.arange(self.config["n_out"]))
                    else:
                        decision = self.model.decision_function(self.X_test)
                        y_score = softmax(decision, axis=1)
                        loss = log_loss(
                        self.y_test,
                        y_score,
                        labels=np.arange(self.config["n_out"])
                        )

                elif self.config["n_out"] == 1: # Binario
                    if hasattr(self.model, "predict_proba"):
                        loss = log_loss(
                            self.y_test,
                            self.model.predict_proba(self.X_test)
                        )
                    else:
                        loss = 1.0 - accuracy_score(
                            self.y_test,
                            y_pred
                        )

            elif self.config["task"] == "regression":
                loss = mean_squared_error(self.y_test, y_pred)

            metrics["round_time [s]"] = self.round_time
            # No tiene sentido agregar el client ID
            # metrics["client_id"] = self.node_name

    #        print(f"Client {self.node_name} Evaluation after aggregated model: {metrics['balanced_accuracy']}")

            return loss, len(y_pred),  metrics
        except Exception as e:
            from flcore.utils import log_detailed_error
            log_detailed_error("Model Evaluation (Local Validation)", e, config=getattr(self, "config", None), X=getattr(self, "X_test", None), y=getattr(self, "y_test", None))
            raise e

    def save_model(self):
        save_path = Path(self.config["experiment_dir"])/"models"
        save_path.mkdir(parents=True, exist_ok=True)
        model_name = self.config["model"]+"_"+self.config["task"]+"_round_"+str(self.round)
        model_path = save_path / f"{model_name}_model.joblib"
        joblib.dump(self.model, model_path)

        metadata = build_checkpoint_metadata(self.config, getattr(self, "last_metrics", None))

        metadata_path = save_path / f"{model_name}_model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        #print(f"Model and metadata saved for inference at {save_path}")

def get_client(config,data) -> fl.client.Client:
    return MnistClient(data,config)
    # # Start Flower client
    # fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=MnistClient())
