import os
import sys
import yaml
import json
import numpy
import logging
import warnings
import argparse
import flwr as fl
from pathlib import Path

from flcore.utils import StreamToLogger, CheckServerConfig, GetModelServerStrategy, log_detailed_error
from flcore.cli_args import (
    add_common_args,
    add_server_only_args,
    add_random_forest_args,
    add_xgb_args,
    add_cox_args,
    SERVER_MODEL_GROUPS,
    warn_unused_args,
)

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reads parameters from command line.")
    add_common_args(parser)
    add_server_only_args(parser)
    add_random_forest_args(parser)
    add_xgb_args(parser)
    add_cox_args(parser)

    args = parser.parse_args()
    config = vars(args)
    warn_unused_args(config, config["model"], SERVER_MODEL_GROUPS)
    try:
        config = CheckServerConfig(config)
    except Exception as e:
        log_detailed_error("Server Configuration Verification", e, config)
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(1)

    # Create sandbox log file path
# Originalmente estaba asi:
#    sandbox_log_file = Path(os.path.join("/sandbox", "log_server.txt"))
# Modificado
    sandbox_log_file = Path(os.path.join(config["sandbox_path"], "log_server.txt"))

    # Set up the file handler (writes to file)
    file_handler = logging.FileHandler(sandbox_log_file)
    file_handler.setLevel(logging.DEBUG)

    # Set up the console handler (writes to Docker logs via stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s')

    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    # Get the root logger and configure it
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Change default level to INFO
    logger.handlers = []  # Clear any default handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Silence noisy dependencies
#    logging.getLogger("flwr").setLevel(logging.WARNING)
    logging.getLogger("flwr").setLevel(logging.ERROR)

    # Create two sub-loggers
    stdout_logger = logging.getLogger("STDOUT")
    stderr_logger = logging.getLogger("STDERR")

    # Redirect standard output and error to logging
    sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
    sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)

    # Now you can use logging in both places
    logging.info("Starting Flower server...")

    if config["production_mode"] == "True":
        #data_path = ""
        central_ip = os.getenv("FLOWER_CENTRAL_SERVER_IP")
        central_port = os.getenv("FLOWER_CENTRAL_SERVER_PORT")

        ca_cert = Path(os.path.join("/certs","rootCA_cert.pem"))
        server_cert =  Path(os.path.join("/certs","server_cert.pem"))
        server_key =  Path(os.path.join("/certs","server_key.pem"))

        certificates = (
            Path(f"{ca_cert}").read_bytes(),
            Path(f"{server_cert}").read_bytes(),
            Path(f"{server_key}").read_bytes(),
        )
#            Path('.cache/certificates/rootCA_cert.pem').read_bytes(),
#            Path('.cache/certificates/server_cert.pem').read_bytes(),
#            Path('.cache/certificates/server_key.pem').read_bytes(),
    else:
        #data_path = config["data_path"]
        central_ip = "LOCALHOST"
        central_port = config["local_port"]
        certificates = None


    # Create experiment directory
    experiment_dir = Path(os.path.join(config["sandbox_path"],config["experiment_name"]))
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config["experiment_dir"] = experiment_dir

    # Checkpoint directory for saving the model
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)


    # Checkpoint directory for saving the model
    #checkpoint_dir = config["experiment_dir"] + "/checkpoints"
    #checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # # History directory for saving the history
    # history_dir = experiment_dir / "history"
    # history_dir.mkdir(parents=True, exist_ok=True)

    try:
        server, strategy = GetModelServerStrategy(config)
    except Exception as e:
        log_detailed_error("Server Strategy / Model Setup", e, config)
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(1)

    # Start Flower server for three rounds of federated learning
    try:
        history = fl.server.start_server(
            server_address=f"{central_ip}:{central_port}",
            config=fl.server.ServerConfig(num_rounds=config["num_rounds"], round_timeout=None ),
            server=server,
            strategy=strategy,
            certificates = certificates,
        )
    except Exception as e:
        log_detailed_error("Flower Server Start / Execution Loop", e, config)
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(1)
    # # Save the model and the history
    # filename = os.path.join( checkpoint_dir, 'final_model.pt' )
    # joblib.dump(model, filename)
    # Save the history as a yaml file
    print(history)
    """
    with open(experiment_dir / "metrics.txt", "w") as f:
        f.write(f"Results of the experiment {config['experiment']['name']}\n")
        f.write(f"Model: {config['model']}\n")
        f.write(f"Data: {config['dataset']}\n")
        f.write(f"Number of clients: {config['num_clients']}\n")

        # selection_metric = 'val ' + config['checkpoint_selection_metric']
        selection_metric = config['checkpoint_selection_metric']
        # Get index of tuple of the best round
        best_round = int(numpy.argmax([round[1] for round in history.metrics_distributed[selection_metric]]))
        training_time = history.metrics_distributed_fit['training_time [s]'][-1][1]
        f.write(f"Total training time: {training_time:.2f} [s] \n")
        f.write(f"Best checkpoint based on {selection_metric} after round: {best_round}\n\n")
        print(f"Best checkpoint based on {selection_metric} after round: {best_round}\n\n")

        f.write(f"\nAggregated results:\n\n")

        # best_round = best_round - 1
        per_client_values = {}
        for metric in history.metrics_distributed:
            metric_value = history.metrics_distributed[metric][best_round][1]
            if type(metric_value) in [int, float, numpy.float64]:
                f.write(f"{metric} {metric_value:.4f} \n")
            else:
                for per_client_metric_value in metric_value:
                    metric = metric.replace("per client ", "")
                    if metric not in per_client_values:
                        per_client_values[metric] = []
                    per_client_values[metric].append(round(per_client_metric_value, 3))
        
        f.write(f"\n\nPer client results:\n\n")
        for metric in per_client_values:
            f.write(f"{metric} {per_client_values[metric]} \n")
        
        f.write(f"\n\nHeld out set evaluation:\n\n")
        for metric in history.metrics_centralized:
            # print(f"Len of centralized metric {metric} ", len(history.metrics_centralized[metric]))
            if len(history.metrics_centralized[metric]) == 1:
                metric_value = history.metrics_centralized[metric][0][1]
            else:
                metric_value = history.metrics_centralized[metric][best_round][1]
            if type(metric_value) in [int, float, numpy.float64]:
                f.write(f"{metric} {metric_value:.4f} \n")
# Compile the results
compile_results(experiment_dir)
"""

dict_history = {}
history = history.__dict__
for logs in history.keys():
    if isinstance(history[logs], list):
        history[logs] = [float(loss) for (round, loss) in history[logs]]
    if isinstance(history[logs], dict):
        for metric in history[logs]:
            extracted_values = [value for (round, value) in history[logs][metric]]
            if isinstance(extracted_values[0], list):
                # Convert list elements to float
                extracted_values = [[float(value) for value in sublist] for sublist in extracted_values]
            else:
                extracted_values = [float(value) for value in extracted_values]
            history[logs][metric] = extracted_values


with open(experiment_dir / "history.yaml", "w") as f:
    yaml.dump(history, f)

