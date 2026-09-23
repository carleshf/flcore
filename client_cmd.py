import sys
import os
import glob

import time
from pathlib import Path
import flwr as fl
import yaml
import argparse
import json
import logging
#import grpc

import flcore.datasets as datasets
from flcore.utils import StreamToLogger, GetModelClient, CheckClientConfig, survival_models_list, log_detailed_error
from flcore.cli_args import (
    add_common_args,
    add_client_only_args,
    add_random_forest_args,
    add_xgb_args,
    add_linear_model_args,
    add_nn_args,
    add_survival_args,
    CLIENT_MODEL_GROUPS,
    warn_unused_args,
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Reads parameters from command line.")
    add_common_args(parser)
    add_client_only_args(parser)
    add_random_forest_args(parser)
    add_xgb_args(parser)
    add_linear_model_args(parser)
    add_nn_args(parser)
    add_survival_args(parser)

    args = parser.parse_args()
    config = vars(args)
    warn_unused_args(config, config["model"], CLIENT_MODEL_GROUPS)
    try:
        config = CheckClientConfig(config)
    except Exception as e:
        log_detailed_error("Client Configuration Verification", e, config)
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(1)

    # Create sandbox log file path
    sandbox_log_file = Path(os.path.join(config["sandbox_path"], "log_client.txt"))

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
    logging.info("Starting Flower client...")

#### PODRIAMOS QUITAR ESTO DE PRODUCTION MODE; NO TIENE NINGUN SENTIDO
    #model = config["model"]
    if config["production_mode"] == "True":
        node_name = os.getenv("NODE_NAME")
#        num_client = int(node_name.split("_")[-1])
        data_path = os.getenv("DATA_PATH")
        ca_cert = Path(os.path.join(config["certs_path"],"rootCA_cert.pem"))
        root_certificate = Path(f"{ca_cert}").read_bytes()
#        root_certificate = ca_cert
#        root_certificate =( Path(os.path.join(config["certs_path"],"rootCA_cert.pem")).read_bytes(),
#            Path(os.path.join(config["certs_path"],"rootCA_cert.pem")).read_bytes(),
#            Path(os.path.join(config["certs_path"],"rootCA_key.pem")).read_bytes() )

        root_cert = Path(os.path.join(config["certs_path"],"rootCA_cert.pem")).read_bytes()
        client_cert = Path(os.path.join(config["certs_path"],config["node_name"]+"_client_cert.pem")).read_bytes()
        client_key = Path(os.path.join(config["certs_path"],config["node_name"]+"_client_key.pem")).read_bytes()

        #ssl_credentials = grpc.ssl_channel_credentials(
        #    root_certificates=root_cert,  # Certificado raíz del servidor
        #    private_key=client_key,  # Clave privada del cliente
        #    certificate_chain=client_cert  # Certificado del cliente
        #)

        central_ip = os.getenv("FLOWER_CENTRAL_SERVER_IP")
        central_port = os.getenv("FLOWER_CENTRAL_SERVER_PORT")
        #channel = grpc.secure_channel(f"{central_ip}:{central_port}", ssl_credentials)

    else:
        data_path = config["data_path"]
        root_certificate = None
        central_ip = "LOCALHOST"
        central_port = config["local_port"]
#        if len(sys.argv) == 1:
#            raise ValueError("Please provide the client id when running in simulation mode")
#        num_client = int(sys.argv[1])

# *******************************************************************************************
# Aquí lo correcto es cargar todo como instancias de dataloader de torch
num_client = 0 # config["client_id"]
try:
    data = datasets.load_dataset(config, num_client)
except Exception as e:
    log_detailed_error(
        "Client Dataset Loading",
        e,
        config=config,
        data_path=config.get("data_id") or config.get("data_path")
    )
    sys.stderr.flush()
    sys.stdout.flush()
    os._exit(1)

try:
    client = GetModelClient(config, data)
except Exception as e:
    X_train_diag, y_train_diag = None, None
    if data and isinstance(data, tuple) and len(data) >= 1:
        if isinstance(data[0], tuple) and len(data[0]) >= 2:
            X_train_diag, y_train_diag = data[0][0], data[0][1]
    log_detailed_error(
        "Client Model Setup / Initialization",
        e,
        config=config,
        X=X_train_diag,
        y=y_train_diag
    )
    sys.stderr.flush()
    sys.stdout.flush()
    os._exit(1)
# *******************************************************************************************
for attempt in range(3):
    try:
        if isinstance(client, fl.client.NumPyClient):
            fl.client.start_numpy_client(
                server_address=f"{central_ip}:{central_port}",
                #credentials=ssl_credentials,
                root_certificates=root_certificate,
                client=client,
                #channel=channel,
            )
        else:
            fl.client.start_client(
                server_address=f"{central_ip}:{central_port}",
                # credentials=ssl_credentials,
                root_certificates=root_certificate,
                client=client,
                #channel=channel,
            )
        break  # Si todo salió bien, salimos del bucle
    except Exception as e:
        print(f"Attempt {attempt + 1} failed: {e}")
        if attempt < 2:
            time.sleep(2)  # Espera un poco antes de reintentar
        else:
            print("All connection attempts failed.")
            X_train_diag, y_train_diag = None, None
            if 'data' in locals() and data and isinstance(data, tuple) and len(data) >= 1:
                if isinstance(data[0], tuple) and len(data[0]) >= 2:
                    X_train_diag, y_train_diag = data[0][0], data[0][1]
            log_detailed_error("Flower Client Start / Execution Loop", e, config=config, X=X_train_diag, y=y_train_diag)
            sys.stderr.flush()
            sys.stdout.flush()
            os._exit(1)

sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
