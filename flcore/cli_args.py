"""Single source of truth for the CLI arguments server_cmd.py and client_cmd.py
each used to define independently (which is how task/model/balanced/dropout_method
ended up with mismatched defaults between the two -- see CLAUDE.md Sec 5.4).

Arguments are grouped into functions that just call parser.add_argument(...);
server_cmd.py and client_cmd.py each build their parser by calling the groups
they need. Both parsers stay permissive (every group's flags are always
registered, regardless of --model) so passing an irrelevant flag never errors --
instead, call warn_unused_args() after parsing to print which explicitly-passed
flags don't apply to the chosen model and will be ignored.
"""
import argparse
import sys


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=str, default=None, help="Model to train")
    parser.add_argument("--task", type=str, default=None, help="Task to train")
    parser.add_argument("--num_rounds", type=int, default=50, help="Number of federated iterations")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--sandbox_path", type=str, default="/sandbox", help="Sandbox path to use")
    parser.add_argument("--local_port", type=int, default=8081, help="Local port")
    parser.add_argument(
        "--testing_mode",
        action="store_true",
        help="Connect via LOCALHOST/--local_port instead of production env vars "
        "(FLOWER_CENTRAL_SERVER_IP/PORT, DATA_PATH, NODE_NAME). A real bool, not a string -- pass "
        "the flag to enable, omit it for production (the default). Independent of --enable_certs.",
    )
    parser.add_argument(
        "--enable_certs",
        action="store_true",
        help="Require and load TLS certificates (server: /certs; client: --certs_path). Independent "
        "of --testing_mode -- omit to skip TLS entirely (the default); when passed, missing cert "
        "files are a fatal error rather than silently proceeding without TLS.",
    )
    parser.add_argument("--experiment_name", type=str, default="experiment_1", help="Experiment directory")
    # Recomputed by CheckClientConfig from metadata.json; kept as real CLI args on
    # both entry points for parity with local debugging.
    parser.add_argument("--n_feats", type=int, default=0, help="Number of input features")
    parser.add_argument("--n_out", type=int, default=0, help="Number of output features")


def add_server_only_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num_clients", type=int, default=1, help="Number of clients")
    parser.add_argument("--min_fit_clients", type=int, default=0, help="Minimum number of fit clients")
    parser.add_argument("--min_evaluate_clients", type=int, default=0, help="Minimum number of evaluate clients")
    parser.add_argument("--min_available_clients", type=int, default=0, help="Minimum number of available clients")
    parser.add_argument("--strategy", type=str, default="FedAvg", help="Metrics")
    parser.add_argument("--smooth_method", type=str, default="EqualVoting", help="Weight smoothing")
    parser.add_argument("--smoothing_strenght", type=float, default=0.5, help="Smoothing strenght")
    parser.add_argument("--dropout_method", type=str, default="None", help="Determines if dropout is used")
    parser.add_argument("--dropout_percentage", type=float, default=0.0, help="Ratio of dropout nodes")
    parser.add_argument("--checkpoint_selection_metric", type=str, default="precision", help="Metric used for checkpoints")
    parser.add_argument("--metrics_aggregation", type=str, default="weighted_average", help="Metrics")


def add_client_only_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--node_name", type=str, default="./", help="Node name for certificates")
    parser.add_argument("--certs_path", type=str, default="/certs", help="Certificates path")
    parser.add_argument("--data_path", type=str, default="/data", help="Data path")
    parser.add_argument("--dataset", type=str, default="dt4h_format", help="Dataloader to use")
    parser.add_argument("--data_id", type=str, default="data_id.parquet", help="Dataset ID")
    parser.add_argument("--normalization_method", type=str, default="IQR", help="Type of normalization: IQR STD MIN_MAX")
    parser.add_argument("--train_labels", type=str, nargs="+", default=[], help="Feature columns for training")
    parser.add_argument("--target_labels", type=str, nargs="+", default=[], help="Target columns")
    parser.add_argument("--train_size", type=float, default=0.7, help="Fraction of dataset to use for training. [0,1)")
    parser.add_argument("--test_size", type=float, default=0.1, help="Fraction of dataset to use for testing. [0,1)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate when needed")
    parser.add_argument("--device", type=str, default="cpu", help="Device for training, CPU, GPU")
    parser.add_argument("--local_epochs", type=int, default=10, help="Number of local epochs to train in each round")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size to train")
    parser.add_argument("--penalty", type=str, default="none", help="Penalties: none, l1, l2, elasticnet, smooth l1")
    parser.add_argument("--save_every_n_rounds", type=int, default=1, help="Save model checkpoints every N rounds")


def add_random_forest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--balanced", type=str, default="True", help="Balanced Random Forest: True or False")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of estimators")
    parser.add_argument("--max_depth", type=int, default=2, help="Max depth")
    parser.add_argument("--class_weight", type=str, default="balanced", help="Class weight")
    parser.add_argument("--levelOfDetail", type=str, default="DecisionTree", help="Level of detail (weighted_random_forest, client_ensemble mode only)")
    parser.add_argument("--regression_criterion", type=str, default="squared_error", help="Criterion for training")
    parser.add_argument(
        "--wrf_aggregation_mode",
        type=str,
        default="server_merge",
        choices=["server_merge", "client_ensemble"],
        help="weighted_random_forest only: 'server_merge' merges all clients' trees into one "
        "global forest server-side (like random_forest, default); 'client_ensemble' does no "
        "server-side merge, broadcasting every client's own model+weight so each client "
        "locally re-ensembles all of them via weighted majority voting at evaluate() time",
    )


def add_xgb_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--booster", type=str, default="gbtree", help="Booster to use: gbtree, gblinear or dart")
    parser.add_argument("--tree_method", type=str, default="hist", help="Tree method: exact, approx hist")
    parser.add_argument("--train_method", type=str, default="bagging", help="Train method: bagging, cyclic")
    parser.add_argument("--eta", type=float, default=0.1, help="ETA value")


def add_cox_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--l1_penalty", type=float, default=0.0, help="L1 Penalty")


def add_linear_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--solver", type=str, default="saga", help="Numerical solver of optimization method")
    parser.add_argument("--l1_ratio", type=str, default=0.5, help="L1-L2 Ratio, necessary for ElasticNet, 0 -> L1 ; 1 -> L2")
    parser.add_argument("--max_iter", type=int, default=100000, help="Max iterations of optimizer")
    parser.add_argument("--tol", type=float, default=0.001, help="Gamma for SVR")
    parser.add_argument("--kernel", type=str, default="linear", help="Kernel of SVR")
    parser.add_argument("--degree", type=int, default=3, help="Degree of polinonial")
    parser.add_argument("--gamma", type=str, default="scale", help="Gamma for SVR")


def add_nn_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dropout_p", type=float, default=0.0, help="Montecarlo dropout rate")
    parser.add_argument("--T", type=int, default=20, help="Samples of MC dropout")


def add_survival_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--time_col", type=str, default=None, help="Column containing survival times")
    parser.add_argument("--event_col", type=str, default=None, help="Column indicating event occurrence")
    parser.add_argument("--accumulative_pattern_col", type=str, default=None, help="Prefix of accumulative horizon columns")
    parser.add_argument("--negative_duration_strategy", type=str, default="clip", help="clip/shift/remove")


# Model-specific groups keyed by name, used to look up which CLI flags a group
# defines (via _flags_of) and to build the per-model applicability maps below.
MODEL_SPECIFIC_GROUPS = {
    "random_forest": add_random_forest_args,
    "xgb": add_xgb_args,
    "cox": add_cox_args,
    "linear_models": add_linear_model_args,
    "nn": add_nn_args,
    "survival": add_survival_args,
}

LINEAR_MODEL_KEYS = [
    "logistic_regression",
    "linear_regression",
    "lsvc",
    "svr",
    "svm",
    "lasso_regression",
    "ridge_regression",
    "logistic_regression_elasticnet",
]
SURVIVAL_MODEL_KEYS = ["cox", "rsf", "gbs"]
RANDOM_FOREST_MODEL_KEYS = ["random_forest", "weighted_random_forest"]

# Which model-specific groups apply to a given --model, per entry point (the two
# differ: e.g. Cox's --l1_penalty is server-only, broadcast to clients via
# on_fit_config_fn rather than a client-side flag).
SERVER_MODEL_GROUPS = {
    **{k: ["random_forest"] for k in RANDOM_FOREST_MODEL_KEYS},
    "xgb": ["xgb"],
    "cox": ["cox"],
}
CLIENT_MODEL_GROUPS = {
    **{k: ["random_forest"] for k in RANDOM_FOREST_MODEL_KEYS},
    "xgb": ["xgb"],
    "nn": ["nn"],
    **{k: ["linear_models"] for k in LINEAR_MODEL_KEYS},
    **{k: ["survival"] for k in SURVIVAL_MODEL_KEYS},
}


def _flags_of(add_fn) -> set:
    """The CLI flag (dest) names a group function registers, found by calling it
    against a throwaway parser -- avoids hand-maintaining a second flag list that
    could drift from the actual add_argument() calls above."""
    throwaway = argparse.ArgumentParser(add_help=False)
    add_fn(throwaway)
    return {action.dest for action in throwaway._actions}


def warn_unused_args(config: dict, model: str, model_groups: dict, argv=None) -> None:
    """Print a warning (not an error) listing explicitly-passed CLI flags that
    don't apply to `model`, with the values they were given, then let the caller
    continue running with `model`'s own arguments untouched.

    argparse's Namespace doesn't distinguish "explicitly passed" from "left at
    its default", so this inspects argv (defaults to sys.argv) directly for
    `--flag` / `--flag=value` tokens.
    """
    argv = sys.argv[1:] if argv is None else argv
    passed_flags = {tok[2:].split("=", 1)[0] for tok in argv if tok.startswith("--")}

    applicable_flags = set()
    for group_name in model_groups.get(model, []):
        applicable_flags |= _flags_of(MODEL_SPECIFIC_GROUPS[group_name])

    all_model_specific_flags = set()
    for add_fn in MODEL_SPECIFIC_GROUPS.values():
        all_model_specific_flags |= _flags_of(add_fn)

    irrelevant = sorted((all_model_specific_flags - applicable_flags) & passed_flags)
    if irrelevant:
        unused = {name: config.get(name) for name in irrelevant}
        print(
            f"WARNING: the following arguments are not used by model '{model}' "
            f"and will be ignored: {unused}"
        )
