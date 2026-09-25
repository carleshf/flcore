"""Real-subprocess end-to-end run of every model key through scripts/run_local.py:
one server_cmd.py + 2 client_cmd.py processes talking over localhost gRPC.

test_model_smoke.py already exercises each model's Strategy/Client logic
in-process, but it builds its config dict directly and so never goes through
argparse -> argv -> argparse. Bugs that only exist on that path (e.g. a flag
parsed as the wrong type, a model alias CheckClientConfig mishandles) only
show up here.

A run counts as failed if run_local.py exits non-zero, history.yaml isn't
written, or any process printed a Traceback -- the last check matters because
the server finishes and exits 0 even when every client's fit() raised.
"""
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from fixtures.synthetic_dt4h import make_dt4h_fixture, make_survival_fixture

REPO_ROOT = Path(__file__).parent.parent

# (model key, task, extra CLI args, test id)
E2E_CASES = [
    ("logistic_regression", "classification", [], None),
    ("logistic_regression_elasticnet", "classification", [], None),
    ("lsvc", "classification", [], None),
    ("svm", "classification", [], None),
    ("linear_regression", "regression", [], None),
    ("lasso_regression", "regression", [], None),
    ("ridge_regression", "regression", [], None),
    ("svr", "regression", [], None),
    ("random_forest", "classification", ["--n_estimators", "5"], "random_forest-classification"),
    ("random_forest", "regression", ["--n_estimators", "5"], "random_forest-regression"),
    ("weighted_random_forest", "classification", [], "weighted_random_forest-server_merge"),
    (
        "weighted_random_forest",
        "classification",
        ["--wrf_aggregation_mode", "client_ensemble"],
        "weighted_random_forest-client_ensemble",
    ),
    ("xgb", "classification", [], None),
    ("nn", "classification", ["--local_epochs", "2"], "nn-classification"),
    ("nn", "regression", ["--local_epochs", "2"], "nn-regression"),
    ("nn", "classification", ["--local_epochs", "2", "--strategy", "UncertaintyWeighted"], "nn-uncertainty_weighted"),
    ("cox", "survival", [], None),
    ("rsf", "survival", [], None),
    ("gbs", "survival", [], None),
    (
        "random_forest",
        "classification",
        ["--n_estimators", "5", "--num_clients", "3", "--num_rounds", "3",
         "--dropout_method", "random_dropout", "--dropout_percentage", "50"],
        "random_forest-dropout",
    ),
]


def _free_port():
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e_data")
    return {
        "classification": make_dt4h_fixture(root / "cls", task="classification", n_rows=160),
        "regression": make_dt4h_fixture(root / "reg", task="regression", n_rows=160),
        "survival": make_survival_fixture(root / "surv", n_rows=160),
    }


@pytest.mark.parametrize(
    "model,task,extra",
    [pytest.param(m, t, e, id=i or m) for m, t, e, i in E2E_CASES],
)
def test_model_runs_end_to_end(model, task, extra, datasets, tmp_path):
    data = datasets[task]
    sandbox = tmp_path / "sandbox"
    argv = [
        sys.executable, str(REPO_ROOT / "scripts" / "run_local.py"),
        "--model", model,
        "--task", task,
        "--train_labels", *data["train_labels"],
        "--data_id", data["data_dir"],
        "--num_rounds", "2",
        "--num_clients", "2",
        "--sandbox_path", str(sandbox),
        "--local_port", str(_free_port()),
    ]
    if data["target_labels"]:
        argv += ["--target_labels", *data["target_labels"]]
    if task == "survival":
        argv += ["--dataset", "survival", "--time_col", data["time_col"], "--event_col", data["event_col"]]
    argv += extra  # later flags win, so extra can override num_rounds/num_clients

    result = subprocess.run(argv, cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)

    output = result.stdout + result.stderr
    output += "".join(p.read_text(errors="replace") for p in sandbox.rglob("log_*.txt"))
    details = f"exit code {result.returncode}\n--- output (last 4000 chars) ---\n{output[-4000:]}"
    assert result.returncode == 0, details
    assert (sandbox / "experiment_1" / "history.yaml").exists(), details
    assert "Traceback" not in output, details
