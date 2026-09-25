"""Regression test for scripts/run_local.py: spawns it as a real
subprocess (which itself spawns server_cmd.py + client_cmd.py), against the
synthetic fixture -- catches issues like the sandbox_path mkdir-ordering bug
and the --data_id required-check found while building the script.
"""
import subprocess
import sys
from pathlib import Path

from fixtures.synthetic_dt4h import make_dt4h_fixture

REPO_ROOT = Path(__file__).parent.parent


def test_run_local_succeeds_with_two_clients(tmp_path):
    data_info = make_dt4h_fixture(tmp_path / "data", task="classification", n_rows=120)
    sandbox = tmp_path / "sandbox"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_local.py"),
            "--model", "random_forest",
            "--task", "classification",
            "--train_labels", *data_info["train_labels"],
            "--target_labels", *data_info["target_labels"],
            "--data_id", data_info["data_dir"],
            "--num_rounds", "1",
            "--num_clients", "2",
            "--n_estimators", "5",
            "--max_depth", "2",
            "--sandbox_path", str(sandbox),
            "--dropout_method", "None",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert (sandbox / "experiment_1" / "history.yaml").exists()


def test_run_local_fails_fast_without_data_id():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_local.py"), "--model", "random_forest"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "--data_id is required" in result.stderr


def test_run_local_fails_fast_without_model():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_local.py"), "--data_id", "/tmp/whatever"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "--model is required" in result.stderr


def test_run_local_tears_down_server_when_client_fails(tmp_path):
    """A client that can never connect (bad --data_id) must not leave the
    server hanging in start_server forever -- see scripts/run_local.py's
    docstring/comments for why this needed its own handling."""
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_local.py"),
            "--model", "random_forest",
            "--task", "classification",
            "--data_id", str(tmp_path / "does_not_exist"),
            "--dropout_method", "None",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,  # would hang until this timeout if the teardown regressed
    )
    assert result.returncode == 1, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
