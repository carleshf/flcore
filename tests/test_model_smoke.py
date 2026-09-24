"""Phase-0 smoke tests: for each supported model key, build a real client/server
config against a small synthetic dataset, construct the real GetModelClient /
GetModelServerStrategy objects, and drive `num_rounds` of federated training +
evaluation through the real Strategy/Client code (see fed_driver.py) -- no
subprocesses, no gRPC, no certs.

This is the safety net later phases (argument standardization, cert/testing-mode
sanitation, aggregation-strategy homogenization) get regression-tested against.
Run with `--save-golden` to (re)write the tests/golden/ baseline snapshots.
"""
import json
from pathlib import Path

import pytest

from flcore.datasets import load_dataset
from flcore.seeding import seed_everything
from flcore.utils import GetModelClient, GetModelServerStrategy

from config_builder import build_validated_config
from fed_driver import run_federated_rounds

GOLDEN_DIR = Path(__file__).parent / "golden"

# Known-broken model configs go here, xfail'd (strict=True) rather than left
# red so a genuine regression in a *passing* model still fails the suite while
# these stay visible and self-documenting. strict=True means the day someone
# fixes the underlying bug, this test flips to XPASS and fails until the xfail
# is removed -- that's the intended "please update this" signal, not a bug in
# the test. xgb was fixed this session -- see CLAUDE.md Sec 5.10.
# weighted_random_forest was fixed by the Phase 3 migration (see CLAUDE.md
# Sec 5.11) -- both its aggregation modes are exercised below, not xfail'd.
_XFAIL_REASONS = {}


def _case(model, task, data_fixture, golden_name=None, **overrides):
    golden_name = golden_name or model
    if model in _XFAIL_REASONS:
        return pytest.param(
            model,
            task,
            data_fixture,
            golden_name,
            overrides,
            marks=pytest.mark.xfail(reason=_XFAIL_REASONS[model], strict=True),
            id=golden_name,
        )
    return pytest.param(model, task, data_fixture, golden_name, overrides, id=golden_name)


# (model key, task, dataset fixture name)
MODEL_CASES = [
    _case("logistic_regression", "classification", "classification_data"),
    _case("linear_regression", "regression", "regression_data"),
    _case("random_forest", "classification", "classification_data"),
    _case("weighted_random_forest", "classification", "classification_data"),
    _case(
        "weighted_random_forest",
        "classification",
        "classification_data",
        golden_name="weighted_random_forest_client_ensemble",
        wrf_aggregation_mode="client_ensemble",
    ),
    _case("xgb", "classification", "classification_data"),
    _case("nn", "classification", "classification_data"),
    _case("cox", "survival", "survival_data"),
    _case("rsf", "survival", "survival_data"),
    _case("gbs", "survival", "survival_data"),
]


def _to_jsonable(value):
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return repr(value)


@pytest.mark.parametrize("model,task,data_fixture,golden_name,overrides", MODEL_CASES)
def test_model_round_trip(model, task, data_fixture, golden_name, overrides, request, sandbox_path):
    data_info = request.getfixturevalue(data_fixture)

    config = build_validated_config(
        model=model,
        task=task,
        data_dir=data_info["data_dir"],
        train_labels=data_info["train_labels"],
        target_labels=data_info["target_labels"],
        sandbox_path=sandbox_path,
        time_col=data_info.get("time_col"),
        event_col=data_info.get("event_col"),
        node_name="server",
        **overrides,
    )

    # nn's torch model init + DataLoader shuffling are otherwise unseeded --
    # without this, two runs of the same code produce different losses, which
    # makes golden-diffing meaningless (see CLAUDE.md Sec 5.11).
    seed_everything(config["seed"])

    _, strategy = GetModelServerStrategy(config)
    assert strategy is not None

    clients = []
    for i in range(config["num_clients"]):
        client_config = dict(config)
        client_config["node_name"] = f"client_{i}"
        data = load_dataset(client_config, i)
        clients.append(GetModelClient(client_config, data))

    result = run_federated_rounds(strategy, clients, num_rounds=config["num_rounds"])

    assert result["parameters"] is not None
    assert len(result["rounds"]) == config["num_rounds"]
    for round_info in result["rounds"]:
        assert not round_info["fit_failures"], round_info["fit_failures"]
        assert not round_info["eval_failures"], round_info["eval_failures"]

    if request.config.getoption("--save-golden"):
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "model": model,
            "task": task,
            "rounds": [
                {
                    "round": r["round"],
                    "fit_metrics": _to_jsonable(r["fit_metrics"]),
                    "loss": _to_jsonable(r["loss"]),
                    "eval_metrics": _to_jsonable(r["eval_metrics"]),
                }
                for r in result["rounds"]
            ],
        }
        (GOLDEN_DIR / f"{golden_name}.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True))
