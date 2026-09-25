"""Tests for flcore/cli_args.py: the shared argument-group definitions
server_cmd.py/client_cmd.py both build their parsers from, and the
warn_unused_args() irrelevant-flag warning.
"""
import argparse
import sys
from pathlib import Path

import pytest

from flcore.cli_args import (
    CLIENT_MODEL_GROUPS,
    MODEL_SPECIFIC_GROUPS,
    SERVER_MODEL_GROUPS,
    add_client_only_args,
    add_common_args,
    add_cox_args,
    add_linear_model_args,
    add_nn_args,
    add_random_forest_args,
    add_server_only_args,
    add_survival_args,
    add_xgb_args,
    warn_unused_args,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from run_local import _config_to_argv  # noqa: E402


def _build_server_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_server_only_args(parser)
    add_random_forest_args(parser)
    add_xgb_args(parser)
    add_cox_args(parser)
    return parser


def _build_client_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_client_only_args(parser)
    add_random_forest_args(parser)
    add_xgb_args(parser)
    add_linear_model_args(parser)
    add_nn_args(parser)
    add_survival_args(parser)
    return parser


@pytest.mark.parametrize("model", ["random_forest", "weighted_random_forest", "xgb", "cox"])
def test_server_parser_accepts_model(model):
    parser = _build_server_parser()
    args = parser.parse_args(["--model", model, "--task", "classification"])
    assert args.model == model


@pytest.mark.parametrize(
    "model",
    ["random_forest", "weighted_random_forest", "xgb", "nn", "cox", "rsf", "gbs", "logistic_regression"],
)
def test_client_parser_accepts_model(model):
    parser = _build_client_parser()
    args = parser.parse_args(["--model", model, "--task", "classification"])
    assert args.model == model


def test_server_and_client_agree_on_shared_default_values():
    """Guards the bug class where task/model/balanced/dropout_method used to
    silently disagree between the two entry points -- now
    structurally impossible for any flag defined via add_common_args, since
    both parsers call the same function."""
    server_defaults = vars(_build_server_parser().parse_args([]))
    client_defaults = vars(_build_client_parser().parse_args([]))
    shared_flags = set(server_defaults) & set(client_defaults)
    assert shared_flags, "expected server/client parsers to share at least one flag"
    for flag in shared_flags:
        assert server_defaults[flag] == client_defaults[flag], (
            f"--{flag} defaults disagree: server={server_defaults[flag]!r} "
            f"client={client_defaults[flag]!r}"
        )


def test_warn_unused_args_flags_irrelevant_model_specific_flags(capsys):
    config = {"model": "random_forest", "eta": 0.42, "booster": "gblinear", "n_estimators": 100}
    argv = ["--model", "random_forest", "--eta", "0.42", "--booster", "gblinear", "--n_estimators", "100"]
    warn_unused_args(config, "random_forest", SERVER_MODEL_GROUPS, argv=argv)
    out = capsys.readouterr().out
    assert "eta" in out
    assert "booster" in out
    assert "n_estimators" not in out  # applicable to random_forest -- must not be flagged


def test_warn_unused_args_silent_when_nothing_irrelevant_passed(capsys):
    config = {"model": "random_forest", "n_estimators": 100}
    argv = ["--model", "random_forest", "--n_estimators", "100"]
    warn_unused_args(config, "random_forest", SERVER_MODEL_GROUPS, argv=argv)
    assert capsys.readouterr().out == ""


def test_warn_unused_args_silent_for_flags_never_passed_on_argv(capsys):
    """argparse can't tell "explicitly passed" from "left at its default", so
    warn_unused_args deliberately looks only at argv: a flag that was never
    typed on the command line is never flagged, even though its default value
    is present in `config` and technically belongs to another model's group."""
    config = {"model": "random_forest", "eta": 0.1}  # --eta's default, never passed
    warn_unused_args(config, "random_forest", SERVER_MODEL_GROUPS, argv=["--model", "random_forest"])
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("groups", SERVER_MODEL_GROUPS.values())
def test_every_server_model_group_name_is_registered(groups):
    for group_name in groups:
        assert group_name in MODEL_SPECIFIC_GROUPS


@pytest.mark.parametrize("groups", CLIENT_MODEL_GROUPS.values())
def test_every_client_model_group_name_is_registered(groups):
    for group_name in groups:
        assert group_name in MODEL_SPECIFIC_GROUPS


@pytest.mark.parametrize("build_parser", [_build_server_parser, _build_client_parser], ids=["server", "client"])
def test_defaults_survive_argv_round_trip(build_parser):
    """Every flag's default must parse back to the same value (and type) when
    passed explicitly on the command line -- which is what scripts/run_local.py
    always does via _config_to_argv. A flag declared type=str with a numeric
    default (as --l1_ratio once was) hands the model '0.5' instead of 0.5
    whenever it's passed explicitly, while in-process tests that never pass it
    keep seeing the float and stay green."""
    parser = build_parser()
    defaults = vars(parser.parse_args([]))
    reparsed = vars(parser.parse_args(_config_to_argv(defaults, set(defaults))))
    mismatches = {
        name: (defaults[name], reparsed[name])
        for name in defaults
        if defaults[name] != reparsed[name] or type(defaults[name]) is not type(reparsed[name])
    }
    assert not mismatches, f"default -> argv -> parsed changed these flags (default, reparsed): {mismatches}"
