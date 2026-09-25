"""Local test-setup helper: spawns one server_cmd.py process +
--num_clients client_cmd.py processes locally, against a real dataset. Replaces
what the old YAML-config-driven run.py launcher used to do (since deleted) --
built on top of the server_cmd.py/client_cmd.py stack, and always
CLI-argument-driven, never a config file.

Always forced into --testing_mode: this is a local dev/testing launcher, not a
production one, so it connects everything over LOCALHOST rather than reading
production env vars. --enable_certs still defaults off (pass it explicitly if
you want to test against real certs locally).

All N client processes currently load the *same* dataset slice (no per-center
partitioning -- the dt4h_format loader ignores the client id), so --num_clients > 1 exercises multi-client aggregation
code paths but not realistic data heterogeneity.

Usage:
    python scripts/run_local.py --model random_forest --task classification \\
        --train_labels age systolic_bp --target_labels outcome_flag \\
        --data_id /path/to/dataset_dir --num_rounds 3 --num_clients 2 \\
        --sandbox_path ./sandbox
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from flcore.cli_args import (  # noqa: E402
    add_common_args,
    add_cox_args,
    add_client_only_args,
    add_linear_model_args,
    add_nn_args,
    add_random_forest_args,
    add_server_only_args,
    add_survival_args,
    add_xgb_args,
    _bool_optional_flags_of,
    _flags_of,
)

# Mirrors exactly what server_cmd.py / client_cmd.py each register (see
# flcore/cli_args.py) -- used to split this script's single combined config back
# into the two entry points' own argv.
_ALL_GROUPS = (
    add_common_args,
    add_server_only_args,
    add_client_only_args,
    add_random_forest_args,
    add_xgb_args,
    add_cox_args,
    add_linear_model_args,
    add_nn_args,
    add_survival_args,
)
_SERVER_GROUPS = (add_server_only_args, add_random_forest_args, add_xgb_args, add_cox_args)
_CLIENT_GROUPS = (
    add_client_only_args,
    add_random_forest_args,
    add_xgb_args,
    add_linear_model_args,
    add_nn_args,
    add_survival_args,
)

COMMON_FLAGS = _flags_of(add_common_args)
SERVER_FLAGS = set().union(*(_flags_of(fn) for fn in _SERVER_GROUPS))
CLIENT_FLAGS = set().union(*(_flags_of(fn) for fn in _CLIENT_GROUPS))
BOOLEAN_OPTIONAL_FLAGS = set().union(*(_bool_optional_flags_of(fn) for fn in _ALL_GROUPS))


def _config_to_argv(config: dict, flags: set) -> list:
    """Serialize the subset of `config` restricted to `flags` back into
    --flag/--flag value CLI tokens, the way argparse would have produced it --
    so server_cmd.py/client_cmd.py re-parse exactly what this script parsed."""
    argv = []
    for name in sorted(flags):
        if name not in config:
            continue
        value = config[name]
        flag = f"--{name}"
        if isinstance(value, bool):
            if name in BOOLEAN_OPTIONAL_FLAGS:
                # Has a --no-<flag> negative form and a default that isn't
                # necessarily False (e.g. --balanced defaults True) -- omitting
                # the flag would silently fall back to that default instead of
                # forcing False, so always state it explicitly either way.
                argv.append(flag if value else f"--no-{name}")
            elif value:
                argv.append(flag)
        elif isinstance(value, list):
            if value:
                argv.extend([flag, *[str(v) for v in value]])
        elif value is not None:
            argv.extend([flag, str(value)])
    return argv


def _wait_until_ready_or_dead(proc, timeout=20):
    """No clean readiness signal is exposed by server_cmd.py (see
    tests/test_e2e_local.py), so poll briefly and bail out early -- instead of a
    blind fixed sleep -- if the server already died."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server_cmd.py exited early with code {proc.returncode}")
        time.sleep(0.5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    add_server_only_args(parser)
    add_client_only_args(parser)
    add_random_forest_args(parser)
    add_xgb_args(parser)
    add_cox_args(parser)
    add_linear_model_args(parser)
    add_nn_args(parser)
    add_survival_args(parser)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = vars(args)

    argv = sys.argv[1:]
    if not config.get("model"):
        parser.error("--model is required")
    if "--data_id" not in argv:
        # --data_id's shared default ("data_id.parquet", from add_client_only_args)
        # is a non-empty placeholder, not None -- can't just check truthiness.
        parser.error("--data_id is required (a directory containing *.parquet + metadata.json)")

    config["testing_mode"] = True

    server_argv = _config_to_argv(config, COMMON_FLAGS | SERVER_FLAGS)
    server_cmd = [sys.executable, str(REPO_ROOT / "server_cmd.py"), *server_argv]
    print("Starting server:", " ".join(server_cmd))
    server = subprocess.Popen(server_cmd, cwd=REPO_ROOT)

    client_procs = []
    try:
        _wait_until_ready_or_dead(server)

        num_clients = config.get("num_clients") or 1
        for i in range(num_clients):
            client_config = dict(config)
            client_config["node_name"] = f"client_{i}"
            client_argv = _config_to_argv(client_config, COMMON_FLAGS | CLIENT_FLAGS)
            client_cmd = [sys.executable, str(REPO_ROOT / "client_cmd.py"), *client_argv]
            print(f"Starting client {i}:", " ".join(client_cmd))
            client_procs.append(subprocess.Popen(client_cmd, cwd=REPO_ROOT))

        exit_codes = [proc.wait() for proc in client_procs]
        for i, code in enumerate(exit_codes):
            if code != 0:
                print(f"client {i} exited with code {code}")

        if any(code != 0 for code in exit_codes):
            # A client that never connected (e.g. bad --data_id) leaves the
            # server blocked in start_server forever waiting for one -- don't
            # wait for a "completion" that can't happen.
            print("At least one client failed; stopping the server instead of waiting for it.")
            if server.poll() is None:
                server.terminate()
            return 1

        return server.wait()
    except KeyboardInterrupt:
        print("Interrupted, stopping server and clients...")
        return 130
    finally:
        for proc in client_procs:
            if proc.poll() is None:
                proc.terminate()
        if server.poll() is None:
            server.terminate()
        for proc in client_procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        if server.poll() is None:
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    sys.exit(main())
