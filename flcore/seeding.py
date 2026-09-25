"""Central place to seed every source of randomness this package touches.

Before this, --seed (config["seed"], default 42) was only honored in a handful
of call sites (random_forest/linear_models' sklearn random_state=, a few
dataset loaders' train_test_split) -- everything else was silently unseeded:
Python's own `random` module (used by flcore/dropout.py's random_dropout and
random_forest/weighted_random_forest's aggregation), numpy's global RNG, and
torch entirely (model init, DataLoader shuffling). Found while migrating `nn`
onto BaseFLStrategy -- two runs of the *same* code produced different losses, which
made golden-diffing that migration impossible without a workaround.
"""
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python's random, numpy, and torch's global RNGs. Call once, early,
    before any client/server/strategy object is constructed."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
