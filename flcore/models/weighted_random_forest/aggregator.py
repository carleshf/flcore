"""Two aggregation modes for weighted_random_forest, selected by
config["wrf_aggregation_mode"] (default "server_merge", see flcore/cli_args.py):

- server_merge (default): reuses random_forest's proven RandomForestAggregator
  (tree pooling + probability-weighted downsampling + round-to-round growth)
  as-is, parameterized with this model's own narrower model factory
  (get_model(bal_RF, random_state) -- classification-only, no n_estimators/
  max_depth/class_weight knobs, unlike random_forest's own get_model(config))
  instead of duplicating that algorithm.

- client_ensemble: this model's original design (previously broken -- see
  CLAUDE.md Sec 5.10). The server does no merge at all: it forwards every
  client's own (model, num_examples, weight) tuple to every client, and each
  client locally re-ensembles all of them via weighted majority voting at
  evaluate() time (flcore/models/weighted_random_forest/client.py::
  ensambleRFTrees / ensambleDecisionTrees + mlxtend.EnsembleVoteClassifier).
  WeightedRandomForestBroadcastAggregator.aggregate() returns that untouched
  tuple list instead of a merged model -- the "broadcast, don't merge" case
  the Phase 3 plan anticipated needing new BaseFLStrategy machinery for, which
  turned out not to be needed: it fits the existing aggregator_cls/serialize_fn
  contract as-is, just with a different Aggregator and a different codec (see
  broadcast_serialize/broadcast_deserialize below).
"""
import pickle

import numpy as np
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays

from flcore.base_aggregator import BaseAggregator


class WeightedRandomForestBroadcastAggregator(BaseAggregator):
    """self.weights already reflects BaseFLStrategy._compute_weights'
    homogeneous-when-disabled default (see random_forest's aggregator.py
    docstring for why that's equivalent to the pre-migration "no smoothing"
    fallback), so every tuple here always carries a real weight -- removing
    the 2-tuple-vs-3-tuple ambiguity ensambleRFTrees/ensambleDecisionTrees
    used to branch on (`if len(parameters[i]) == 3`), though they still accept
    either shape unchanged."""

    def __init__(self, models, weights, num_examples):
        super().__init__(models=models, weights=weights)
        self.num_examples = num_examples

    def aggregate(self):
        # Each self.models[i] is deserialize_RF's result -- a 1-element list
        # holding that client's whole fitted model, i.e. exactly the shape
        # client.py's ensambleRFTrees/ensambleDecisionTrees already expect at
        # parameters[i][0][0]. Left wrapped (not unwrapped to self.models[i][0])
        # to match that shape.
        return [
            (model, n, w)
            for model, n, w in zip(self.models, self.num_examples, self.weights)
        ]


def broadcast_serialize(payload) -> Parameters:
    """The per-client tuple list is ragged (each tuple mixes a model object
    with scalars) -- can't go through serialize_RF's per-element np.save,
    which is exactly the pre-existing bug this migration fixes (np.save tries
    to build one rectangular array per top-level list element; a tuple of
    mismatched-shape things isn't one). Pickling the whole payload into a
    single opaque bytes blob sidesteps that: same "arbitrary object wrapped as
    one ndarray" approach xgb's migration already established for a similarly
    non-tensor payload (flcore/models/xgb/server.py::_xgb_serialize)."""
    return ndarrays_to_parameters([np.frombuffer(pickle.dumps(payload), dtype=np.uint8)])


def broadcast_deserialize(parameters) -> list:
    ndarrays = parameters_to_ndarrays(parameters)
    raw = ndarrays[0].tobytes()
    if not raw:
        return []
    return pickle.loads(raw)
