"""Aggregator for random_forest: pools every client's trees into one probability-
weighted sample, downsampled back to one client's worth of trees per round, then
grown across rounds by concatenating each round's sample onto the previous one.

Ports flcore/models/random_forest/aggregatorRF.py::aggregateRFwithSizeCenterProbs /
..._withprevious as-is, restructured around BaseAggregator's models/weights
contract instead of the old (deserialized, num_examples) tuple list, and around
BaseFLStrategy's _aggregator_kwargs/_after_aggregate hooks instead of the Strategy
calling two different module-level functions depending on server_round.

Per-client weighting is unchanged: BaseFLStrategy._compute_weights (the
unmodified default, computeSmoothedWeights) already replaces this model's old
inline fallback ("if smooth_method == 'None': weights = [1]*num_clients, else
computeSmoothedWeights(...)") -- both produce the same uniform distribution once
normalized below, so this is exactly behavior-preserving, unlike linear_models'
migration (see CLAUDE.md Sec 5.11) where the pre-migration 'None' path used a
genuinely different (proportional, not uniform) weighting scheme.

model_factory (a zero-arg callable returning a fresh, unfitted model instance)
is injected rather than importing this package's own get_model directly, so
weighted_random_forest's server_merge mode (Sec 5.11) can reuse this exact
pooling/sampling/growth algorithm with its own narrower model factory
(get_model(bal_RF, random_state), classification-only, no n_estimators/
max_depth/class_weight knobs) instead of duplicating it.
"""
import numpy as np

from flcore.base_aggregator import BaseAggregator


class RandomForestAggregator(BaseAggregator):
    def __init__(self, models, weights, model_factory, previous_estimators=None, previous_estimator_weights=None):
        super().__init__(models=models, weights=weights)
        self.model_factory = model_factory
        self.previous_estimators = previous_estimators
        self.previous_estimator_weights = previous_estimator_weights
        # Populated by aggregate(); BaseFLStrategy._after_aggregate reads these
        # back to carry state into next round's _aggregator_kwargs().
        self.updated_estimators = None
        self.updated_estimator_weights = None

    def aggregate(self):
        rfa = self.model_factory()
        number_clients = len(self.models)
        # Each self.models[i] is deserialize_RF's result: a 1-element list
        # holding that client's whole fitted RF model object -- sklearn's
        # BaseEnsemble supports len()/iteration/indexing by delegating to its
        # own .estimators_, so treating it as a sequence-of-trees (as the old
        # np.concatenate((..., rfs[i][0][0])) calls implicitly relied on) works
        # the same way here.
        trees_per_client = len(self.models[0][0])

        list_classifiers = []
        weights_classifiers = []
        for i in range(number_clients):
            list_classifiers = np.concatenate((list_classifiers, self.models[i][0]))
            weights_classifiers = np.concatenate(
                (weights_classifiers, [self.weights[i]] * len(self.models[i][0]))
            )

        weights_classifiers = weights_classifiers / sum(weights_classifiers)
        selected_indices = np.random.choice(
            len(list_classifiers), trees_per_client, p=weights_classifiers
        )
        selected_trees = list_classifiers[selected_indices]
        selected_weights = weights_classifiers[selected_indices]

        if self.previous_estimators is not None:
            selected_trees = np.concatenate((self.previous_estimators, selected_trees))
            selected_weights = np.concatenate((self.previous_estimator_weights, selected_weights))

        rfa.estimators_ = np.array(selected_trees)
        rfa.n_estimators = len(rfa.estimators_)

        self.updated_estimators = rfa.estimators_
        self.updated_estimator_weights = selected_weights

        return [rfa]
