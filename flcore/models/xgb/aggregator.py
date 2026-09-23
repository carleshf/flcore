"""Aggregator for xgb: bagging (sequential JSON tree-merge across every
client's model) or cyclic (keep only the newest client's model) -- ports
flcore/models/xgb/server.py's aggregate_bagging/_get_tree_nums as-is.

Unlike every other migrated model, this aggregation doesn't average or weight
client contributions at all -- bagging appends each client's trees onto a
running combined model, cyclic just replaces it -- so self.weights (computed
by BaseFLStrategy._compute_weights, computeSmoothedWeights by default) is
accepted for interface consistency but never read, same as rsf/gbs's
aggregators (see CLAUDE.md Sec 5.11).
"""
import json

from flcore.base_aggregator import BaseAggregator


def _get_tree_nums(xgb_model_org: bytes):
    """Extract total tree numbers from XGBoost JSON model."""
    bst = json.loads(bytearray(xgb_model_org))
    model = bst["learner"]["gradient_booster"]["model"]
    tree_num = int(model["gbtree_model_param"]["num_trees"])
    paral_tree_num = int(model["gbtree_model_param"]["num_parallel_tree"])
    return tree_num, paral_tree_num


def aggregate_bagging(bst_prev_org: bytes, bst_curr_org: bytes) -> bytes:
    """Conduct bagging aggregation for given trees."""
    if bst_prev_org == b"":
        return bst_curr_org

    tree_num_prev, _ = _get_tree_nums(bst_prev_org)
    _, paral_tree_num_curr = _get_tree_nums(bst_curr_org)

    bst_prev = json.loads(bytearray(bst_prev_org))
    bst_curr = json.loads(bytearray(bst_curr_org))

    previous_model = bst_prev["learner"]["gradient_booster"]["model"]
    previous_model["gbtree_model_param"]["num_trees"] = str(
        tree_num_prev + paral_tree_num_curr
    )

    trees_curr = bst_curr["learner"]["gradient_booster"]["model"]["trees"]

    for tree_count in range(paral_tree_num_curr):
        trees_curr[tree_count]["id"] = tree_num_prev + tree_count
        previous_model["trees"].append(trees_curr[tree_count])
        previous_model["tree_info"].append(0)

    return bytes(json.dumps(bst_prev), "utf-8")


class XGBAggregator(BaseAggregator):
    def __init__(self, models, weights, train_method="bagging", current_model=b""):
        super().__init__(models=models, weights=weights)
        self.train_method = train_method
        self.current_model = current_model
        # Populated by aggregate(); BaseFLStrategy._after_aggregate reads this
        # back to carry state into next round's _aggregator_kwargs().
        self.updated_current_model = None

    def aggregate(self) -> bytes:
        # A client submitting literally empty bytes (e.g. a booster that
        # produced no trees) would crash aggregate_bagging's _get_tree_nums
        # (json.loads(b"") raises) -- filter it out rather than letting one
        # empty client abort the whole round. Pre-migration this was a
        # strategy-level "if not models: return None, {}" that dropped params
        # *and* metrics for the round; falling back to "keep current_model
        # unchanged" here is strictly more permissive (metrics still get
        # computed by the caller) but only differs in the all-clients-empty
        # edge case, which shouldn't occur in practice.
        valid_models = [m for m in self.models if m]

        if self.train_method == "bagging":
            combined = self.current_model
            for model_bytes in valid_models:
                combined = aggregate_bagging(combined, model_bytes)
        else:
            combined = valid_models[-1] if valid_models else self.current_model

        self.updated_current_model = combined
        return combined
