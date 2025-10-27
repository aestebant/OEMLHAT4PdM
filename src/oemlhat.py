import collections
from math import factorial
from itertools import combinations
from river import base
from river.ensemble import BaggingClassifier, SRPClassifier
from river.ensemble.streaming_random_patches import BaseSRPClassifier
from river.metrics import F1
from river.metrics.multioutput.base import MultiOutputMetric
from river.neighbors import LazySearch
from river.optim.sgd import SGD
from river.utils.random import poisson

from metrics.multioutput import *
from multioutput import *
from neighbors.knn_classifier import KNNClassifier
from linear_model.log_reg import LogisticRegression


class OEMLHAT(SRPClassifier, base.MultiLabelClassifier):
    def __init__(
        self,
        n_models = 10,
        subspace_size = 0.6,
        lam = 5,
        disable_weighted_vote = False,
        grace_period = 205,
        delta = 2.5093683326920607e-07,
        tau = 0.05,
        cardinality_th = 135,
        entropy_th = 0.75,
        drift_window_threshold = 208,
        perf_metric = MicroAverage(F1()),
        switch_significance = 0.036,
        lc_n_neighbors = 13,
        lc_window_size = 140,
        hc_lr = 0.95,
        seed = 1,
        grouping_features = None,
    ):
        self.grace_period = grace_period
        self.delta = delta
        self.tau = tau
        self.cardinality_th = cardinality_th
        self.entropy_th = entropy_th
        self.drift_window_threshold = drift_window_threshold
        self.perf_metric = perf_metric
        self.switch_significance = switch_significance
        self.lc_n_neighbors = lc_n_neighbors
        self.lc_window_size = lc_window_size
        self.hc_lr = hc_lr
        self.model = MLHAT(
            grace_period=self.grace_period,
            delta=self.delta,
            tau=self.tau,
            cardinality_th=self.cardinality_th,
            entropy_th=self.entropy_th,
            bootstrap_sampling=False,
            drift_window_threshold=self.drift_window_threshold,
            perf_metric=self.perf_metric,
            switch_significance=self.switch_significance,
            low_card_clf=LabelCombination(KNNClassifier(n_neighbors=self.lc_n_neighbors, engine=LazySearch(window_size=self.lc_window_size))),
            high_card_clf=BinaryRelevance(BaggingClassifier(LogisticRegression(SGD(self.hc_lr)))),
            seed=seed,
        )
        self.grouping_features = grouping_features
        super().__init__(
            model=self.model,
            n_models=n_models,
            subspace_size=subspace_size,
            training_method="patches",
            lam=lam,
            drift_detector=None,
            warning_detector=None,
            disable_detector="drift",
            disable_weighted_vote=disable_weighted_vote,
            seed=seed,
            metric=perf_metric
        )
        self._base_learner_class = BaseOEMLHAT

    def learn_one(self, x, y, **kwargs):
        self._n_samples_seen += 1
        if not self:
            self._init_ensemble(list(x.keys()))
        else:
            if x.keys() - set(self._known_features):
                self._update_ensemble(list(x.keys() - set(self._known_features)))
        for model in self:
            if self.training_method == self._TRAIN_RANDOM_SUBSPACES:
                k = 1
            else:
                k = poisson(rate=self.lam, rng=self._rng)
                if k == 0:
                    continue
            model.learn_one(x=x, y=y, sample_weight=k, n_samples_seen=self._n_samples_seen)

    def _init_ensemble(self, features):
        self._generate_subspaces(features=features)
        subspace_indexes = list(range(len(self._subspaces)))  # For matching subspaces with ensemble members
        if (
            self.training_method == self._TRAIN_RANDOM_PATCHES
            or self.training_method == self._TRAIN_RANDOM_SUBSPACES
        ):
            # Shuffle indexes
            self._rng.shuffle(subspace_indexes)

        # Initialize the ensemble
        for i in range(len(self._subspaces)):
            # If self.training_method == self._TRAIN_RESAMPLING then subspace is None
            subspace = self._subspaces[subspace_indexes[i]]
            self.append(
                self._base_learner_class(  # type: ignore
                    idx_original=i,
                    model=self.model,
                    metric=self.metric,
                    created_on=self._n_samples_seen,
                    drift_detector=self.drift_detector,
                    warning_detector=self.warning_detector,
                    is_background_learner=False,
                    rng=self._rng,
                    features=subspace,
                )
            )

    def _generate_subspaces(self, features):
        if self.grouping_features:
            features_aux = list(set(features) - set(self.grouping_features))
            features_aux.append(self.grouping_features[0])
            self._known_features = features_aux + self.grouping_features[1:]
        else:
            features_aux = features
            self._known_features = features_aux
        n_features = len(features_aux)


        # 1. Calculate number of features per subspace k
        if isinstance(self.subspace_size, float) and 0.0 < self.subspace_size <= 1:
            k = round(n_features * self.subspace_size)
            if k < 2:
                k = round(n_features * self.subspace_size) + 1
        else:
            raise ValueError(f"Invalid subspace_size: {self.subspace_size}")

        # 2. Generate subspaces: a 2D array (n_models, k) where each row contains the k-feature indices used per model
        if factorial(n_features) / (factorial(n_features - k) * factorial(k)) <= self.n_models:
            self._subspaces = [list(comb) for comb in combinations(features_aux, k)]
        else:
            self._subspaces = list()
            while len(self._subspaces) < self.n_models:
                subspace = self._rng.sample(features_aux, k)
                if subspace not in self._subspaces:
                    self._subspaces.append(subspace)
        if n_features < 10:
            self._subspaces.append(self._known_features) # Let's see how good is to add another model with all the features for lower feature spaces
        if self.grouping_features:
            for i, subspace in enumerate(self._subspaces[:-1]):
                if self.grouping_features[0] in subspace:
                    self._subspaces[i] += self.grouping_features[1:]

    def _update_ensemble(self, new_features):
        raise NotImplementedError

    def predict_proba_one(self, x, **kwargs):
        if not self.models:
            self._init_ensemble(features=list(x.keys()))

        y_pred_dist = collections.defaultdict(lambda: collections.defaultdict(lambda: 0.0))

        for model in self.models:
            y_pred_proba = model.predict_proba_one(x)
            metric_value = model.metric.get()
            if (model.metric.bigger_is_better and metric_value > 0.0) or (not model.metric.bigger_is_better and metric_value < 1.0) and not self.disable_weighted_vote: # TODO: assume metric is defined in [0,1]
                for lab, probs in y_pred_proba.items():
                    for val, prob in probs.items():
                        if model.metric.bigger_is_better:
                            y_pred_dist[lab][val] += prob * metric_value
                        else:
                            y_pred_dist[lab][val] += prob * (1-metric_value)
        y_pred = dict()
        for label, vals in y_pred_dist.items():
            total = sum(vals.values())
            if total > 0:
                y_pred[label] = {val: prob / total for val, prob in vals.items()}
            else:
                y_pred[label] = {val: 0.0 for val in vals}
        return y_pred

    def predict_one(self, x):
        y_pred = self.predict_proba_one(x)
        if y_pred:
            return {label: max(y_pred[label], key=y_pred[label].get) for label in y_pred}
        return None


class BaseOEMLHAT(BaseSRPClassifier):
    def __init__(
        self, idx_original, model: MLHAT, metric: MultiOutputMetric, created_on, drift_detector, warning_detector, is_background_learner, rng, features=None):
        super().__init__(idx_original, model, metric, created_on, drift_detector, warning_detector, is_background_learner, rng, features)
        self._sample_weight = 0

    def learn_one(self, x, y, *, sample_weight, n_samples_seen, **kwargs):
        self._sample_weight += sample_weight
        if self.features is not None:
            x_subset = {k: x[k] for k in self.features if k in x}
        else:
            x_subset = x

        y_pred = self.model.predict_one(x)
        if y_pred:
            self.metric.update(y_true=y, y_pred=y_pred)

        self.model.learn_one(x=x_subset, y=y, sample_weight=sample_weight)

    def predict_one(self, x, **kwargs):
        x_subset = {k: x[k] for k in self.features if k in x} if self.features is not None else x
        return self.model.predict_one(x_subset, **kwargs)
