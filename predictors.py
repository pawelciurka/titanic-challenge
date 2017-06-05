from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

NAIVE_BAYES_PREDICTORS = ['Pclass', 'AgeGroup', 'Title', 'FareGroup']


class Predictor(object):

    def __init__(self, id):
        self.id = id

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class NaiveBayes(Predictor):
    def __init__(self, id='naive_bayes'):
        super(NaiveBayes, self).__init__(id)

    def fit(self, X, y):
        n_examples = len(X)
        self.class_priors = {
            0: float(y.value_counts()[0])/n_examples,
            1: float(y.value_counts()[1])/n_examples
        }
        self._fit_feature_prob_models(X, y)
        self._train_weights(X, y)
        print(self.weights)

    def _fit_feature_prob_models(self, training_data, y):
        self.feature_prob_models = dict()
        for feature_name in training_data:
            self.feature_prob_models[feature_name] = _DiscreteFeatureProbModel(training_data[feature_name], y)

    def _train_weights(self, X, y):
        self.single_scores = self.single_feature_scores(X)
        self.y = y

        feature_names = self.single_scores.columns._data

        space = [hp.uniform(feature_name, 0, 1) for feature_name in feature_names]

        trials = Trials()
        self.weights = fmin(
            self.loss_fun,
            space=space,
            algo=tpe.suggest,
            max_evals=1200,
            trials=trials
        )
        import matplotlib.pyplot as plt
        plt.plot(trials.losses())

        pass

    def loss_fun(self, a):
        scores = [sum(np.asarray(ss) * np.asarray(a)) for i, ss in self.single_scores.iterrows()]
        decisions = np.asarray(scores) > 0
        matches = self.y == decisions
        error_rate = 1 - float(sum(matches)) / len(matches)
        return error_rate

    def single_feature_scores(self, X):

        single_feature_scores = pd.DataFrame()
        for i, passenger_data in X.iterrows():
            for feature_name, fpm in self.feature_prob_models.items():
                p0 = self.class_priors[0] * fpm.p[passenger_data[feature_name]][0]
                p1 = self.class_priors[1] * fpm.p[passenger_data[feature_name]][1]
                single_feature_scores.loc[i, feature_name] = p1 - p0
        return single_feature_scores

    def predict(self, test_data):
        single_scores = self.single_feature_scores(test_data)
        keys = self.weights.keys()
        weights = np.asarray([self.weights[key] for key in keys])
        scores = [sum(np.asarray(ss[keys]) * weights) for i, ss in single_scores.iterrows()]
        predictions = np.asarray(scores) > 0
        return predictions.astype(np.int32)


class _DiscreteFeatureProbModel():
    def __init__(self, data_series, y):
        unique_classes = y.unique()
        unique_feature_values = data_series.unique()
        self.p = defaultdict(lambda: {0: 0.0, 1: 0.0})
        for unique_class in unique_classes:
            class_series = data_series[y == unique_class]
            class_count = len(class_series)
            for feature_value in unique_feature_values:
                self.p[feature_value][unique_class] = float(sum(class_series==feature_value))/class_count
