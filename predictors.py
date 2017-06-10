from sklearn import cross_validation, ensemble, pipeline
from sklearn import svm
import pandas as pd
from collections import defaultdict
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

descriptors = []

NAIVE_BAYES_PREDICTORS = ['Sex', 'Embarked', 'nCabins', 'Pclass']

class Predictor(object):

    def __init__(self, id):
        self.id = id

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class SimpleNN(Predictor):
    def __init__(self):
        super(SimpleNN, self).__init__('simple_nn')
        self.model = self.prepare_graph()

    def prepare_graph(self):
        model = Sequential()
        model.add(Dense(12, input_dim=4, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # compile
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def fit(self, X, y):
        self.model.fit(np.asarray(X), y, epochs=150, batch_size=10)

    def predict(self, X):
        return np.round(self.model.predict(np.asarray(X))).reshape(-1)


class NaiveBayes(Predictor):
    def __init__(self, id='naive_bayes'):
        super(NaiveBayes, self).__init__(id)
        self.discrete_features = ['Sex', 'Embarked', 'nCabins', 'Pclass']
        self.continous_features = ['Age']

    def fit(self, X, y):
        n_examples = len(X)
        self.class_priors = {
            0: float(y.value_counts()[0])/n_examples,
            1: float(y.value_counts()[1])/n_examples
        }
        self._fit_feature_prob_models(X, y)

    def _fit_feature_prob_models(self, training_data, y):
        self.feature_prob_models = {}
        for feature_name in training_data:
            if feature_name in self.discrete_features:
                self.feature_prob_models[feature_name] = _DiscreteFeatureProbModel(training_data[feature_name], y)
            elif feature_name in self.continous_features:
                pass

    def score(self, X):
        scores = []
        for i, passenger_data in X.iterrows():
            p0 = []
            p1 = []
            ratios = []
            for feature_name, fpm in self.feature_prob_models.items():
                p0.append(self.class_priors[0]*fpm.p[passenger_data[feature_name]][0])
                p1.append(self.class_priors[1]*fpm.p[passenger_data[feature_name]][1])
                ratios.append(p1[-1]-p0[-1])
            survive_score = sum(ratios*np.array([1, 0.5, 0.1, 0.1]))
            scores.append(survive_score)
        return scores

    def predict(self, test_data):
        scores = self.score(test_data)
        predictions = np.asarray(scores) > 0
        return predictions.astype(np.int32)


class _DiscreteFeatureProbModel():
    def __init__(self, data_series, y):
        unique_classes = y.unique()
        unique_feature_values = data_series.unique()
        self.p = defaultdict(dict)
        for unique_class in unique_classes:
            class_series = data_series[y == unique_class]
            class_count = len(class_series)
            for feature_value in unique_feature_values:
                self.p[feature_value][unique_class] = float(sum(class_series==feature_value))/class_count


