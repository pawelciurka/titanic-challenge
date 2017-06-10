from collections import defaultdict
from copy import deepcopy

from sklearn import cross_validation

import features
import predictors
from sklearn import cross_validation
from copy import deepcopy
import pandas as pd

class CrossvalidationConfig(dict):
    def __init__(self, n_folds=5):
        super(CrossvalidationConfig, self).__init__(
            n_folds=n_folds)


class Crossvalidation(object):
    def __init__(self, predictors, titanic_features, crossvalidation_config=None):
        """
        :type predictor: list[predictors.Predictor]
        :type crossvalidation_config: CrossvalidationConfig
        """
        if crossvalidation_config is None:
            crossvalidation_config = CrossvalidationConfig()

        self.titanic_features = features.TitanicFeatures('train', descriptors)
        self.predictors = predictors
        self.config = crossvalidation_config

    def perform_test(self):
        folds = cross_validation.KFold(len(self.titanic_features.data), n_folds=self.config['n_folds'])
        for train_idx, test_idx in folds:

            # make a deepcopy of predictors for each fold
            _predictors = deepcopy(self.predictors)

            # get data and labels for training and testing data
            X_train, y_train = self.titanic_features.get_X_y(train_idx)
            X_test, y_test = self.titanic_features.get_X_y(test_idx)

            # fit all predictors using the training data
            self.fit_predictors(X_train, y_train, _predictors)

            # predict on testing data
            predictions = self.predict(X_test, _predictors)

            self.attach_predictions(predictions, test_idx)

        print(self.calculate_accuracy())

    @staticmethod
    def fit_predictors(X, y, _predictors):
        for predictor in _predictors:
            predictor.fit(X, y)

    @staticmethod
    def predict(X, _predictors):
        predictions = {}
        for predictor in _predictors:
            predictions[predictor.id] = predictor.predict(X)
        return predictions

    def attach_predictions(self, predictions, idx):
        for id, _predictions in predictions.items():
            self.titanic_features.data.ix[idx, id] = _predictions

    def calculate_accuracy(self):

        results = defaultdict(dict)
        for predictor in self.predictors:
            predictor_id = predictor.id
            matches = self.titanic_features.data['Survived'] == self.titanic_features.data[predictor_id]
            accuracy = float(sum(matches)) / len(matches)
            results[predictor_id]['accuracy'] = accuracy
            results[predictor_id]['error_ratio'] = 1 - accuracy
        return results

if __name__ == '__main__':
    crossvalidation = Crossvalidation([predictors.SimpleNN()], predictors.NAIVE_BAYES_PREDICTORS)
    crossvalidation.perform_test()