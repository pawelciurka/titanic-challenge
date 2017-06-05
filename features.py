import numpy as np
import pandas as pd

_DATA = dict()
_DATA['train'] = pd.read_csv(r'./data/train.csv')
_DATA['test'] = pd.read_csv(r'./data/test.csv')


class TitanicFeatures(object):
    def __init__(self, train_or_test, descriptors):
        self.train_or_test = train_or_test
        self.data = _DATA[self.train_or_test]
        self._clean()
        self._form_new_features()
        self._attach_to_bins()
        self._numberize()
        self.descriptors = descriptors

    def _clean(self):
        self.data['Age'] = self.data['Age'].fillna(_DATA['train']['Age'].median())
        self.data['Embarked'] = self.data['Embarked'].fillna('S')
        self.data['Cabin'] = self.data['Cabin'].fillna('X')

    def _numberize(self):
        """Non-number values are translated into numbers"""

        self.data.loc[self.data['Sex'] == 'female', 'Sex'] = 0
        self.data.loc[self.data['Sex'] == 'male', 'Sex'] = 1

        self.data.loc[self.data['Embarked'] == 'C', 'Embarked'] = 0
        self.data.loc[self.data['Embarked'] == 'Q', 'Embarked'] = 1
        self.data.loc[self.data['Embarked'] == 'S', 'Embarked'] = 2

        self.data.loc[self.data['Sector'] == 'A', 'Sector'] = 1
        self.data.loc[self.data['Sector'] == 'B', 'Sector'] = 2
        self.data.loc[self.data['Sector'] == 'C', 'Sector'] = 3
        self.data.loc[self.data['Sector'] == 'D', 'Sector'] = 4
        self.data.loc[self.data['Sector'] == 'E', 'Sector'] = 5
        self.data.loc[self.data['Sector'] == 'F', 'Sector'] = 6
        self.data.loc[self.data['Sector'] == 'G', 'Sector'] = 7
        self.data.loc[self.data['Sector'] == 'X', 'Sector'] = 0
        self.data.loc[self.data['Sector'] == 'T', 'Sector'] = 8

        for i, title in enumerate(self.data['Title'].unique()):
            self.data.loc[self.data['Title'] == title, 'Title'] = i

    def _form_new_features(self):

        # number of cabins reserved
        self.data['nCabins'] = self.data['Cabin'].map(lambda x: len(x.split(' ')))
        self.data.loc[self.data['Cabin'] == 'X', 'nCabins'] = 0

        # first letter of 'Cabin' feature
        self.data['Sector'] = self.data['Cabin'].map(lambda x: x[0])

        # title
        self.data['Title'] = self.data['Name'].map(lambda x: x.split(', ')[1].split('.')[0])

        # surname
        self.data['Surname'] = self.data['Name'].map(lambda x: x.split(',')[0])

    def _attach_to_bins(self):
        """
        Split Age and Fare into groups, attach group id.
        """
        age_bin_ranges = np.linspace(self.data['Age'].min(), self.data['Age'].max(), num=20)
        age_bin_ranges = [self.data['Age'].min(), 5, 15, 60, self.data['Age'].max() + 1]
        age_bin_ranges[-1] += 1
        for i in range(len(age_bin_ranges) - 1):
            self.data.loc[
                (self.data['Age'] >= age_bin_ranges[i]) & (self.data['Age'] < age_bin_ranges[i + 1]), 'AgeGroup'] = i

        fare_ranges = [self.data['Fare'].min(), 200, 500, self.data['Fare'].max() + 1]
        for i in range(len(fare_ranges) - 1):
            self.data.loc[
                (self.data['Fare'] >= fare_ranges[i]) & (self.data['Fare'] < fare_ranges[i + 1]), 'FareGroup'] = i

    def get_X(self, idxs=None):
        if idxs is None:
            return self.data[self.descriptors]
        else:
            return self.data.loc[idxs, self.descriptors]

    def get_y(self, idxs=None):
        if idxs is None:
            return self.data['Survived']
        else:
            return self.data.loc[idxs, 'Survived']

    def get_X_y(self, idxs=None):
        return self.get_X(idxs), self.get_y(idxs)
