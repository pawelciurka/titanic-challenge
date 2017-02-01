import features
import my_classifier
import submissions

test_titanic_features = features.TitanicFeatures('test')
train_titanic_features = features.TitanicFeatures('train')

descriptors = ['Age', 'Sex', 'Embarked', 'nCabins', 'Pclass', 'Sector', 'SibSp']

classifier = my_classifier.EvaluationClassifier()
classifier.fit(train_titanic_features.get_descriptors(descriptors), train_titanic_features.get_labels())

predictions = classifier.predict(test_titanic_features.get_descriptors(descriptors))

submissions.form_submission_file(test_titanic_features.data['PassengerId'], predictions, 'test_sub.csv')