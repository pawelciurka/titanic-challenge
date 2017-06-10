import features
import predictors
import submissions

descriptors = predictors.NAIVE_BAYES_PREDICTORS

test_titanic_features = features.TitanicFeatures('test', descriptors)
train_titanic_features = features.TitanicFeatures('train', descriptors)

predictor = predictors.SimpleNN()

predictor.fit(train_titanic_features.get_X(), train_titanic_features.get_y())
predictions = predictor.predict(test_titanic_features.get_X())

submissions.form_submission_file(test_titanic_features.data['PassengerId'], predictions)