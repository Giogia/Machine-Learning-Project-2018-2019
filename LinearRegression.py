from src.CNN_v1 import CNN
from src.Classifier import Classifier
from src.DataHandler import load_data
from src.FeaturesSelector import FeaturesSelector

'''
The default dictionary for the Linear Regression parameters in scikit-learn

linear_dict = {
    'fit_intercept': True,  # Whether to calculate the intercept for this model. 
                            # If set to False, data is expected to be already centered.
    'normalize': False,     # Ignored when fit_intercept is set to False.
                            # If True, the regressors X will be normalized before regression 
                            # by subtracting the mean and dividing by the l2-norm. 
    'copy_X': True,         # If True, X will be copied; else, it may be overwritten.
    'n_jobs': None          # The number of jobs to use for the computation. -1 means using all processors.
}
'''

linear_dict = {
    'fit_intercept': True,
    'normalize': False,
    'copy_X': True,
    'n_jobs': None
}

USE_CNN = True
OPTIMAL_FEATURE_NUMBER_LDA = 9

sets, class_names = load_data(eval_percentage=0.2)

if USE_CNN:

    # Compute high level features
    feature_extractor = CNN()
    sets.train.x, sets.eval.x, sets.test.x = feature_extractor.extract(sets.train.x, sets.eval.x, sets.test.x)

linear_classifier = Classifier(Classifier.LINEAR, **linear_dict)

feature_selector = FeaturesSelector(FeaturesSelector.LDA, OPTIMAL_FEATURE_NUMBER_LDA)
sets = feature_selector.fit(sets)

# Predict the training, evaluation and test set
train_predict, eval_predict, test_predict = linear_classifier.get_predictions(features=sets.train.x,
                                                                           labels=sets.train.y,
                                                                           eval_features=sets.eval.x,
                                                                           eval_labels=sets.eval.y,
                                                                           test_features=sets.test.x)

train_accuracy = sum([train_predict[i] == sets.train.y[i] for i in range(len(train_predict))]) / len(train_predict)
eval_accuracy = sum([eval_predict[i] == sets.eval.y[i] for i in range(len(eval_predict))]) / len(eval_predict)
test_accuracy = sum([test_predict[i] == sets.test.y[i] for i in range(len(test_predict))]) / len(test_predict)

print("\nTrain Accuracy: {}".format(train_accuracy))
print("Validation Accuracy: {}".format(eval_accuracy))
print("Test Accuracy: {}".format(test_accuracy))
