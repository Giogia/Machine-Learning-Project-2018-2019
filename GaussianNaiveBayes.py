from DataHandler import load_data
from CNN_v1 import CNN
from Classifier import Classifier
from FeaturesSelector import FeaturesSelector

'''
The default dictionary for the Gaussian Naive Bayes parameters in scikit-learn

gnb_dict = {
    'priors': None,        # Array of dimension equal to the number of classes.
                           # It contains the prior distributions of the classes.
    'var_smoothing': 1e-9  # Do not touch :)
}
'''

gnb_dict = {
    'priors': None,
    'var_smoothing': 1e-9
}

USE_CNN = False
OPTIMAL_FEATURE_NUMBER_LDA = 9

sets, class_names = load_data(eval_percentage=0.2)

if USE_CNN:

    # Compute high level features
    feature_extractor = CNN()
    sets.train.x, sets.eval.x, sets.test.x = feature_extractor.extract(sets.train.x, sets.eval.x, sets.test.x)

gnb_classifier = Classifier(Classifier.GAUSSIAN_NAIVE_BAYES, **gnb_dict)

feature_selector = FeaturesSelector(FeaturesSelector.LDA, OPTIMAL_FEATURE_NUMBER_LDA)
sets = feature_selector.fit(sets)

train_predict, eval_predict, test_predict = gnb_classifier.get_predictions(features=sets.train.x,
                                                                           labels=sets.train.y,
                                                                           eval_features=sets.eval.x,
                                                                           eval_labels=sets.eval.y,
                                                                           test_features=sets.test.x)

train_accuracy = sum([train_predict[i] == sets.train.y[i] for i in range(len(train_predict))]) / len(train_predict)
eval_accuracy = sum([eval_predict[i] == sets.eval.y[i] for i in range(len(eval_predict))]) / len(eval_predict)
test_accuracy = sum([test_predict[i] == sets.test.y[i] for i in range(len(test_predict))]) / len(test_predict)

print("\n\n\nTrain Accuracy: {}".format(train_accuracy))
print("Validation Accuracy: {}".format(eval_accuracy))
print("Test Accuracy: {}".format(test_accuracy))

print("\n\n\nScore: {}".format(gnb_classifier.classifier.score(sets.test.x, sets.test.y)))
