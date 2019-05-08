from Classifier import Classifier
from DataHandler import load_data

# The default dictionary for the Gaussian Naive Bayes parameters in scikit-learn
gnb_dict = {
    'priors': None,  # Array of dimension equal to the number of classes.
    # It contains the prior distributions of the classes.
    'var_smoothing': 1e-9  # Do not touch :)
}

sets, class_names = load_data(linearized=True)

gnb_classifier = Classifier(Classifier.GAUSSIAN_NAIVE_BAYES, **gnb_dict)
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
