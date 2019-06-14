from src.CNN_v1 import CNN
from src.Classifier import Classifier
from src.DataHandler import load_data
from src.FeaturesSelector import FeaturesSelector

'''
The default dictionary for the Support Vector Machine parameters in scikit-learn

svm_dict = {
    'C': 1.0,                          # Penalty parameter C of the error term.
    'kernel': 'rbf',                   # Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    'degree': 2,                       # Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
    'gamma': 'auto',                   # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
    'coef0': 0.0,                      # Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
    'shrinking': True,                 # Whether to use the shrinking heuristic.
    'probability': False,              # Whether to enable probability estimates.
    'tol': 0.001,                      # Tolerance for stopping criterion.
    'cache_size': 200,                 # Specify the size of the kernel cache (in MB).
    'class_weight': None,              # Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one.
    'verbose': False,                  # Enable verbose output.
    'max_iter': -1,                    # Hard limit on iterations within solver, or -1 for no limit.
    'decision_function_shape': 'ovr',  # return a one-vs-rest (‘ovr’) decision function or the original one-vs-one (‘ovo’) decision function of libsvm 
    'random_state': None,              # The seed of the pseudo random number generator used when shuffling the data for probability estimates.
}
'''

svm_dict = {
    'C': 1.0,
    'kernel': 'rbf',
    'degree': 2,
    'gamma': 'auto',
    'coef0': 0.0,
    'shrinking': True,
    'probability': False,
    'tol': 0.001,
    'cache_size': 200,
    'class_weight': None,
    'verbose': False,
    'max_iter': -1,
    'decision_function_shape': 'ovr',
    'random_state': None,
}

USE_CNN = False
OPTIMAL_FEATURE_NUMBER_PCA = 785 if not USE_CNN else 1000

sets, class_names = load_data(eval_percentage=0.2)

if USE_CNN:

    # Compute high level features
    feature_extractor = CNN()
    sets.train.x, sets.eval.x, sets.test.x = feature_extractor.extract(sets.train.x, sets.eval.x, sets.test.x)

gnb_classifier = Classifier(Classifier.SVM, **svm_dict)

feature_selector = FeaturesSelector(FeaturesSelector.LDA, OPTIMAL_FEATURE_NUMBER_PCA)
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
