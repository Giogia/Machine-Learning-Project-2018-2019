from DataHandler import load_data
from CNN import CNN
from Classifier import Classifier
from FeaturesSelector import FeaturesSelector

'''
The default dictionary for the Logistic Regression parameters in scikit-learn

lor_dict = {
    'penalty': 'l2',        # 'l1' or 'l2'
    'dual': False,          # True if #feature > #samples (only if l2 active)
    'tol': 1e-4,            # tollerance for early stopping
    'C': 1.0,               # inverse of the regularization term of LoR smaller values implies stronger regularization
    'fit_intercept': True,  # True if we want the Bias
    'intercept_scaling': 1, # Useful only when solver 'liblinear'.
                            # The higher it is, the less the bias are regularized, the bigger they can become
    'class_weight': None,   # None,'balanced' or dict. For giving a weight to the various classes.
                            # Usefull if unbalanced datsets.
    'random_state': None,   # Seed for initializing the random generator --> for experiments reproducibility
    'solver': 'warn',       # 'newton-cg','lbfgs','liblinear','sag','saga'.
                            # 'liblinear' good for small datasets.
                            # 'sag' and 'saga' are faster on big datasets.
                            # 'liblinear' has a one-vs-rest approach, the others use multinnomials.
                            # 'newton-cg', 'lbfgs' and 'sag' use just L2 regularization.
                            # 'sag' and 'saga' guarantee a fast convergence if the features are approximated on the same scale.
    'max_iter': 100,        # maximum number of iterations
    'multi_class': 'auto',  # 'ovr', 'multinomial' or 'auto'. We should use multinomial.
                            # If we use liblinear solver, we have to use ovr.
    'verbose': 0,           # 0,1 o 2. Levels of verbosity.
    'warm_start': False,    # It True it reuse the solution of previous fit.
    'n_jobs': None          # Number of processors used by the computation.
}

'''

lor_dict = {
    'penalty': 'l2',
    'dual': False,
    'tol': 1e-4,
    'C': 0.5,
    'fit_intercept': True,
    'intercept_scaling': 1,
    'class_weight': None,
    'random_state': None,
    'solver': 'warn',
    'max_iter': 100,
    'multi_class': 'auto',
    'verbose': 1,
    'warm_start': False,
    'n_jobs': None
}


USE_CNN = True
OPTIMAL_FEATURE_NUMBER_PCA = 370 if not USE_CNN else 1000

sets, class_names = load_data()

if USE_CNN:

    # Compute high level features
    feature_extractor = CNN()
    sets.train.x, sets.eval.x, sets.test.x = feature_extractor.extract(sets.train.x, sets.eval.x, sets.test.x)

lor_classifier = Classifier(Classifier.LOGISTIC, **lor_dict)

feature_selector = FeaturesSelector(FeaturesSelector.PCA, OPTIMAL_FEATURE_NUMBER_PCA)
sets = feature_selector.fit(sets)

train_predict, eval_predict, test_predict = lor_classifier.get_predictions(features=sets.train.x,
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

print("\n\n\nScore: {}".format(lor_classifier.classifier.score(sets.test.x, sets.test.y)))
