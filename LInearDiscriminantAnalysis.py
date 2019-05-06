from DataHandler import load_data
from CNN import CNN
from HyperparameterTuning import tune
from FeaturesSelector import FeaturesSelector
from Classifier import Classifier


'''
The default dictionary for the Linear Discriminant Analysis parameters in scikit-learn

lda_dict = {
    'solver': 'svd',             # 'svd', 'lsqr' or 'eigen'
    'shrinkage': None,           # Optional: 'auto' or float:[0,1]
    'priors': None,              # Class priors
    'n_components': None,        # Optional: Number of components (< n_classes - 1) for dimensionality reduction.
    'store_covariance': False,   # Additionally compute class covariance matrix, used only in ‘svd’ solver.
    'tol': 0.0001,               # Threshold used for rank estimation in SVD solver.
}
'''

lda_dict = {
    'solver': 'lsqr',
    'shrinkage': None,
    'priors': None,
    'n_components': None,
    'store_covariance': False,
    'tol': 0.0001,
}

# Linearized must be True if not using CNN otherwise False
sets, class_names = load_data(eval_percentage=0.2, linearized=False)

# Create features extractor
feature_extractor = CNN()

# Compute high level features
sets.train.x, sets.eval.x, sets.test.x = feature_extractor.extract(sets.train.x, sets.eval.x, sets.test.x)

# Create Classifier
classifier = Classifier('lda', **lda_dict)

# Tune feature selector parameter, disable feature selector prints for visually
optimal_features_number = tune(classifier, 'pca', sets)

# Create feature selector
feature_selector = FeaturesSelector('pca', optimal_features_number)

# Predict the training, evaluation and test set
train_predict, eval_predict, test_predict = classifier.get_predictions(features=sets.train.x,
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
