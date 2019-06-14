from src.CNN_v1 import CNN
from src.Classifier import Classifier
from src.DataHandler import load_data
from src.FeaturesSelector import FeaturesSelector

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

USE_CNN = True
OPTIMAL_FEATURE_NUMBER_PCA = 695 if not USE_CNN else 970

sets, class_names = load_data(eval_percentage=0.2)

if USE_CNN:

    # Compute high level features
    feature_extractor = CNN()
    sets.train.x, sets.eval.x, sets.test.x = feature_extractor.extract(sets.train.x, sets.eval.x, sets.test.x)

lda_classifier = Classifier(Classifier.LDA, **lda_dict)

# Extract most significant features
feature_selector = FeaturesSelector(FeaturesSelector.PCA, OPTIMAL_FEATURE_NUMBER_PCA)
sets = feature_selector.fit(sets)

# Predict the training, evaluation and test set
train_predict, eval_predict, test_predict = lda_classifier.get_predictions(features=sets.train.x,
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
