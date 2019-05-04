from Classifier import Classifier
from DataHandler import load_data

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
    'solver': 'svd',
    'shrinkage': None,
    'priors': None,
    'n_components': None,
    'store_covariance': False,
    'tol': 0.0001,
}

sets, class_names = load_data(linearized=True)

# Create Classifier
lor_classifier = Classifier('lda', **lda_dict)

# Predict the training, evaluation and test set
train_predict, eval_predict, test_predict = lor_classifier.get_predictions\
    (features=sets.train.x, labels=sets.train.y, eval_features=sets.eval.x, test_features=sets.test.x)

train_accuracy = sum([train_predict[i] == sets.train.y[i] for i in range(len(train_predict))])/len(train_predict)
eval_accuracy = sum([eval_predict[i] == sets.eval.y[i] for i in range(len(eval_predict))])/len(eval_predict)
test_accuracy = sum([test_predict[i] == sets.test.y[i] for i in range(len(test_predict))])/len(test_predict)

print("\nTrain Accuracy: {}".format(train_accuracy))
print("Validation Accuracy: {}".format(eval_accuracy))
print("Test Accuracy: {}".format(test_accuracy))
