from DataHandler import load_data
from CNN import CNN
from FeaturesSelector import FeaturesSelector
from Classifier import Classifier


'''
The default dictionary for the Linear Discriminant Analysis parameters in scikit-learn

qda_dict = {
    'priors' : None              # Array, shape = [n_classes], priors on classes
    'reg_param' : 0.0            # Regularizes the covariance estimate as (1-reg_param)*Sigma + reg_param*np.eye(n_features)
    'store_covariance' : False   # If True the covariance matrices are computed and stored in the self.covariance_ attribute
    'tol' : 0.0001               # Threshold used for rank estimation.
}
'''

qda_dict = {
    'priors': None,
    'reg_param': 0.0,
    'store_covariance': False,
    'tol': 0.0001,
}

OPTIMAL_FEATURE_NUMBER_PCA = 500

# Linearized must be True if not using CNN otherwise False
sets, class_names = load_data(eval_percentage=0.2, linearized=False)

# Create features extractor
feature_extractor = CNN()

# Compute high level features
sets.train.x, sets.eval.x, sets.test.x = feature_extractor.extract(sets.train.x, sets.eval.x, sets.test.x)

# Create Classifier
classifier = Classifier(Classifier.QDA, **qda_dict)

# Create feature selector
feature_selector = FeaturesSelector(FeaturesSelector.PCA, OPTIMAL_FEATURE_NUMBER_PCA)

# Extract most significant features
sets = feature_selector.fit(sets)

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