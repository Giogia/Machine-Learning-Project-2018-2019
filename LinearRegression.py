from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

from DataHandler import load_data
from CNN_v1 import CNN
from Classifier import Classifier
from FeaturesSelector import FeaturesSelector

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

USE_CNN = False
OPTIMAL_FEATURE_NUMBER_PCA = 570 if not USE_CNN else 1000

sets, class_names = load_data(eval_percentage=0.2)

if USE_CNN:

    # Compute high level features
    feature_extractor = CNN()
    sets.train.x, sets.eval.x, sets.test.x = feature_extractor.extract(sets.train.x, sets.eval.x, sets.test.x)

linear_classifier = Classifier(Classifier.LINEAR, **linear_dict)

feature_selector = FeaturesSelector(FeaturesSelector.PCA, OPTIMAL_FEATURE_NUMBER_PCA)
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

'''
print('Original: \n')

linear_model = LinearRegression() #declare a linear model
linear_model.fit(sets.train.x, sets.train.y) #fit the training set (X = n_samples ,Y = n_outputs)
y_prediction = linear_model.predict(sets.test.x) #predict Y from the test set




# The coefficients
print('Coefficients: \n', linear_model.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(sets.test.x, y_prediction))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(sets.test.y, y_prediction))

# Plot outputs
plt.scatter(sets.test.x, sets.test.y, color='black')
plt.plot(sets.test.x, y_prediction, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

'''