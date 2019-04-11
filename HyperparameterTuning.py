from FeaturesSelector import FeaturesSelector
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# set first number of feature selected
START = 1

# set number of feature increment at each iterations
STEP = 10


def tune(classifier, selector, sets):

    training_error = []
    validation_error = []

    i = START

    while i < sets.x.size:

        # extract reduced features
        feature_selector = FeaturesSelector(selector, i)
        reduced_sets = feature_selector.fit(sets)

        # predict with reduced features
        training_predictions, validation_predictions = \
            classifier.get_predictions(reduced_sets.train.x, reduced_sets.train.y, reduced_sets.eval.x)

        # add errors results to array
        training_error.append(1 - accuracy_score(training_predictions, reduced_sets.train.y))
        validation_error.append(1 - accuracy_score(validation_predictions, reduced_sets.eval.y))

        print(i, "Training error" + str(1 - accuracy_score(training_predictions, reduced_sets.train.y)),
                 "Validation error" + str(1 - accuracy_score(validation_predictions, reduced_sets.eval.y)))

        # save best number of features approximated to step resolution (real value with step=1)
        best_features_number = STEP * validation_error.index(min(validation_error))

        i += STEP

        print("Minimum error for " + str(best_features_number) + " features")

    plt.plot(range(0, sets.x.size-1, STEP), validation_error)
    plt.plot(range(0, sets.x.size-1, STEP), training_error)
    plt.show

    return best_features_number

