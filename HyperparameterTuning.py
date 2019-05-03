from FeaturesSelector import FeaturesSelector
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Set first number of feature selected
START = 1

# Set number of feature increment at each iterations
STEP = 10


def tune(classifier, selector, sets):

    training_error = []
    validation_error = []

    best_training_accuracy = 0
    best_validation_accuracy = 0

    for i in range(START, sets.x.size, STEP):

        try:
            # Extract reduced features
            feature_selector = FeaturesSelector(selector, i)
            reduced_sets = feature_selector.fit(sets)

            # Predict with reduced features
            training_predictions, validation_predictions = \
                classifier.get_predictions(reduced_sets.train.x, reduced_sets.train.y, reduced_sets.eval.x)

            training_accuracy = accuracy_score(training_predictions, reduced_sets.train.y)
            validation_accuracy = accuracy_score(validation_predictions, reduced_sets.eval.y)

            # Add errors results to array
            training_error.append(1 - training_accuracy)
            validation_error.append(1 - validation_accuracy)

            print("\rIteration: ".format(i), str(i), end="")

            # Print training error and best training error
            print("   Training error: ".format(1 - training_accuracy)
                  + str(1 - training_accuracy), end="")

            if training_accuracy > best_training_accuracy:
                best_training_accuracy = training_accuracy

            print("  (Best Training error".format(1 - best_training_accuracy)
                  + str(best_training_accuracy) + ")", end="")

            # Print validation error and best validation error
            print("   Validation error: ".format(1 - validation_accuracy)
                  + str(1 - validation_accuracy), end="")

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy

            print("  (Best Validation error".format(1 - best_validation_accuracy)
                  + str(best_validation_accuracy) + ")", end="")

            # Save best number of features approximated to step resolution (real value with step=1)
            best_features_number = STEP * validation_error.index(min(validation_error))

        except KeyboardInterrupt:

            break

    print("\nMinimum error for " + str(best_features_number) + " features")

    plt.plot(range(0, sets.x.size-1, STEP), validation_error)
    plt.plot(range(0, sets.x.size-1, STEP), training_error)
    plt.show

    return best_features_number

