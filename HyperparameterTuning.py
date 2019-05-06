from FeaturesSelector import FeaturesSelector
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Set minimum number of feature selected
START = 1

# Set maximum number of feature selected
END = 0

# Set number of feature increment at each iterations
STEP = 10


def tune(classifier, selector, sets):

    training_error = []
    validation_error = []

    best_training_accuracy = 0
    best_validation_accuracy = 0

    global END

    END = sets.train.x.shape[1]

    for i in range(START, END, STEP):

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

            print("\rNumber of features: ".format(i), str(i), end="")

            # Print training error and best training error
            print("   Training error: ".format(1 - training_accuracy)
                  + str(1 - training_accuracy), end="")

            if training_accuracy > best_training_accuracy:
                best_training_accuracy = training_accuracy

            print("  (Best Training error: ".format(1 - best_training_accuracy)
                  + str(1 - best_training_accuracy) + ")", end="")

            # Print validation error and best validation error
            print("   Validation error: ".format(1 - validation_accuracy)
                  + str(1 - validation_accuracy), end="")

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy

            print("  (Best Validation error: ".format(1 - best_validation_accuracy)
                  + str(1 - best_validation_accuracy) + ")", end="")

        except KeyboardInterrupt:

            break

    # Save best number of features approximated to step resolution (real value with step=1)
    best_features_number = STEP * validation_error.index(min(validation_error))

    print("\nMinimum error for " + str(best_features_number) + " features")

    plt.plot(range(0, END - 1, STEP), validation_error)
    plt.plot(range(0, END - 1, STEP), training_error)
    plt.show()
    plt.savefig(classifier.kind + '_' + selector + '.png')

    return best_features_number

