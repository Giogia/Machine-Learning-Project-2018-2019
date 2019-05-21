from sklearn import linear_model, discriminant_analysis, naive_bayes, svm
import numpy as np


# Interface class for classifiers
class Classifier:

    """
    USAGE

    classifier = Classifier(SVM, gamma='scale', decision_function_shape='ovo')

    train_predictions = classifier.get_predictions(set.train.x, set.train.y)
    or
    train_predictions, eval_predictions = classifier.get_predictions(set.train.x,
                                                                     set.train.y,
                                                                     eval_features=set.eval.x)
    or
    train_predictions, eval_predictions, test_predictions = classifier.get_predictions(set.train.x, set.train.y,
                                                                                       eval_features=set.eval.x,
                                                                                       eval_labels=set.eval.y,
                                                                                       test_features=set.test.x)
    """

    SVM = 'svm'
    LDA = 'lda'
    QDA = 'qda'
    LOGISTIC = 'logistic'
    GAUSSIAN_NAIVE_BAYES = 'gaussian_naive_bayes'
    LINEAR = 'linear'

    # Create classifier object from SciKitLearn library
    def __init__(self, kind, **kwargs):

        if kind == Classifier.LINEAR:
            self.classifier = linear_model.LinearRegression(**kwargs)

        elif kind == Classifier.LDA:
            self.classifier = discriminant_analysis.LinearDiscriminantAnalysis(**kwargs)

        elif kind == Classifier.QDA:
            self.classifier = discriminant_analysis.QuadraticDiscriminantAnalysis(**kwargs)

        elif kind == Classifier.LOGISTIC:
            self.classifier = linear_model.LogisticRegression(**kwargs)

        elif kind == Classifier.GAUSSIAN_NAIVE_BAYES:
            # TODO gaussian and complement naive bayes
            self.classifier = naive_bayes.GaussianNB(**kwargs)

        elif kind == Classifier.SVM:
            self.classifier = svm.SVC(**kwargs)

        else:
            raise NotImplementedError

    # Return training predictions and evaluation or test prediction
    def get_predictions(self, features, labels, eval_features=None, eval_labels=None, test_features=None):

        # Here the model is trained
        self.classifier.fit(features, labels)

        if eval_features is None and test_features is None:

            # Returns only the prediction of the training set
            return self.classifier.predict(features)

        elif test_features is None:

            # Returns the predictions of both the the training set and the evaluation set
            return self.classifier.predict(features), self.classifier.predict(eval_features)

        else:
            predictions = [self.classifier.predict(features), self.classifier.predict(eval_features)]

            features = np.append(features, eval_features, axis=0)
            labels = np.append(labels, eval_labels)

            # Here the model is re-trained because for the prediction of the test set
            # Both the training and the evaluation set need to be used
            self.classifier.fit(features, labels)

            predictions.append(self.classifier.predict(test_features))
            return predictions
