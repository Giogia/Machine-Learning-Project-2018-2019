from sklearn import linear_model, discriminant_analysis, naive_bayes, svm

SVM = 'svm'
LDA = 'lda'
QDA = 'qda'
LOGISTIC = 'logistic'
NAIVE_BAYES = 'naive_bayes'
LINEAR = 'linear'


# Interface class for classifiers
class Classifier:
    """
    USAGE

    classifier = Classifier(SVM, gamma='scale', decision_function_shape='ovo')

    train_predictions = classifier.get_predictions(set.train.x, set.train.y)
    or
    test_predictions = classifier.get_predictions(set.test.x, set.test.y)
    or
    train_predictions, eval_predictions = classifier.get_predictions(set.train.x, set.train.y, set.eval.x)

    """

    # Create classifier object from SciKitLearn library
    def __init__(self, kind, **args):

        if kind == LINEAR:
            # TODO find correct class for linear classifier
            self.classifier = linear_model.RidgeClassifier(**args)

        elif kind == LDA:
            self.classifier = discriminant_analysis.LinearDiscriminantAnalysis(**args)

        elif kind == QDA:
            self.classifier = discriminant_analysis.QuadraticDiscriminantAnalysis(**args)

        elif kind == LOGISTIC:
            self.classifier = linear_model.LogisticRegression(**args)

        elif kind == NAIVE_BAYES:
            # TODO gaussian and complement naive bayes
            self.classifier = naive_bayes.MultinomialNB(**args)

        elif kind == SVM:
            self.classifier = svm.SVC(**args)

        else:
            raise NotImplementedError

    # Return training predictions and evaluation or test prediction
    def get_predictions(self, features, labels, eval_features=None):

        self.classifier.fit(features, labels)

        if eval_features is None:
            return self.classifier.predict(features)

        else:
            return self.classifier.predict(features), self.classifier.predict(eval_features)
