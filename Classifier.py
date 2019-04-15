from sklearn import linear_model, discriminant_analysis, naive_bayes, svm


# interface class for classifiers
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

    SVM = 'svm'
    LDA = 'lda'
    QDA = 'qda'
    LOGISTIC = 'logistic'
    GAUSSIAN_NAIVE_BAYES = 'gaussian_naive_bayes'
    LINEAR = 'linear'

    # create classifier object from SciKitLearn library
    def __init__(self, kind, **kwargs):

        if kind == Classifier.LINEAR:
            # TODO find correct class for linear classifier
            self.classifier = linear_model.RidgeClassifier(**kwargs)

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

    # return training predictions and
    def get_predictions(self, features, labels, eval_features=None):

        self.classifier.fit(features, labels)

        if eval_features is None:
            return self.classifier.predict(features)
        else:
            return self.classifier.predict(features), self.classifier.predict(eval_features)
