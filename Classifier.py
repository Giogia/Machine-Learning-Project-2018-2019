from sklearn import linear_model, discriminant_analysis,  naive_bayes, svm

SVM = 'svm'
LDA = 'lda'
QDA = 'qda'
LOGISTIC = 'logistic'
NAIVE_BAYES = 'naive_bayes'
LINEAR = 'linear'


# interface class for classifiers
class Classifier:

    """
    USAGE
    classifier = Classifier(SVM, gamma='scale', decision_function_shape='ovo')
    classifier.fit(set.train.x, set.train.y)
    predictions = classifier.predict(set.train.x)

    """

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

    def fit(self, features, labels):

        self.classifier.fit(features, labels)

    def predict(self, features):

        return self.classifier.predict(features)
