from sklearn import linear_model, discriminant_analysis,  naive_bayes, svm

SVM = 'svm'
LDA = 'lda'
QDA = 'qda'
LOGISTIC = 'logistic'
NAIVE_BAYES = 'naive_bayes'
LINEAR = 'linear'


# interface function for classifiers
def predict(kind, features, labels, **args):

    """
    USAGE
    x = [[1],[2],[3],[4]]
    y = [1,2,3,4]
    predictions = predict(SVM, x,y, gamma='scale', decision_function_shape='ovo')

    """

    if kind == LINEAR:
        classifier = linear_model

    elif kind == LDA:
        classifier = discriminant_analysis.LinearDiscriminantAnalysis(**args)

    elif kind == QDA:
        classifier = discriminant_analysis.QuadraticDiscriminantAnalysis(**args)

    elif kind == LOGISTIC:
        classifier = linear_model.LogisticRegression(**args)

    elif kind == NAIVE_BAYES:
        # TODO gaussian and complement naive bayes
        classifier = naive_bayes.MultinomialNB(**args)

    elif kind == SVM:
        classifier = svm.SVC(**args)

    else:
        raise NotImplementedError

    classifier.fit(features, labels)

    return classifier.predict(features)

