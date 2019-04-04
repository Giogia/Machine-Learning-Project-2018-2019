from sklearn import linear_model, discriminant_analysis,  naive_bayes, svm

SVM = 'svm'
LDA = 'lda'
QDA = 'qda'
LOGISTIC = 'logistic'
NAIVE_BAYES = 'naive_bayes'
LINEAR = 'linear'

# interface function for classifiers
def predict(kind, X, Y, **args):

    if kind == LINEAR:
        classifier = linear_model

    elif kind == LDA:
        classifier = discriminant_analysis.LinearDiscriminantAnalysis(**args)

    elif kind == QDA:
        classifier = discriminant_analysis.QuadraticDiscriminantAnalysis(**args)

    elif kind == LOGISTIC:
        classifier = linear_model.LogisticRegression(**args)

    elif kind == NAIVE_BAYES:
        #TODO gaussian and complement naive bayes
        classifier =  naive_bayes.MultinomialNB(**args)

    elif kind == SVM:
        classifier = svm.SVC(**args)

    else:
        raise NotImplementedError

    classifier.fit(X,Y)

    return classifier.predict(X)

'''USAGE
X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
predictions = predict(SVM, X,Y, gamma='scale', decision_function_shape='ovo')
print(predictions)
'''