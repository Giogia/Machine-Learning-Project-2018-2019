from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class FeaturesSelector:

    LDA = 'lda'
    PCA = 'pca'
    NO_REDUCTION = ''

    def __init__(self, kind, n_comp):
        self.kind = kind
        self.n_comp = n_comp

    def fit(self, sets):

        if self.kind == FeaturesSelector.NO_REDUCTION:
            print("This is the shape of the training set (with NO reduction): ", sets.train.x.shape)
            x_train_red = sets.train.x
            print("This is the shape of the evaluation set (with NO reduction): ", sets.eval.x.shape)
            x_eval_red = sets.eval.x
        else:
            if self.kind == FeaturesSelector.LDA:
                selector = LinearDiscriminantAnalysis(n_components=self.n_comp)

            elif self.kind == FeaturesSelector.PCA:
                selector = decomposition.PCA(n_components=self.n_comp)
            else:
                raise NotImplementedError

            print("This is the old dimension of the training data: ", sets.train.x.shape)
            x_train_red = selector.fit_transform(sets.train.x, sets.train.y)
            print("This is the new dimension of the train data: ", x_train_red.shape)

            print("This is the old dimension of the eval data: ", sets.eval.x.shape)
            x_eval_red = selector.transform(sets.eval.x)
            print("This is the new dimension of the reduced eval data: ", x_eval_red.shape)

        return sets.dup(x_train_red, x_eval_red)



