from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class FeaturesSelector:

    """
    Usage

    feat_sel = FeatureSelector(kind, n_comp)
    kind -> FeatureSelector.LDA
            FeatureSelector.PCA
            FeatureSelector.NO_REDUCTION
    using a kind different then the ones provided will generate a NotImplemented exception

    After defining the object FeatureSelector (feat_sel), call the fit method by giving as input the sets objects,
    it will return a new object with the "training x" and "eval x" reduced

    """

    LDA = 'lda'
    PCA = 'pca'
    NO_REDUCTION = 'no_reduction'

    def __init__(self, kind, n_comp):
        self.kind = kind
        self.n_comp = n_comp

    def fit(self, sets):

        if self.kind == FeaturesSelector.NO_REDUCTION:
            x_train_red = sets.train.x
            x_eval_red = sets.eval.x
            x_test_red = sets.test.x
        else:
            if self.kind == FeaturesSelector.LDA:
                selector = LinearDiscriminantAnalysis(n_components=self.n_comp)

            elif self.kind == FeaturesSelector.PCA:
                selector = decomposition.PCA(n_components=self.n_comp)

            else:
                raise NotImplementedError

            x_train_red = selector.fit_transform(sets.train.x, sets.train.y)
            x_eval_red = selector.transform(sets.eval.x)
            x_test_red = selector.transform(sets.test.x)

        return sets.dup(x_train_red, x_eval_red, x_test_red)
