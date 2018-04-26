class LoggingFeatureSelector:

    def __init__(self, estimator):
        self.estimator = estimator

    def transform(self, X):
        print("Features selected: {}".format(self.estimator.get_support()))
        return self.estimator.transform(X)

    def fit_transform(self, X, y=None):
        return self.estimator.fit_transform(X, y)

    def fit(self, X, y=None):
        return self.estimator.fit(X, y)
