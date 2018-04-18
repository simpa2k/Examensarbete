from scipy.stats import spearmanr


class OptimalHalsteadModelSelector:

    def transform(self, X):
        return X[:, [self.loc_metric, self.halstead_V_metric, self.entropy]]

    def fit_transform(self, X, y=None):
        transformed_X = X

        if X.shape[1] == 5:
            project_loc_correlation = spearmanr(X[:, 0], y)
            method_mean_loc_correlation = spearmanr(X[:, 3], y)

            project_V_correlation = spearmanr(X[:, 1], y)
            method_mean_V_correlation = spearmanr(X[:, 4], y)

            if abs(project_loc_correlation[0]) > abs(method_mean_loc_correlation[0]):
                self.loc_metric = 0
            else:
                self.loc_metric = 3

            if abs(project_V_correlation[0]) > abs(method_mean_V_correlation[0]):
                self.halstead_V_metric = 1
            else:
                self.halstead_V_metric = 4

            self.entropy = 2

            transformed_X = X[:, [self.loc_metric, self.halstead_V_metric, self.entropy]]

        return transformed_X

    def fit(self, X, y=None):
        return self
