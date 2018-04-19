import itertools

import numpy as np
from scipy.stats import spearmanr, pearsonr


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
            self.feature_subset_merits(X, y)

        return transformed_X

    def feature_subset_merits(self, X, y):
        all_project = X[:, [0, 1, 2]]
        all_mean = X[:, [3, 4, 2]]
        project_loc_mean_V = X[:, [0, 4, 2]]
        mean_loc_project_V = X[:, [3, 1, 2]]

        ms1 = self.feature_subset_merit(all_project, y)
        ms2 = self.feature_subset_merit(all_mean, y)
        ms3 = self.feature_subset_merit(project_loc_mean_V, y)
        ms4 = self.feature_subset_merit(mean_loc_project_V, y)

        x = 1

    def feature_subset_merit(self, X, y):
        k = X.shape[1]
        rcf = self.mean_feature_class_correlation(X, y)
        rff = self.mean_feature_feature_correlation(X)
        return (k * rcf) / (np.sqrt(k + k * (k - 1) * rff))

    def mean_feature_class_correlation(self, X, y):
        #return np.array([spearmanr(column, y) for column in X.T]).mean()
        return np.array([pearsonr(column, y) for column in X.T]).mean()

    def mean_feature_feature_correlation(self, X):
        #return np.array([spearmanr(f1, f2) for f1, f2 in itertools.combinations(X.T, r=2)]).mean()
        return np.array([pearsonr(f1, f2) for f1, f2 in itertools.combinations(X.T, r=2)]).mean()

    def fit(self, X, y=None):
        return self
