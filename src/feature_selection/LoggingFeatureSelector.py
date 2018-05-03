import numpy as np
import matplotlib.pyplot as plt


class LoggingFeatureSelector:

    selected_feature_counts = {}
    scores_by_feature_set = {}
    selected_features_by_cv = {}
    feature_ranking_by_cv = {}
    feature_importance_by_cv = {}

    cv_counter = 0

    def __init__(self, estimator):
        self.estimator = estimator

    def transform(self, X):
        selected_features = tuple(self.estimator.get_support())

        if selected_features not in LoggingFeatureSelector.selected_feature_counts:
            LoggingFeatureSelector.selected_feature_counts[selected_features] = 1
        else:
            LoggingFeatureSelector.selected_feature_counts[selected_features] += 1

        LoggingFeatureSelector.scores_by_feature_set.setdefault(selected_features, []).append(np.max(self.estimator.grid_scores_))
        self.log_by_cv(LoggingFeatureSelector.selected_features_by_cv, selected_features)
        self.log_by_cv(LoggingFeatureSelector.feature_ranking_by_cv, self.estimator.ranking_)

        LoggingFeatureSelector.cv_counter += 1

        return self.estimator.transform(X)

    def log_by_cv(self, dict, value):
        dict.setdefault((LoggingFeatureSelector.cv_counter // 10) % 10, [])\
            .append(value)

    def fit_transform(self, X, y=None):
        results = self.estimator.fit_transform(X, y)
        self.log_by_cv(LoggingFeatureSelector.feature_importance_by_cv, self.estimator.estimator_.feature_importances_)

        return results

    def fit(self, X, y=None):
        return self.estimator.fit(X, y)
