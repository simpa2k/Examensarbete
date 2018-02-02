import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, RepeatedKFold
from sklearn import datasets
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from src.experiments.posnett_hindle_devanbu import ten_times_ten_fold_cross_validation
from src.experiments.posnett_hindle_devanbu import visualize_results


def run():
    dataset = datasets.load_breast_cancer()

    clf = LogisticRegression()
    ten_times_ten_fold_cross_validation(dataset.data, dataset.target, clf)


if __name__ == "__main__":
    run()