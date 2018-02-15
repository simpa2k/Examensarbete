from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from src.datasets.posnett_hindle_devanbu import ten_times_ten_fold_cross_validation


def run():
    dataset = datasets.load_breast_cancer()

    clf = LogisticRegression()
    ten_times_ten_fold_cross_validation(dataset.data, dataset.target, clf)


if __name__ == "__main__":
    run()