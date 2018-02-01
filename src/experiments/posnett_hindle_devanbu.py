import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from src.featurizers.phd.posnett_hindle_devanbu_featurizer import featurize
from src.loaders.phd.buse_weimer_loader import create_file_loader


def run():
    data_root = "../../data/bw"

    load_data = create_file_loader(data_root + "/snippets", data_root + "/votes.csv")
    documents, votes = load_data()

    features = featurize(documents)
    votes = np.apply_along_axis(lambda votes_for_document: [np.sum(votes_for_document) / len(votes_for_document)],
                                0,
                                votes)

    votes = np.apply_along_axis(lambda avg_vote: [0 if avg_vote < 3.14 else 1], 0, votes)
    votes = np.reshape(votes, (100,))

    model = LogisticRegression()

    scoring = ["f1", "accuracy"]
    scores = cross_validate(model, features, votes, scoring=scoring, cv=RepeatedKFold(n_splits=10, n_repeats=10))

    fpr, tpr, thresholds = roc_curve(votes, scores["test_f1"])

    plt.plot(fpr, tpr)
    plt.show()

    plt.boxplot(scores["test_accuracy"])
    plt.show()

    y_scores = roc_auc_score(votes, scores["test_f1"])

    print("Area under ROC curve: ", y_scores)
    print("Percent correct: ", scores["test_accuracy"].mean())


if __name__ == "__main__":
    run()
