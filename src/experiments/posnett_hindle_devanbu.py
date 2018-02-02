import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, interp
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import cross_validate, cross_val_predict, cross_val_score, KFold, StratifiedKFold, RepeatedKFold, train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, roc_auc_score, make_scorer
from src.featurizers.phd.posnett_hindle_devanbu_featurizer import featurize
from src.loaders.phd.buse_weimer_loader import create_file_loader


roc_auc = "roc_auc_weighted"
accuracy = "accuracy"
f1 = "f1_weighted"


def prepare_target_data(target_data):

    target_data = np.mean(target_data, axis=0)
    target_data = np.where(target_data < 3.14, 0, 1)
    target_data = np.reshape(target_data, (100,))

    return target_data


def visualize_results(scores, votes):

    # fpr, tpr, thresholds = roc_curve(votes, scores["test_f1"])

    # plt.plot(fpr, tpr)
    # plt.show()

    plt.boxplot(scores["test_" + f1])
    plt.show()

    plt.boxplot(scores["test_" + accuracy])
    plt.show()

    plt.boxplot(scores["test_" + roc_auc])
    plt.show()

    print("F1: ", scores["test_" + f1].mean())
    print("Accuracy: ", scores["test_" + accuracy].mean())
    print("Area under ROC curve: ", scores["test_" + roc_auc].mean())


def simple_split_evaluation(X, y, model):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0
    )

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    model.fit(X_train, y_train)

    # plotting results
    log_fpr, log_tpr, _ = roc_curve(y_test,
                                    model.predict_proba(X_test)[:, 1])
    log_roc_auc = auc(log_fpr, log_tpr)

    plt.plot(log_fpr, log_tpr, color='seagreen', linestyle='--',
             label='LOG (area = %0.2f)' % log_roc_auc, lw=2)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
             label='Luck')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves of Model')
    plt.legend(loc="lower right")
    plt.show()

    # score = model.score(X_test, y_test)

    # print(score)


def ten_times_ten_fold_cross_validation(X, y, model):

    custom_accuracy = make_scorer(accuracy_score)
    weighted_f1 = make_scorer(f1_score, average='weighted')
    weighted_roc_auc = make_scorer(roc_auc_score, average='weighted')

    scoring = {accuracy: custom_accuracy, f1: weighted_f1, roc_auc: weighted_roc_auc}
    scores = cross_validate(model, X, y, scoring=scoring, cv=RepeatedStratifiedKFold(n_splits=10,
                                                                                     n_repeats=10,
                                                                                     random_state=36851234))
    visualize_results(scores, y)


def grid_searched(X, y):
    clf = LogisticRegression()
    C = [0.001, 0.01, 10, 100, 1000]

    n_folds = 10
    n_repeats = 10

    skfold = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats,
                                     random_state=0)

    n_jobs = 4

    clf_pipe = Pipeline(steps=[('scale', StandardScaler()), ('clf', clf)])
    clf_est = GridSearchCV(estimator=clf_pipe, cv=skfold,
                           scoring='roc_auc', n_jobs=n_jobs,
                           param_grid=dict(clf__C=C))

    custom_accuracy = make_scorer(accuracy_score)
    weighted_f1 = make_scorer(f1_score, average='weighted')
    weighted_roc_auc = make_scorer(roc_auc_score, average='weighted')

    scoring = {accuracy: custom_accuracy, f1: weighted_f1, roc_auc: weighted_roc_auc}
    scores = cross_validate(clf_est, X, y, scoring=scoring, cv=RepeatedStratifiedKFold(n_splits=10,
                                                                                       n_repeats=10,
                                                                                       random_state=36851234))
    visualize_results(scores, y)


def from_teh_internetz(X, y):
    my_rand_state = 3

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=my_rand_state)

    # Define simple model
    ###############################################################################
    log_clf = LogisticRegression()
    C = [0.001, 0.01, 10, 100, 1000]

    # Simple pre-processing estimators
    ###############################################################################
    std_scale = StandardScaler()

    # Defining the CV method: Using the Repeated Stratified K Fold
    ###############################################################################
    n_folds = 10
    n_repeats = 30

    skfold = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats,
                                     random_state=my_rand_state)

    # Creating simple pipeline and defining the gridsearch
    ###############################################################################
    n_jobs = 4

    scoring = {'AUC': make_scorer(roc_auc_score, average='weighted'), 'F1': make_scorer(f1_score, average='weighted'), 'ACCURACY': make_scorer(accuracy_score)}

    log_clf_pipe = Pipeline(steps=[('scale', std_scale), ('clf', log_clf)])
    log_clf_est = GridSearchCV(estimator=log_clf_pipe, cv=skfold,
                               scoring=scoring, n_jobs=n_jobs,
                               param_grid=dict(clf__C=C),
                               refit='AUC')

    # Fit the Model & Plot the Results
    ###############################################################################
    # log_clf_est.fit(X_train, y_train)
    log_clf_est.fit(X, y)
    results = log_clf_est.cv_results_

    plt.figure(figsize=(13, 13))
    plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
              fontsize=16)

    plt.xlabel("clf__C")
    plt.ylabel("Score")
    plt.grid()

    ax = plt.axes()
    ax.set_xlim(0, 402)
    ax.set_ylim(0.73, 1)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results['param_clf__C'].data, dtype=float)

    for scorer, color in zip(sorted(scoring), ['g', 'k']):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                    (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid('off')
    plt.show()

    # plotting results
    """
    log_fpr, log_tpr, _ = roc_curve(y_test,
                                    log_clf_est.predict_proba(X_test)[:, 1])
    log_roc_auc = auc(log_fpr, log_tpr)

    plt.plot(log_fpr, log_tpr, color='seagreen', linestyle='--',
             label='LOG (area = %0.2f)' % log_roc_auc, lw=2)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
             label='Luck')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves of Model')
    plt.legend(loc="lower right")
    plt.show()
    """

def run():
    np.set_printoptions(suppress=True)

    data_root = "../../data/bw"

    load_data = create_file_loader(data_root + "/snippets", data_root + "/votes.csv")
    documents, votes = load_data()

    features = featurize(documents)
    votes = prepare_target_data(votes)

    breast_cancer = datasets.load_breast_cancer()

    from_teh_internetz(breast_cancer.data, breast_cancer.target)

    model = LogisticRegression(C=1000)
    # ten_times_ten_fold_cross_validation(features, votes, model)


if __name__ == "__main__":
    run()
