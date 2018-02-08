import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, interp
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
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
                               scoring='roc_auc', n_jobs=n_jobs,
                               param_grid=dict(clf__C=C))

    # Fit the Model & Plot the Results
    ###############################################################################
    log_clf_est.fit(X_train, y_train)

    # plotting results
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


def several_different(X, y):

    test_size = 0.20
    seed = 7

    models = [{'name': 'LR',
               'model': Pipeline(steps=[('scale', StandardScaler()), ('m', LogisticRegression())]),
               'grid': [{'m__C': [1, 10, 100, 1000]}]
               },
              {'name': 'LDA',
               'model': Pipeline(steps=[('scale', StandardScaler()), ('m', LinearDiscriminantAnalysis())]),
               'grid': [{'m__solver': ['svd', 'lsqr', 'eigen']}]
               },
              {'name': 'KNN',
               'model': Pipeline(steps=[('scale', StandardScaler()), ('m', KNeighborsClassifier())]),
               # 'grid': [{'m__weights': ['uniform', 'distance'], 'm__algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto']}]},
               'grid': []
               },
              {'name': 'CART',
               'model': Pipeline(steps=[('scale', StandardScaler()), ('m', DecisionTreeClassifier())]),
               'grid': []
               },
              {'name': 'NB',
               'model': Pipeline(steps=[('scale', StandardScaler()), ('m', GaussianNB())]),
               'grid': []
               },
              {'name': 'SVM',
               'model': Pipeline(steps=[('scale', StandardScaler()), ('m', SVC())]),
               'grid': []
               },
              {'name': 'MLPC',
               'model': Pipeline(steps=[('scale', StandardScaler()), ('m', MLPClassifier())]),
               'grid': []
               },
              {'name': 'RFC',
               'model': Pipeline(steps=[('scale', StandardScaler()), ('m', RandomForestClassifier())]),
               'grid': []
               }
              ]

    results = []
    names = []
    num_runs = 10

    for config in models:

        model_results = np.array([])

        for i in range(0, num_runs):

            kfold = KFold(n_splits=10, random_state=i)
            estimator = GridSearchCV(estimator=config['model'],
                                     cv=kfold,
                                     scoring='accuracy',
                                     param_grid=config['grid'])

            # cv_results = cross_val_score(estimator, X, y, cv=kfold2)
            estimator.fit(X, y)
            cv_results = estimator.best_score_

            model_results = np.append(model_results, cv_results)

        mean = model_results.mean()
        results.append(mean)
        names.append(config['name'])

        msg = "%s: %f (%f)" % (config['name'], mean, model_results.std())
        print(msg)


def run():
    np.set_printoptions(suppress=True)

    data_root = "../../data/bw"

    load_data = create_file_loader(data_root + "/snippets", data_root + "/votes.csv")
    documents, votes = load_data()

    features = featurize(documents)
    votes = prepare_target_data(votes)

    # from_teh_internetz(features, votes)
    several_different(features, votes)


if __name__ == "__main__":
    run()
