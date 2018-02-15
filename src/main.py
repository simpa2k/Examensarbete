import argparse
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

from src.datasets import posnett_hindle_devanbu
from src.featurizers.bow_featurize import featurize
from src.loaders.file_loader import create_file_loader
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline


def default_pipeline_of(estimator):
    return Pipeline(steps=[('scale', StandardScaler()), ('estimator', estimator)])


estimator_labels = ['mlpc', 'nb']
estimators = [
    default_pipeline_of(MLPClassifier()),
    MultinomialNB()
]
estimators_by_label = dict(zip(estimator_labels, estimators))

dataset_labels = ['phd']
datasets = [posnett_hindle_devanbu.get]
datasets_by_label = dict(zip(dataset_labels, datasets))


def setup_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-root', help='Path to the common parent directory of everything to be used as data.',
                        default='../data/bw')
    parser.add_argument('--document-directory',
                        help='Path to the directory containing the documents to be processed, relative to the data-root.',
                        default='/snippets')
    parser.add_argument('--annotation-directory',
                        help='Path to the correct labels of the documents to be processed, relative to the data-root.',
                        default='/votes.csv')
    parser.add_argument('--estimator', help='The machine learning algorithm to be used.',
                        choices=estimator_labels, required=True)
    parser.add_argument('--dataset', help='A function that loads the data to be used.',
                        default='phd', choices=dataset_labels)

    return parser


def perform_experiment(X, y, estimator):
    seed = 3
    kfold = KFold(n_splits=10, random_state=seed)

    cv_results = cross_val_score(estimator, X, y, cv=kfold)
    print(cv_results.mean())


def main():
    parser = setup_parser()
    args = parser.parse_args()

    estimator = estimators_by_label[args.estimator]
    dataset = datasets_by_label[args.dataset]
    X, y = dataset(args.data_root, args.document_directory, args.annotation_directory)

    perform_experiment(X, y, estimator)


if __name__ == '__main__':
    main()
