import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict

from src.utils.save_data import save_results

cross_validation_column_label = 'cv'
cross_validation_fold_column_label = 'fold'
prediction_column_label = 'pred'

predicted_class_label = 'Class'
document_label = 'Document'


def perform(experiment, dataset, data_root, annotation_path,  output_path, scoring_directory='scoring', force_feature_generation=False):
    dataset.load(data_root, annotation_path, force_feature_generation)
    dataset.describe(output_path)

    experiment(dataset, dataset.get_annotations(), output_path, scoring_directory)


def run(X, y, estimator, scoring, annotations_by_document, output_directory, scoring_directory):
    results, predictions = repeated_stratified_kfold_with_different_seeds(X, y, estimator, scoring)

    for score_label, scoring in results.items():
        print('Mean {} score was: {}'.format(score_label, scoring['mean'].mean()))

    formatted_results = save_results(results, os.path.join(output_directory, scoring_directory), '')

    pd.concat([annotations_by_document.reset_index(), pd.DataFrame(predictions).transpose().reset_index(drop=True)],
              axis=1)\
        .set_index('document')\
        .to_csv(os.path.join(output_directory, scoring_directory, 'predictions.csv'))

    return formatted_results


def repeated_stratified_kfold_with_different_seeds(X, y, estimator, scoring):
    columns = np.append([i for i in range(0, 10)], ['mean', 'std'])

    predictions = []
    results = {}
    for score in scoring.keys():
        results[score] = pd.DataFrame(columns=columns)

    for i in range(0, 10):
        k_fold = StratifiedKFold(n_splits=10, random_state=i)

        predictions.append(cross_val_predict(estimator, X, y, cv=k_fold))
        scores = cross_validate(estimator, X, y, cv=k_fold, scoring=scoring, return_train_score=False)

        for score in scoring.keys():
            results[score] = results[score].append(
                pd.DataFrame(
                    [[score for score in np.append(
                        scores['test_{}'.format(score)],
                        [scores['test_{}'.format(score)].mean(),
                         scores['test_{}'.format(score)].std()]
                    )]],
                    columns=columns),
                ignore_index=True
            )

    return results, predictions

