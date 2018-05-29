import os
import numpy as np
import pandas as pd

from src.feature_selection.LoggingFeatureSelector import LoggingFeatureSelector


def save_as_csv(output_path, filename, data, header):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    np.savetxt(os.path.join(output_path, filename),
               data,
               header=header,
               comments='',
               delimiter=',',
               fmt='%5.2f')


def save_fig(output_path, filename, plt):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plt.savefig(os.path.join(output_path, filename), transparent=True)


def save_scores_as_csv(results, output_path, k_fold_label):
    for score_label, scoring in results.items():
        scoring = pd.DataFrame(scoring.loc[0])
        scoring.index = np.append([i for i in range(1, 11)], ['Medelvärde', 'Standardavvikelse'])

        scoring.index.name = 'Del'
        scoring = scoring.rename(columns={0: score_label})

        scoring = scoring.round(2)

        score_directory = os.path.join(output_path, score_label)
        if not os.path.exists(score_directory):
            os.makedirs(score_directory)

        scoring.transpose().to_csv(os.path.join(score_directory, 'results.csv'), index=False)
        scoring.iloc[0:10].to_csv(os.path.join(score_directory, 'plottable_results.csv'))
        scoring[score_label].iloc[0:9].to_csv(os.path.join(score_directory, 'boxplottable_results.csv'), index=False)
        scoring.iloc[10:13].transpose().to_csv(os.path.join(score_directory, 'errorplottable_results.csv'), index=False)

        results[score_label] = scoring

    return results


def save_results(results, output_path, k_fold_label):
    return save_scores_as_csv(results, output_path, k_fold_label)


def save_common_results_without_transpose(data_frames, output_directory, results_filename, errorplottable_results_filename, ignore_index=False, separator=','):
    concatenated = pd.concat(data_frames, ignore_index=ignore_index)

    concatenated.index.name = 'x'

    concatenated.to_csv(os.path.join(output_directory, results_filename), sep=separator)

    errorplottable = concatenated.copy()[['Medelvärde', 'Standardavvikelse']]
    errorplottable.columns = ['mean', 'std']

    errorplottable.to_csv(os.path.join(output_directory, errorplottable_results_filename), sep=separator)


def save_common_results_with_and_without_index(data_frames, output_directory, filename, separator=','):
    save_common_results(data_frames, output_directory, 'with_index_{}'.format(filename), separator=separator)
    save_common_results(data_frames, output_directory, 'without_index_{}'.format(filename), ignore_index=True, separator=separator)


def save_common_results(data_frames, output_directory, filename, ignore_index=False, separator=','):
    concatenated = pd.concat(data_frames, ignore_index=ignore_index)
    concatenated.index.name = 'x'
    concatenated.to_csv(os.path.join(output_directory, filename), sep=separator, decimal=',')


def save_feature_selection_results(output_directory):
    feature_names = np.array(['\({}\)'.format(feature_name) for feature_name in ['R_P', 'V_P', 'E_P', '\overline{R_M}', '\overline{V_M}']])

    feature_counts = pd.DataFrame.from_dict(LoggingFeatureSelector.selected_feature_counts, orient='index')
    feature_counts.columns = ['Antal']
    feature_counts.index = [' '.join(feature_name for feature_name in feature_names[np.array(mask)]) for mask in
                            list(feature_counts.index)]
    feature_counts.index.name = 'Egenskapsuppsättning'

    features_by_first_cv = pd.DataFrame([' '.join(feature_name for feature_name in feature_names[np.array(mask)]) for mask in LoggingFeatureSelector.selected_features_by_cv[0]]).transpose()
    features_by_first_cv.columns = [str(i) for i in range(1, 11)]

    mean_feature_importances_by_cv = [np.mean(importances, axis=0) for importances in LoggingFeatureSelector.feature_importance_by_cv.values()]
    print(mean_feature_importances_by_cv)

    output_path = os.path.join(output_directory, 'feature_selection')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    feature_counts.to_csv(os.path.join(output_path, 'featureset_counts.csv'))
    features_by_first_cv.to_csv(os.path.join(output_path, 'features_by_cv.csv'), index=False)
