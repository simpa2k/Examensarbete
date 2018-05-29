import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.save_data import save_fig


def generate_all(output_path):
    feature_sets(output_path)
    feature_scatters(output_path)
    #feature_correlations(output_path)
    #inter_annotator_agreement(output_path)


def feature_sets(output_path):
    plt.figure(figsize=(12, 6))

    experimental_data = pd.read_csv('olofsson_2018/scoring/with_index_selected_aggregated.csv', sep=';', decimal=',')
    experimental_data = experimental_data[['accuracym', 'tprm', 'tnrm']].values
    markers = ['o', '^', 'v']
    colors = ['blue', 'green', 'red']

    for i in range(experimental_data.shape[1]):
        plt.scatter(experimental_data[:, i], range(7), marker=markers[i], c=colors[i])

    feature_set_labels = [
        r'$R_P$',
        r'$E_P$',
        r'$\overline{V_M}$',
        r'$R_P, E_P, \overline{V_M}$',
        r'$V_P, \overline{R_M}, \overline{V_M}$',
        r'$V_P, E_P, \overline{R_M}, \overline{V_M}$',
        r'$R_P, V_P, E_P, \overline{R_M}, \overline{V_M}$'
    ]
    plt.yticks(range(len(feature_set_labels)), feature_set_labels)

    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()

    save_fig(output_path, 'feature_sets.png', plt)


def feature_scatters(output_path):
    plt.figure(figsize=(12, 9))

    correlations = pd.read_csv('olofsson_2018/features/all_features/correlations.csv', sep=';', decimal=',')

    data = pd.read_csv('olofsson_2018/annotations_and_all_features.csv')
    label = data['Bedömning']
    features = data.iloc[:, 2:data.shape[1]]

    feature_labels = [r'$R_P$', r'$V_P$', r'$E_P$', r'$\overline{R_M}$', r'$\overline{V_M}$']

    i = 1
    for feature in features.columns:
        plt.subplot(3, 2, i)
        plt.scatter(features[feature].values, label.values)

        plt.yticks(range(1, 6))
        plt.grid(True, alpha=0.3)

        plt.xlabel(feature_labels[i - 1])
        plt.ylabel('Bedömning')

        plt.title('ρ = {}'.format(correlations[feature].iloc[5]))

        i += 1

    plt.subplots_adjust(hspace=0.5)
    save_fig(output_path, 'feature_scatters.png', plt)


def feature_correlations(output_path):
    plt.figure(figsize=(12, 7))

    correlations = pd.read_csv('olofsson_2018/features/all_features/correlations.csv', sep=';', decimal=',')
    correlations = correlations.iloc[:, 1:correlations.shape[1]].values

    plt.matshow(correlations, cmap='RdBu')

    for (i, j), z in np.ndenumerate(correlations):
        plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')

    plt.colorbar()

    feature_labels = [r'$R_P$', r'$V_P$', r'$E_P$', r'$\overline{R_M}$', r'$\overline{V_M}$', 'Bedömning']

    plt.xticks(range(len(feature_labels)), feature_labels)
    plt.yticks(range(len(feature_labels)), feature_labels)

    #save_fig(output_path, 'feature_correlations.png', plt)
    plt.show()


def inter_annotator_agreement(output_path):
    plt.figure(figsize=(12, 9))

    scatter_names = ['0x1', '0x2', '0x3', '1x2', '1x3', '2x3']

    i = 1
    for scatter_name in scatter_names:
        plt.subplot(3, 2, i)

        scatter_df = pd.read_csv('olofsson_2018/annotations/annotation_scatters/' + '{}.csv'.format(scatter_name))

        weight = scatter_df['weight'] * 3
        plt.scatter(scatter_df['x'], scatter_df['y'], s=weight, c=weight, cmap='Blues')

        annotator_labels = scatter_name.split('x')
        plt.xlabel('Bedömare {}'.format(annotator_labels[0]))
        plt.ylabel('Bedömare {}'.format(annotator_labels[1]))

        plt.xticks(range(1, 6))
        plt.yticks(range(1, 6))

        plt.grid(True, alpha=0.3)
        i += 1

    plt.subplots_adjust(hspace=0.5)
    save_fig(output_path, 'inter_annotator_agreement.png', plt)


if __name__ == '__main__':
    generate_all('presentation/')