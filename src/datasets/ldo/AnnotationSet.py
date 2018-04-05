import os
import collections

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from itertools import permutations, combinations
from src.utils.save_data import save_as_csv, save_fig
from src.datasets.ldo.inter_annotator_agreement import calculate_inter_annotator_agreement
from src.utils.probability_density_function import probability_density_function_from_samples


class AnnotationSet:
    def __init__(self):
        self.unprocessed_annotations = None
        self.averaged_annotations = None
        self.no_neutral_annotations = None
        self.binarized_annotations = None

        self.annotation_column = 'Label'

    def load_annotations(self, path_to_annotations):
        self.unprocessed_annotations = np.genfromtxt(path_to_annotations, delimiter=',')[1:, 1:4]
        self.averaged_annotations = self.unprocessed_annotations.mean(axis=1)

        averaged_as_dataset = pd.DataFrame(self.averaged_annotations, columns=[self.annotation_column])
        self.no_neutral_annotations = averaged_as_dataset[averaged_as_dataset.Label != 3]

        self.binarized_annotations = pd.DataFrame((self.no_neutral_annotations[self.annotation_column] > 3).astype(int))

    def describe_annotations(self, output_path):
        self.output_annotation_csv(output_path)
        self.output_normal_test(output_path)
        self.output_inter_annotator_agreement(output_path)
        self.output_annotation_plots(output_path)

    def output_annotation_csv(self, output_path):
        self.describe_averaged_annotations(output_path)
        self.describe_annotation_binarization(output_path)

    def describe_averaged_annotations(self, output_path):
        nobs, minmax, mean, variance, skewness, kurtosis = stats.describe(self.averaged_annotations)
        save_as_csv(
            output_path,
            'annotations_description.csv',
            np.array([nobs, minmax[0], minmax[1], mean, variance, skewness, kurtosis])[np.newaxis],
            'Antal observationer,Minimum,Maximum,Medelvärde,Varians,Skevhet,Kurtosis'
        )

    def describe_annotation_binarization(self, output_path):
        class_counts, bin_edges = np.histogram(self.no_neutral_annotations, bins=[0, 3, 5])
        sum_class_counts = np.sum(class_counts)

        neutral_count = self.averaged_annotations.shape[0] - sum_class_counts

        column_labels = ['Mindre läsbara', 'Neutrala', 'Mer läsbara', 'Mindre läsbara + mer läsbara', 'Totalt']
        df = pd.DataFrame([
            np.append(
                np.insert(class_counts, 1, neutral_count),
                np.array([sum_class_counts, sum_class_counts + neutral_count])
            )], columns=column_labels)

        df.to_csv(os.path.join(output_path, 'binarized_description.csv'), index=False)
        self.binarized_annotations.to_csv(os.path.join(output_path, 'binarized_human_annotations.csv'))

    def output_normal_test(self, output_path):
        k2, p = stats.normaltest(self.averaged_annotations)
        save_as_csv(
            output_path,
            'annotations_normal_test.csv',
            np.array([k2, p])[np.newaxis],
            'k$^2$,p (Tvåsidigt)'
        )

    def output_inter_annotator_agreement(self, output_path):
        #correlations = calculate_inter_annotator_agreement(self.unprocessed_annotations)
        #correlations.to_csv(os.path.join(output_path, 'inter_annotator_agreement.csv'))

        plt.subplots_adjust(hspace=0.4, wspace=0.8)

        i = 1
        for annotator_one, annotator_two in combinations([i for i in range(self.unprocessed_annotations.shape[1])], 2):
            plt.subplot(3, 3, i)
            plt.scatter(
                self.unprocessed_annotations[0:, annotator_one],
                self.unprocessed_annotations[0:, annotator_two],
                color=['red', 'green']
            )

            plt.xlabel(annotator_one)
            plt.ylabel(annotator_two)

            df = pd.DataFrame(columns=['x', 'y', 'weight'])
            unique_combinations = collections.Counter(zip(
                self.unprocessed_annotations[0:, annotator_one],
                self.unprocessed_annotations[0:, annotator_two]
            ))

            if not os.path.exists(os.path.join(output_path, 'annotation_scatters')):
                os.makedirs(os.path.join(output_path, 'annotation_scatters'))

            df = pd.concat([pd.DataFrame([[xy[0], xy[1], weight]], columns=['x', 'y', 'weight']) for xy, weight in unique_combinations.items()], ignore_index=True)
            df.to_csv(os.path.join(output_path, 'annotation_scatters', '{}x{}.csv'.format(annotator_one, annotator_two)))

            i += 1

        save_fig(output_path, 'annotator_scatter_plots.png', plt)
        plt.clf()

    def output_annotation_plots(self, output_path):
        plt.hist(self.averaged_annotations, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])

        plt.xlabel('Läsbarhetsbetyg')
        plt.ylabel('Antal')

        save_fig(output_path, 'annotations_histogram.png', plt)
        plt.clf()

        probability_density_function_from_samples(self.averaged_annotations,
                                                  1.5, 5,
                                                  .22,
                                                  os.path.join(output_path, 'annotations_pdf.png'),
                                                  x_axis_label='Genomsnittligt läsbarhetsbetyg',
                                                  y_axis_label='Kodavsnittets täthetsvärde')
