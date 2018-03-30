import os
import subprocess
from natsort import natsorted

import numpy as np
import pandas as pd

from src.utils.csv_utils import read_csv, read_csv_from_string, get_csv_reader, save_as_csv
from src.utils.entropy import entropy
from src.utils.remove_nan import remove_nan_from_array
from src.loaders.read_project import read_documents


feature_labels = ['Rader kod', 'Halsteads V', 'Entropi']


def count_lines_of_code(path_to_project):
    loc = subprocess.check_output(['cloc', '--csv', '--quiet', path_to_project])
    matrix = np.matrix(read_csv_from_string(loc.decode('utf-8')[1:],
                                            get_csv_reader(['comment', 'code'])),
                       dtype=np.float64)

    return np.sum(matrix)


def generate_source_meter_data_if_not_exists(path_to_project,
                                             project_name,
                                             name_of_output_directory,
                                             destination_directory):

    if not os.path.exists(os.path.join(destination_directory, name_of_output_directory)):
        subprocess.call(['/home/simon/program/SourceMeter/Java/SourceMeterJava',
                         '-currentDate={}'.format(name_of_output_directory),
                         '-projectName={}'.format(project_name),
                         '-runAndroidHunter=false',
                         '-runMetricHunter=false',
                         '-runVulnerabilityHunter=false',
                         '-runFaultHunter=false',
                         '-runRTEHunter=false',
                         '-runDCF=false',
                         '-resultsDir={}'.format(destination_directory),
                         '-projectBaseDir={}'.format(path_to_project)])


def calculate_halsteads_v(path_to_project):
    project_name = os.path.basename(path_to_project)
    name_of_output_directory = project_name
    destination_directory = '/home/simon/programmering/python/Examensarbete/data/ldo/reports/sourceMeter'

    generate_source_meter_data_if_not_exists(path_to_project,
                                             project_name,
                                             name_of_output_directory,
                                             destination_directory)

    method_metrics = np.matrix(read_csv(os.path.join(destination_directory,
                                                     name_of_output_directory,
                                                     'java',
                                                     project_name,
                                                     '{}-Method.csv'.format(project_name)),
                                        get_csv_reader(['HVOL'])),
                               dtype=np.float64)

    return np.mean(remove_nan_from_array(method_metrics))


def calculate_entropy(path_to_project):
    documents = read_documents(path_to_project, ['.java'], 'rb')
    return entropy(b''.join(documents))


def featurize_project(path_to_project):
    loc = count_lines_of_code(path_to_project)
    average_method_V = calculate_halsteads_v(path_to_project)
    H = calculate_entropy(path_to_project)

    print('Featurized project: {}\n\tLOC: {}\n\tV: {}\n\tH: {}'.format(os.path.basename(path_to_project),
                                                                       loc,
                                                                       average_method_V,
                                                                       H))

    return [loc, average_method_V, H]


def featurize(path_to_projects):
    features = []
    features_path = '../data/ldo/reports/features.csv'

    if not os.path.exists(features_path):
        for root, directories, files in os.walk(path_to_projects):
            for directory in natsorted(directories):
                if not directory.startswith('.'):
                    features.append(featurize_project(os.path.join(root, directory)))

            break

        save_as_csv(features_path, features, '', '%5.10f')
    else:
        features = np.genfromtxt(features_path, delimiter=',')

    return pd.DataFrame(features, index=np.arange(0, 100), columns=feature_labels)
