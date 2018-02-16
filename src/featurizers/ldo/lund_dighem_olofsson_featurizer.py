import os
import subprocess
from natsort import natsorted

import numpy as np

from src.utils.csv import read_csv, read_csv_from_string, get_csv_reader
from src.utils.entropy import entropy
from src.loaders.read_project import read_documents


def count_lines_of_code(path_to_project):
    loc = subprocess.check_output(['cloc', '--csv', '--quiet', path_to_project])
    matrix = np.matrix(read_csv_from_string(loc.decode('utf-8')[1:],
                                            get_csv_reader(['comment', 'code'])),
                       dtype=np.float64)

    return np.sum(matrix)


def calculate_halsteads_v(path_to_project):
    project_name = 'temp'
    name_of_output_directory = 'temp'
    destination_directory = '/home/simon/programmering/python/Examensarbete/data/ldo/reports/sourceMeter'

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

    method_metrics = np.matrix(read_csv(os.path.join(destination_directory,
                                                     name_of_output_directory,
                                                     'java',
                                                     project_name,
                                                     '{}-Method.csv'.format(project_name)),
                                        get_csv_reader(['HVOL'])),
                               dtype=np.float64)

    return np.mean(method_metrics)


def calculate_entropy(path_to_project):
    documents = read_documents(path_to_project, ['.java'], 'rb')
    return entropy(b''.join(documents))


def featurize_project(path_to_project):
    loc = count_lines_of_code(path_to_project)
    average_method_V = calculate_halsteads_v(path_to_project)
    H = calculate_entropy(path_to_project)

    return [loc, average_method_V, H]


def featurize(path_to_projects):
    features = []

    for root, directories, files in os.walk(path_to_projects):
        for directory in natsorted(directories):
            if not directory.startswith('.'):
                features.append(featurize_project(os.path.join(root, directory)))

    return features
