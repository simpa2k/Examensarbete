import os
import re
import numpy as np
import csv
from src.featurizers.nlp_utils import create_dictionary, vectorize
from src.featurizers.phd.entropy import H
from src.loaders.phd.buse_weimer_loader import remove_nan


data_root = "../../data/bw/reports"


def read_csv(path, read_function):
    with open(path, "r") as f:
        return read_function(csv.DictReader(f, delimiter=","))


def tokenize(documents):

    hex_documents = [document.hex() for document in documents]
    # From second answer at https://stackoverflow.com/questions/9475241/split-string-every-nth-character
    return [re.findall('..', hex_representation) for hex_representation in hex_documents]


def get_single_column_csv_reader(column):
    def read_single_csv_column(dict_reader):
        return [row[column] for row in dict_reader]

    return read_single_csv_column


def read_halstead():
    return np.rot90(np.matrix(read_csv(data_root + "/Halstead/halstead.csv", get_single_column_csv_reader("Volume")), dtype=np.float64))


def read_lines():
    return np.rot90(np.matrix(read_csv(data_root + "/LOC/loc.csv", get_single_column_csv_reader("code")), dtype=np.float64))


def featurize(documents):

    documents = tokenize(documents)
    dictionary = create_dictionary(documents)

    vectors = vectorize(documents, dictionary)
    vectors = [[entry[1] for entry in vector] for vector in vectors]  # Select token count

    entropy = np.array([[H(x)] for x in vectors])
    halstead = read_halstead()
    lines = read_lines()

    return np.concatenate((halstead, lines, entropy), axis=1)

