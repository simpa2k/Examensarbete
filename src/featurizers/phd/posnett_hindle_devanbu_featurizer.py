import os
import re
import numpy as np
import scipy as sp
import csv
from natsort import natsorted
from sklearn.feature_extraction.text import CountVectorizer
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


def get_csv_reader(columns):
    def read_columns(dict_reader):
        return [[row[column] for column in columns] for row in dict_reader]

    return read_columns


def get_sorted_csv_reader(column_to_sort_by, *columns_to_read):
    def read_sorted_csv_columns(dict_reader):
        sorted_dict_reader = natsorted(dict_reader, key=lambda d: d[column_to_sort_by])
        return get_csv_reader(columns_to_read)(sorted_dict_reader)

    return read_sorted_csv_columns


def read_halstead():
    return np.matrix(read_csv(data_root + "/Halstead/halstead.csv",
                              get_sorted_csv_reader("File name", "Volume")),
                     dtype=np.float64)  # Not explicitly converting to float64 will cause problems during model training.


def read_lines():
    comments_and_code = np.matrix(read_csv(data_root + "/LOC/loc.csv",
                                           get_sorted_csv_reader("filename", "comment", "code")),
                                  dtype=np.float64)

    return np.sum(comments_and_code, axis=1)


def entropy(labels):
    prob_dict = {x: labels.count(x)/len(labels) for x in labels}
    probs = np.array(list(prob_dict.values()))

    return - probs.dot(np.log2(probs))


def featurize(documents):

    # documents = tokenize(documents)
    # dictionary = create_dictionary(documents)

    # vectors = vectorize(documents, dictionary)
    # vectors = [[entry[1] for entry in vector] for vector in vectors]  # Select token count

    # entropy = np.array([[H(x)] for x in vectors])
    # H = np.array([[shannon_entropy(document)] for document in documents])
    H = np.array([[entropy(document)] for document in documents])
    halstead = np.around(read_halstead())
    lines = read_lines()

    return np.array(np.concatenate((halstead, lines, H), axis=1))
