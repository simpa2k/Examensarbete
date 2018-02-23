import os
import re
import numpy as np
import scipy as sp
import csv
from natsort import natsorted
from sklearn.feature_extraction.text import CountVectorizer
from src.featurizers.nlp_utils import create_dictionary, vectorize
from src.featurizers.phd.entropy import H
from src.loaders.phd.buse_weimer_loader import remove_nan_from_matrix
from src.utils.csv_utils import read_csv, get_sorted_csv_reader


# data_root = "../../data/bw/reports"


def tokenize(documents):

    hex_documents = [document.hex() for document in documents]
    # From second answer at https://stackoverflow.com/questions/9475241/split-string-every-nth-character
    return [re.findall('..', hex_representation) for hex_representation in hex_documents]


def read_halstead(data_root):
    return np.matrix(read_csv(data_root + "/Halstead/halstead.csv",
                              get_sorted_csv_reader("File name", "Volume")),
                     dtype=np.float64)  # Not explicitly converting to float64 will cause problems during model training.


def read_lines(data_root):
    comments_and_code = np.matrix(read_csv(data_root + "/LOC/loc.csv",
                                           get_sorted_csv_reader("filename", "comment", "code")),
                                  dtype=np.float64)

    return np.sum(comments_and_code, axis=1)




def featurize(documents, data_root):

    # documents = tokenize(documents)
    # dictionary = create_dictionary(documents)

    # vectors = vectorize(documents, dictionary)
    # vectors = [[entry[1] for entry in vector] for vector in vectors]  # Select token count

    # entropy = np.array([[H(x)] for x in vectors])
    # H = np.array([[shannon_entropy(document)] for document in documents])
    H = np.array([[entropy(document)] for document in documents])
    halstead = np.around(read_halstead(data_root), decimals=2)
    lines = read_lines(data_root)

    return np.array(np.concatenate((halstead, lines, H), axis=1))
