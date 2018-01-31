import numpy as np
from src.featurizers.nlp_utils import create_dictionary, vectorize, to_mm_corpus, to_numpy_mat


def tokenize(data):
    return [dataPoint.split() for dataPoint in data]


def featurize(load_data, path_to_corpus):

    """
    Vectorizes the documents retrieved by the function provided as a bag of words
    and outputs the created vector space to the path passed. Returns the read data
    as a numpy matrix along with the annotation of each data point.

    :param load_data:
    :param path_to_corpus:
    :return:
    """
    documents, targets = load_data()
    documents = tokenize(documents)

    dictionary = create_dictionary(documents)

    vecs = vectorize(documents, dictionary)
    mm_corpus = to_mm_corpus(path_to_corpus, vecs)
    print(mm_corpus)

    return {'data': to_numpy_mat(mm_corpus),
            'target': np.fromiter(targets, np.float)}

