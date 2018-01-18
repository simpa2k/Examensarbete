import os
import numpy as np
from gensim import matutils
from gensim import corpora
from gensim import models
from pprint import pprint
from random import randrange

ALLOWED_EXTENSIONS = [".cpp", ".h"]


def load_data(path):

    documents = []

    for root, dir, files in os.walk(path):
        for file in files:
            with open(os.path.join(root, file), "r") as f:
                filename, file_extension = os.path.splitext(file)
                if file_extension in ALLOWED_EXTENSIONS:
                    documents.append(f.read())

    return documents


def clean(data):
    return [datapoint.split() for datapoint in data]


def create_dictionary(data):
    return corpora.Dictionary(data)


def vectorize(data, dictionary):
    return [dictionary.doc2bow(datapoint) for datapoint in data]


def create_corpus(path, vectors):
    corpora.MmCorpus.serialize(path, vectors)
    return corpora.MmCorpus(path)


def to_numpy_mat(corpus):
    return matutils.corpus2dense(corpus, num_terms=corpus.num_terms)


def annotate(corpus_mat):

    rows = corpus_mat.shape[0]

    grades = np.empty(rows)

    for i in range(0, rows):
        grades[i] = randrange(0, 6)

    return {'data': corpus_mat, 'target': grades}


def featurize(path_to_data, path_to_corpus):

    documents = load_data(path_to_data)
    documents = clean(documents)

    dictionary = create_dictionary(documents)

    vecs = vectorize(documents, dictionary)
    corpus = create_corpus(path_to_corpus, vecs)
    print(corpus)

    # return {'data': to_numpy_mat(corpus),
    #        'target': np.array([6])}

    return annotate(to_numpy_mat(corpus))
