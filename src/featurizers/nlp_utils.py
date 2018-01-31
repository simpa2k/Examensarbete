import numpy as np
from gensim import matutils
from gensim import corpora


def create_dictionary(data):
    return corpora.Dictionary(data)


def vectorize(data, dictionary):
    return [dictionary.doc2bow(dataPoint) for dataPoint in data]


def to_mm_corpus(path, vectors):
    corpora.MmCorpus.serialize(path, vectors)
    return corpora.MmCorpus(path)


def to_numpy_mat(corpus):
    matrix = matutils.corpus2dense(corpus, num_terms=corpus.num_terms)
    return np.rot90(matrix)
