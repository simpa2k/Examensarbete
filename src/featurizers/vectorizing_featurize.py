import numpy as np
from gensim import matutils
from gensim import corpora


def clean(data):
    return [dataPoint.split() for dataPoint in data]


def create_dictionary(data):
    return corpora.Dictionary(data)


def vectorize(data, dictionary):
    return [dictionary.doc2bow(dataPoint) for dataPoint in data]


def create_corpus(path, vectors):
    corpora.MmCorpus.serialize(path, vectors)
    return corpora.MmCorpus(path)


def to_numpy_mat(corpus):
    matrix = matutils.corpus2dense(corpus, num_terms=corpus.num_terms)
    return np.rot90(matrix)


# Vectorizes the documents retrieved by the function provided and outputs the
# created vector space to the path passed. Returns the read data as a numpy
# matrix along with the annotation of each data point.
def featurize(load_data, path_to_corpus):

    documents, targets = load_data()
    documents = clean(documents)

    dictionary = create_dictionary(documents)

    vecs = vectorize(documents, dictionary)
    corpus = create_corpus(path_to_corpus, vecs)
    print(corpus)

    return {'data': to_numpy_mat(corpus),
            'target': np.fromiter(targets, np.float)}

