import numpy as np
from gensim import matutils
from gensim import corpora


def tokenize(data):
    return [dataPoint.split() for dataPoint in data]


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

