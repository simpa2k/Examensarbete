import re
import numpy as np
from src.featurizers.nlp_utils import create_dictionary, vectorize
from src.featurizers.phd.entropy import H


def tokenize(documents):

    hex_documents = [document.hex() for document in documents]
    # From second answer at https://stackoverflow.com/questions/9475241/split-string-every-nth-character
    return [re.findall('..', hex_representation) for hex_representation in hex_documents]


def featurize(documents):

    documents = tokenize(documents)
    dictionary = create_dictionary(documents)

    vectors = vectorize(documents, dictionary)
    vectors = [[entry[1] for entry in vector] for vector in vectors]  # Select token count

    return np.array([[H(x)] for x in vectors])
