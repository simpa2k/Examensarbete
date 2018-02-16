import numpy as np


def entropy(labels):
    """
    From https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
    :param labels:
    :return:
    """
    prob_dict = {x: labels.count(x)/len(labels) for x in labels}
    probs = np.array(list(prob_dict.values()))

    return - probs.dot(np.log2(probs))
