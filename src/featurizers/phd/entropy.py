import numpy as np


def p(X, xi):
    return xi / np.sum([xj for xj in X])


def H(X):
    """
    Calculates the entropy H of a document X.
    For a full explanation, see:

    Posnett, D., Hindle, A., & Devanbu, P. (2011, May).
    A simpler model of software readability.
    In Proceedings of the 8th working conference on mining software repositories (pp. 73-82). ACM.

    :param X: The document on which to calculate entropy.
    :return: The document's entropy.
    """
    return -1 * np.sum([p(X, xi) * np.log2(p(X, xi)) for xi in X if xi != 0])
