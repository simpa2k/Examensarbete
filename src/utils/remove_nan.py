import numpy as np


def remove_nan_from_matrix(matrix):
    """
    Removes all NaN values from a numpy matrix.
    From https://stackoverflow.com/questions/25026486/removing-nan-elements-from-matrix

    :param matrix: The numpy matrix to remove all NaN values from
    :return: A numpy matrix with all NaN values removed
    """
    matrix = matrix[:, ~np.isnan(matrix).all(0)]
    matrix = matrix[~np.isnan(matrix).all(1)]

    return matrix


def remove_nan_from_array(array):
    return array[~np.isnan(array)]
