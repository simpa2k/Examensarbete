import numpy as np
import math
from src.loaders.read_project import read_documents


def remove_nan(matrix):
    """
    Removes all NaN values from a numpy matrix.
    From https://stackoverflow.com/questions/25026486/removing-nan-elements-from-matrix

    :param matrix: The numpy matrix to remove all NaN values from
    :return: A numpy matrix with all NaN values removed
    """
    matrix = matrix[:, ~np.isnan(matrix).all(0)]
    matrix = matrix[~np.isnan(matrix).all(1)]

    return matrix


def create_file_loader(path_to_snippets, path_to_votes):

    def load_data():
        snippets = read_documents(path_to_snippets, [".jsnp"], "rb")  # Read as bytes
        votes = np.genfromtxt(path_to_votes, delimiter=",")
        votes = remove_nan(votes)

        return snippets, votes

    return load_data
