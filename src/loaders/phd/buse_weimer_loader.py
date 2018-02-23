import numpy as np
from src.loaders.read_project import read_documents
from src.utils.remove_nan import remove_nan_from_matrix


def create_file_loader(path_to_snippets, path_to_votes):

    def load_data():
        snippets = read_documents(path_to_snippets, [".jsnp"], "rb")  # Read as bytes
        votes = np.genfromtxt(path_to_votes, delimiter=",")
        votes = remove_nan_from_matrix(votes)

        return snippets, votes

    return load_data
