import os
from src.loaders.read_project import read_project


def create_file_loader(path, allowed_extensions):

    """
    Returns a function enclosing the provided path and allowed file extensions,
    able to read from the path at any given time.

    :param path:
    :param allowed_extensions:
    :return:
    """

    def load_data():

        """
        Travels one directory down from the path provided and then reads the
        contents of each subdirectory recursively and gathers their contents.

        :return:
        """
        documents = []
        targets = []

        for root, directories, files in os.walk(path):
            for directory in directories:

                if not directory.startswith('.'):

                    project_path = os.path.join(root, directory)
                    document, target = read_project(project_path + "documents",
                                                    allowed_extensions,
                                                    project_path + "grade/grade.txt")

                    documents.append(document)
                    targets.append(target)

            break

        return documents, targets

    return load_data
