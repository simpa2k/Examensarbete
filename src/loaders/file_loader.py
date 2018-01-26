import os
from random import randrange


def read_project(root_path, documents_dir, allowed_extensions, target_dir):

    """
    Reads all files with an extension in the allowed_extensions list recursively
    from the path provided and joins the files' contents with newline characters.
    Also annotates the path.

    :param root_path:
    :param documents_dir:
    :param allowed_extensions:
    :param target_dir:
    :return:
    """
    documents = []

    for root, directories, files in os.walk(os.path.join(root_path, documents_dir)):
        for file in files:
            with open(os.path.join(root, file), "r") as f:
                filename, file_extension = os.path.splitext(file)
                if file_extension in allowed_extensions:
                    documents.append(f.read())

    with open(os.path.join(root_path, target_dir), "r") as f:
        target = f.read()

    return '\n'.join(documents), target


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
                    document, target = read_project(os.path.join(root, directory),
                                                    "documents",
                                                    allowed_extensions,
                                                    "grade/grade.txt")

                    documents.append(document)
                    targets.append(target)

            break

        return documents, targets

    return load_data
