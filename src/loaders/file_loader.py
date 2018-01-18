import os
from random import randrange


# Reads all files with an extension in the allowed_extensions list recursively
# from the path provided and joins the files' contents with newline characters.
# Also annotates the path.
def read_project(path, allowed_extensions):

    documents = []

    for root, directories, files in os.walk(path):
        for file in files:
            with open(os.path.join(root, file), "r") as f:
                filename, file_extension = os.path.splitext(file)
                if file_extension in allowed_extensions:
                    documents.append(f.read())

    return '\n'.join(documents), randrange(0, 6)


# Returns a function enclosing the provided path and allowed file extensions,
# able to read from the path at any given time.
def create_file_loader(path, allowed_extensions):

    # Travels one directory down from the path provided and then reads the
    # contents of each subdirectory recursively and gathers their contents.
    def load_data():

        documents = []
        targets = []

        for root, directories, files in os.walk(path):
            for directory in directories:

                if not directory.startswith('.'):
                    document, target = read_project(os.path.join(root, directory), allowed_extensions)

                    documents.append(document)
                    targets.append(target)

            break

        return documents, targets

    return load_data
