import os


def read_annotations(annotations_dir):

    with open(annotations_dir, "r") as f:
        return f.read()


def read_documents(documents_dir, allowed_extensions, read_mode="r"):

    documents = []

    for root, directories, files in os.walk(documents_dir):
        for file in files:
            with open(os.path.join(root, file), read_mode) as f:
                filename, file_extension = os.path.splitext(file)
                if file_extension in allowed_extensions:
                    documents.append(f.read())

    return documents


def read_project(documents_dir, allowed_extensions, target_dir):

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

    documents = read_documents(documents_dir, allowed_extensions)
    target = read_annotations(target_dir)

    return '\n'.join(documents), target
