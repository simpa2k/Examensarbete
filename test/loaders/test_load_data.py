from unittest import TestCase
from src.loaders.file_loader import create_file_loader


class TestLoadData(TestCase):

    data_root = "../resources/data"
    allowed_extensions = [".txt"]

    def test_load_data(self):
        load_data = create_file_loader(self.data_root, self.allowed_extensions)
        documents, grades = load_data()

        self.assertEqual(2, len(documents))
        self.assertEqual(2, len(grades))

        self.assertEqual("This is good code!", documents[0])
        self.assertEqual("This is bad code.", documents[1])

        self.assertEqual("A", grades[0])
        self.assertEqual("F", grades[1])
