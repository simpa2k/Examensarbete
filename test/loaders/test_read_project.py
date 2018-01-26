from unittest import TestCase
from src.loaders.file_loader import read_project


class TestReadProject(TestCase):

    data_root = "../resources/data"

    proj1_root = data_root + "/proj1"
    proj2_root = data_root + "/proj2"

    relative_grade_path = "grade/grade.txt"

    def test_read_project(self):

        documents, target = read_project(self.proj1_root, "documents", [".txt"], self.relative_grade_path)

        self.assertEqual("This is good code!", documents)
        self.assertEqual("A", target)
