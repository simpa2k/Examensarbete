from unittest import TestCase
from src.loaders.phd.buse_weimer_loader import create_file_loader


class TestLoadData(TestCase):

    def test_loads_annotations_as_numpy_array(self):
        root_path = "../../../data/bw"
        load_data = create_file_loader(root_path + "/snippets", root_path + "/votes.csv")
        documents, votes = load_data()

        self.assertEqual(100, len(documents))
        self.assertEqual((121, 100), votes.shape)
