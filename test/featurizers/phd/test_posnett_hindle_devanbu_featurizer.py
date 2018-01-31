from unittest import TestCase
import numpy as np
from src.featurizers.phd.posnett_hindle_devanbu_featurizer import featurize
from src.loaders.phd.buse_weimer_loader import create_file_loader


class TestPosnettHindleDevanbuFeaturizer(TestCase):

    def test_featurize(self):

        data_root = "../../../data/bw"
        load_data = create_file_loader(data_root + "/snippets", data_root + "/votes.csv")
        documents, votes = load_data()

        features = featurize(documents)
        votes = np.rot90(votes)
        votes = np.apply_along_axis(lambda votes_for_document: [np.sum(votes_for_document) / len(votes_for_document)],
                                    1,
                                    votes)

        self.assertEqual((100, 1), votes.shape)
