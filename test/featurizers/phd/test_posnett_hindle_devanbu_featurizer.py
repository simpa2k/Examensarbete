from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from src.featurizers.phd.posnett_hindle_devanbu_featurizer import featurize
from src.loaders.phd.buse_weimer_loader import create_file_loader


class TestPosnettHindleDevanbuFeaturizer(TestCase):

    def test_featurize(self):

        data_root = "../../../data/bw"
        load_data = create_file_loader(data_root + "/snippets", data_root + "/votes.csv")
        documents, votes = load_data()

        features = featurize(documents)

