from unittest import TestCase
from src.featurizers.phd.entropy import H


class TestEntropy(TestCase):

    def test_H(self):
        print(H([1, 3, 5, 0]))
