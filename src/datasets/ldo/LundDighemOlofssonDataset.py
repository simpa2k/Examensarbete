import os

from src.datasets.ldo.FeatureSet import FeatureSet
from src.datasets.ldo.AnnotationSet import AnnotationSet


class LundDighemOlofssonDataset():

    def __init__(self):
        self.feature_set = FeatureSet()
        self.annotation_set = AnnotationSet()

    def load(self, path_to_projects, path_to_annotations):
        self.feature_set.load(path_to_projects)
        self.annotation_set.load_annotations(path_to_annotations)

    def get_features(self):
        return self.feature_set.features

    def get_annotations(self):
        return self.annotation_set.annotations

    def describe(self, output_path):
        self.feature_set.describe_features(os.path.join(output_path, 'features'))
        self.annotation_set.describe_annotations(os.path.join(output_path, 'annotations'))

