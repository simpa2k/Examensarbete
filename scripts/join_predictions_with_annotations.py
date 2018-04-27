import os
import pandas as pd

def joinWithScoringDirectory(path):
    return os.path.join(scoringDirectory, path)

scoringDirectory = '../output/scoring/'
paths = [
    joinWithScoringDirectory('all_features/'),
    joinWithScoringDirectory('mean_method_level_features/'),
    joinWithScoringDirectory('method_level_loc_project_level_V/'),
    joinWithScoringDirectory('project_level_features/'),
    joinWithScoringDirectory('project_level_loc_method_level_V/')
]

predictionDataFrames = [pd.read_csv(os.path.join(path, 'predictions.csv')) for path in paths]
annotationsByDocument = pd.read_csv(joinWithScoringDirectory('annotations_and_all_features.csv'))