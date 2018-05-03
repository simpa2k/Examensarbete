import os
import numpy as np

matrices = []
for root, directories, files in os.walk('output/scoring/all_combinations/'):
    for directory in directories:
        if not directory.startswith('.'):
            matrices.append(
                np.genfromtxt(os.path.join(root, directory, 'confusion_matrix.csv'), delimiter=',').astype(int)
            )
