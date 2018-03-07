import sys
import numpy as np

data = np.genfromtxt(sys.argv[1], delimiter=',')
without_annotators = data[:, 2:]
rotated = np.rot90(without_annotators, axes=(1, 0))
id_column = np.reshape(np.arange(1, rotated.shape[0] + 1), newshape=(100, 1))
with_id = np.concatenate((id_column, rotated), axis=1)