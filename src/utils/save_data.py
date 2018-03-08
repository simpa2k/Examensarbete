import os
import numpy as np


def save_as_csv(output_path, filename, data, header):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    np.savetxt(os.path.join(output_path, filename),
               data,
               header=header,
               comments='',
               delimiter=',',
               fmt='%5.2f')


def save_fig(output_path, filename, plt):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plt.savefig(os.path.join(output_path, filename))
