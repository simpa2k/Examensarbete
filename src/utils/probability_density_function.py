import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def probability_density_function_from_samples(samples,
                                              linspace_start, linspace_stop,
                                              covariance_factor,
                                              output_path,
                                              x_axis_label='Samples',
                                              y_axis_label='Probability density values'):
    """
    From https://stackoverflow.com/questions/4150171/how-to-create-a-density-plot-in-matplotlib
    """
    density = gaussian_kde(samples)
    xs = np.linspace(linspace_start, linspace_stop, 200)
    density.covariance_factor = lambda: covariance_factor
    density._compute_covariance()
    plt.plot(xs, density(xs))

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    plt.savefig(output_path)
