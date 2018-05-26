import argparse

from src.experiments.dighem_lund_2018 import dighem_lund_2018
from src.experiments.olofsson_2018 import olofsson_2018

experiment_labels = ['olofsson_2018', 'dighem_lund_2018']
experiments = [olofsson_2018, dighem_lund_2018]
experiments_by_label = dict(zip(experiment_labels, experiments))


def main():
    parser = setup_command_line_parser()
    args = parser.parse_args()

    experiment = experiments_by_label[args.experiment]

    experiment()


def setup_command_line_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment', help='The experiment to run',
                        choices=experiment_labels, required=True)

    return parser


if __name__ == '__main__':
    main()