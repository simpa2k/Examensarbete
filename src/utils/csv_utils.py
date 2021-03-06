import csv
import io
import os
import numpy as np
import pandas as pd
from natsort import natsorted


def read_csv(path, read_function):
    with open(path, "r") as f:
        return read_function(csv.DictReader(f, delimiter=","))


def read_csv_from_string(string, read_function):
    return read_function(csv.DictReader(io.StringIO(string), delimiter=','))


def get_csv_reader(columns):
    def read_columns(dict_reader):
        return [[row[column] for column in columns] for row in dict_reader]

    return read_columns


def get_sorted_csv_reader(column_to_sort_by, *columns_to_read):
    def read_sorted_csv_columns(dict_reader):
        sorted_dict_reader = natsorted(dict_reader, key=lambda d: d[column_to_sort_by])
        return get_csv_reader(columns_to_read)(sorted_dict_reader)

    return read_sorted_csv_columns


def save_as_csv(output_file, data, header, fmt):
    np.savetxt(output_file,
               data,
               header=header,
               comments='',
               delimiter=',',
               fmt=fmt)


def join_csv_files(output_file, input_files, column_to_join_on, column_names, skiprows=None):
    joined = None

    i = 0
    for file in input_files:
        if skiprows is not None:
            df = pd.read_csv(file, skiprows=skiprows)
        else:
            df = pd.read_csv(file)

        df = df.set_index(column_to_join_on)
        df.columns = column_names(i)

        if joined is not None:
            joined = joined.join(df)
        else:
            joined = df

        i = i + 1

    joined.to_csv(output_file)


if __name__ == '__main__':
    root = 'data/ldo/annotations/'

    def with_root(path):
        return os.path.join(root, path)

    join_csv_files(with_root('annotations.csv'),
                   [with_root('simon_changed.csv'), with_root('maja_changed.csv'), with_root('robert_changed.csv'), with_root('6.csv')],
                   'Uppgift',
                   lambda i: ['Bedömning' + str(i)])
