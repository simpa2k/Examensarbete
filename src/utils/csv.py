import csv
import io
import numpy as np
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


def save_as_csv(output_file, data, header):
    np.savetxt(output_file,
               data,
               header=header,
               comments='',
               delimiter=',')
