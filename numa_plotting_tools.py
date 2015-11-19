import csv
import os

import numpy as np


def get_run_dirs():
    """
    returns list of all dir names in curdir`
    """
    return [x[0] for x in os.walk(os.curdir) if x[0] != '.']


def find_first_repeated(x, first_not=False):
    """
    return the index of the first reappearence of the first element in array x
    if first_not instead return the index of the first appearance of a different
    value from the first element in array x
    """
    val = x[0]
    for i, z in enumerate(x[1:]):
        if not first_not and z == val:
            return i
        elif first_not and z != val:
            return i

class NumaCsvData:
    """
    class to hold data stored in a single Numa output csv file
    """

    def __init__(self, csv_file_name, xcoord_header='xcoord',
                 ycoord_header='ycoord'):
        ## load data
        self.csv_file_name = csv_file_name
        self.headers = self._get_headers(csv_file_name)
        self.data = self._load_csv_data(csv_file_name)
        for j, att in enumerate(self.headers):
            setattr(self, att, self.data[:,j])
        ## reshap inferring shape from data
        if xcoord_header in self.headers and ycoord_header in self.headers:
            x = getattr(self, xcoord_header)
            self.nelx = 1 + find_first_repeated(x, first_not=False)
            self.nely = len(x) / self.nelx
            for att in self.headers:
                temp = getattr(self, att).reshape((self.nely, self.nelx))
                setattr(self, att, temp)

    def _get_headers(self, csv_file_name):
        """
        read the first row of file `csv_file_name` and return a list of the
        headers in that row
        """
        with open(csv_file_name, 'r') as file:
            rdr = csv.reader(file)
            #TODO: check that header can be set as an attribute?
            r0 = [x.lstrip() for x in next(rdr)]
        return r0

    def _load_csv_data(self, csv_file_name):
        """
        load csv data using numpy
        """
        return np.loadtxt(csv_file_name, skiprows=1)

    def __repr__(self):
        return "NumaCsvData({})".format(self.csv_file_name)

class NumaRunData:
    """
    class to hold all data associated with a Numa model run
    """

    def __init__(
            self,
            t_f,
            t_restart,
            run_dir_path,
            shore_file_name='OUT_SHORE_data.dat'
    ):
        self.t_f = t_f
        self.t_restart = t_restart
        self.run_dir_path = run_dir_path
        self.shore_file_name = shore_file_name
        csv_file_root = os.path.split(run_dir_path)[1]
        n_outputs = int(t_f / t_restart)
        csv_file_names = [
            '{}_{:03d}.csv'.format(csv_file_root, i) for i in range(n_outputs)
        ]
        self.data_obj_list = self._load_numa_csv_data(csv_file_names)

    def _load_numa_csv_data(self, csv_file_names):
        data_obj_list = []
        for csv_file_name in csv_file_names:
            csv_file_path = os.path.join(self.run_dir_path, csv_file_name)
            data_obj_list.append(NumaCsvData(csv_file_path))
        return data_obj_list

    def __repr__(self):
        return "NumaRunData({}, {}, {})".format(
            self.t_f,
            self.t_restart,
            self.run_dir_path
        )