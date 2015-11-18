import csv

import numpy as np


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

    def __init__(self, t_f, t_restart, csv_file_root, shore_file_name):
        self.t_f = t_f
        self.t_restart = t_restart
        self.csv_file_root = csv_file_root
        self.shore_file_name = shore_file_name
        # determine n csv files from t_f, t_restart

        # loop through csv files and load each file
        pass

    def load_numa_csv(self, csv_file_path):
        # initialize a NumaCsvData object
        pass

    def __repr__(self):
        return "NumaRunData({}, {}, {}, {})".format(
            self.t_f,
            self.t_restart,
            self.csv_file_root,
            self.shore_file_name,
        )