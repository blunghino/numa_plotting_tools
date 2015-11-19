import csv
import os
import pickle
import warnings

import numpy as np
from matplotlib import pyplot as plt


def saveobj(obj, path):
    """
    pickle an object.
    """
    ## Add correct file extension to path passed with function call
    if path[-4:] != '.pkl':
        path += '.pkl'
    try:
        ## Open file and dump dictionary
        with open(path, 'wb') as output:
            pickle.dump(obj, output)
            output.close()
        return path
    except FileNotFoundError:
        raise

def openobj(path):
    """
    unpickle an object.
    """
    ## Add correct file extension to path passed with function call
    if path[-4:] != '.pkl':
        path += '.pkl'
    ## open file and load dictionary
    with open(path, 'rb') as picklein:
        obj = pickle.load(picklein)
        picklein.close()
    return obj

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

    def __init__(self, csv_file_path, xcoord_header='xcoord',
                 ycoord_header='ycoord'):
        ## load data
        self.csv_file_path = csv_file_path
        self.headers = self._get_headers(csv_file_path)
        self.data = self._load_csv_data(csv_file_path)
        for j, att in enumerate(self.headers):
            setattr(self, att, self.data[:,j])
        ## reshap inferring shape from data
        if xcoord_header in self.headers and ycoord_header in self.headers:
            self.headers.remove(xcoord_header)
            self.headers.remove(ycoord_header)
            x = getattr(self, xcoord_header)
            self.nelx = 1 + find_first_repeated(x, first_not=False)
            self.nely = len(x) / self.nelx
            self.y = getattr(self, ycoord_header).reshape((self.nely,self.nelx))
            self.x = x.reshape((self.nely,self.nelx))
            for att in self.headers:
                temp = getattr(self, att).reshape((self.nely,self.nelx))
                setattr(self, att, temp)
        else:
            ## can't detect x and/or y coord data
            raise warnings.warning(
                'NO X and/or Y data found using headers {} and {} \n{}'.format(
                    xcoord_header,
                    ycoord_header,
                    csv_file_path
                )
            )

    def _get_headers(self, csv_file_path):
        """
        read the first row of file `csv_file_path` and return a list of the
        headers in that row
        """
        with open(csv_file_path, 'r') as file:
            rdr = csv.reader(file)
            #TODO: check that header can be set as an attribute?
            r0 = [x.lstrip() for x in next(rdr)]
        return r0

    def _load_csv_data(self, csv_file_path):
        """
        load csv data using numpy
        """
        return np.loadtxt(csv_file_path, skiprows=1)

    def plot_velocity(self, figsize=(12,7), uvelo='uvelo', vvelo='vvelo'):
        """
        quiver plot of velocities
        """
        U = getattr(self, uvelo)
        V = getattr(self, vvelo)
        fig = plt.figure(figsize=figsize)
        plt.quiver(self.x, self.y, U, V)
        return fig

    def plot_height(self, figsize=(12,7), height='height'):
        """
        pcolor plot of height
        """
        H = getattr(self, height)
        fig = plt.figure(figsize=figsize)
        plt.pcolor(self.x, self.y, H)
        return fig

    def plot_surface(self, figsize=(12,7), height='height'):
        H = getattr(self, height)
        fig = plt.figure(figsize=figsize)
        
        return fig
    def __repr__(self):
        return "NumaCsvData({})".format(self.csv_file_path)

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