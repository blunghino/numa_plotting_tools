import csv
import os
import pickle
import warnings

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import matplotlib as mpl


def saveobj(obj, path):
    """
    pickle an object.
    """
    ## Add correct file extension to path passed with function call
    if path[-4:] != '.pkl':
        path += '.pkl'
    try:
        ## Open file and dump object
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
    ## open file and load object
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

def plot_shore_max_timeseries(nrds, figsize=(12,9), colormap='viridis'):
    """
    call the NumaRunData.plot_shore_max method for a list of NumaRunData objects
    plot all lines on one matplotlib.figure.Figure instance
    """
    ## set up color map and markers
    n = len(nrds)
    cmap_vals = np.linspace(0, 255, n, dtype=int)
    cmap = plt.get_cmap(colormap)
    color = [cmap(val) for val in cmap_vals]
    marker = 'sx^voH'
    marker *= int(np.ceil(n/len(marker)))
    ## initialize figure instance
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    for i, nrd in enumerate(nrds):
        fig = nrd.plot_shore_max_timeseries(
            figure_instance=fig,
            color=color[i],
            marker=marker[i]
        )
    ax.legend(loc=2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Max shore position [m]')
    ax.set_xlim(right=nrd.t_f)
    return fig


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

    def plot_velocity_streamlines(self, figsize=(12,7), return_fig=True,
                                  uvelo='uvelo', vvelo='vvelo'):
        """
        streamline plot of velocity
        """
        U = getattr(self, uvelo)
        V = getattr(self, vvelo)
        if return_fig:
            fig = plt.figure(figsize=figsize)
            p = plt.streamplot(self.x, self.y, U, V)
            return fig
        else:
            p = plt.streamplot(self.x, self.y, U, V)
            return p

    def plot_height_3D(self, figsize=(12,7), return_fig=True,
                       height='height', cmap='jet'):
        H = getattr(self, height)
        colormap = plt.get_cmap(cmap)
        if return_fig:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            p = ax.plot_surface(self.x, self.y, H, cmap=colormap)
            return fig
        else:
            p = plt.plot_surface(self.x, self.y, H, cmap=colormap)
            return p


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
            shore_file_name='OUT_SHORE_data.dat',
            load_csv_data=True,
            name=None,
    ):
        self.t_f = t_f
        self.t_restart = t_restart
        self.run_dir_path = run_dir_path
        self.name = name if name is not None else run_dir_path.split('-')[-1]
        self.n_outputs = int(t_f / t_restart)
        if load_csv_data:
            csv_file_root = os.path.split(run_dir_path)[1]
            csv_file_names = ['{}_{:03d}.csv'.format(
                csv_file_root, i) for i in range(self.n_outputs)]
            self.data_obj_list = self._load_numa_csv_data(csv_file_names)
        self.t, self.shore_max = self._load_shore_data(shore_file_name)

    def _load_numa_csv_data(self, csv_file_names):
        """
        load data from model output csvs
        :param csv_file_names:
        :return:
        """
        data_obj_list = []
        for csv_file_name in csv_file_names:
            csv_file_path = os.path.join(self.run_dir_path, csv_file_name)
            data_obj_list.append(NumaCsvData(csv_file_path))
        return data_obj_list

    def _load_shore_data(self, shore_file_name):
        """
        load shore max timeseries from model run shore line max output data file
        :param shore_file_name:
        :return:
        """
        t = []
        shore_max = []
        shore_file_path = os.path.join(self.run_dir_path, shore_file_name)
        with open(shore_file_path, 'r') as file:
            for i, r in enumerate(file):
                r = r.strip().lstrip().split()
                t.append(float(r[0]))
                shore_max.append(float(r[1]))
        return np.asarray(t), np.asarray(shore_max)

    def _animate_height_3D_helper(self, height='height', cmap='jet'):
        """
        returns iterable of plots from NumaCsvData.plot_height_3D
        to be used by NumaRunData.animate_height_3D
        """
        iterable = [ob.plot_height_3D(
            height=height,
            return_fig=False,
            cmap=cmap
        ) for ob in self.data_obj_list]
        return iterable

    def animate_height_3D(self, save_file_path=None, figsize=(12,7),
                          height='height', cmap='jet'):
        """
        create animation of NumaCsvData.plot_height_3D
        """
        if save_file_path is None:
            save_file_path = os.path.join(self.run_dir_path, '.mp4')
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111, projection='3d')
        ani = mpl.animation.FuncAnimation(
            fig,
            self._animate_height_3D_helper,
            frames=self.n_outputs,
            fargs={'height': height, 'cmap': cmap}
        )
        ani.save(save_file_path)


    def plot_shore_max_timeseries(
            self,
            figsize=(12,7),
            figure_instance=None,
            color='k',
            marker='.'
    ):
        if figure_instance is None:
            figure_instance = plt.figure(figsize=figsize)
        p = plt.plot(self.t, self.shore_max, c=color, marker=marker,
                 figure=figure_instance, mew=0)[0]
        p.set_label(self.name)
        return figure_instance


    def __repr__(self):
        return "NumaRunData({}, {}, {})".format(
            self.t_f,
            self.t_restart,
            self.run_dir_path
        )