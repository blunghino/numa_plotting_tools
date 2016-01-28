import csv
import os
import re
import pickle
import warnings

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import matplotlib as mpl


## set font to times new roman
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'

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

def plot_function_val_spacing(nrds, x_coord, y_coord="avg", plot_max=True,
                                spacing='dist', function="get_kinetic_energy",
                                y_label="",
                                colormap='viridis', regex_string=None):
    """
    plot data values specified in kwarg function at location x_coord in
    run, plot by spacing found using regex_string
    """
    ## set up color map and markers
    n = len(nrds)
    cmap_vals = np.linspace(0, 255, n, dtype=int)
    cmap = plt.get_cmap(colormap)
    color = [cmap(val) for val in cmap_vals]
    marker = 'Hs^Dvo'
    marker *= int(np.ceil(n/len(marker)))
    ## sort input arguments by value
    inds = np.argsort([n.get_sort_val('name', int) for n in nrds])
    nrds = np.asarray(nrds)[inds]
    hands, labs = [], []
    max_x, min_x = 0, -1
    fig = plt.figure(figsize=(18,8))
    ax = plt.subplot(121)
    ## loop over NumaRunData objects
    for i, nrd in enumerate(nrds):
        ## call function to calc y value
        y_val = getattr(nrd, function)(x_coord, y_coord, plot_max)
        ## get spacing vals from run_dir_path attribute
        re_obj = re.compile('\d+')
        m = re_obj.findall(nrd.run_dir_path)
        m = np.asarray(m, dtype=float)
        ## choose which spacing characteristic to plot
        if spacing == "dist":
            x_val = np.sqrt(m[1]**2 + m[2]**2)
            x_label_prefix = "Total "
        elif spacing == "dx":
            x_val = m[1]
            x_label_prefix = "Crossshore "
        elif spacing == "dy":
            x_val = m[2]
            x_label_prefix = "Alongshore "
        p, = plt.plot(x_val, y_val,
                color=color[i],
                marker=marker[i],
            )  
        if x_val > max_x:
            max_x = x_val
        elif x_val < min_x or min_x < 0:
            min_x = x_val
        hands.append(p)  
        labs.append(nrd.legend_label(regex_string=regex_string))
    plt.xlim(left=min_x-(0.1*max_x), right=1.1*max_x)
    plt.title("x = {}".format(x_coord))
    plt.xlabel("{}distance between obstacle centers [m]".format(x_label_prefix))
    plt.ylabel(y_label)
    fig.legend(hands, labs, loc=5, numpoints=1, ncol=2)
    return fig

def plot_shore_max_timeseries(nrds, figsize=(12,9), colormap='viridis',
                              legend_title=None, legend_position=2, 
                              regex_string=None):
    """
    call the NumaRunData.plot_shore_max method for a list of NumaRunData objects
    plot all lines on one matplotlib.figure.Figure instance
    """
    ## set up color map and markers
    n = len(nrds)
    cmap_vals = np.linspace(0, 255, n, dtype=int)
    cmap = plt.get_cmap(colormap)
    color = [cmap(val) for val in cmap_vals]
    marker = 'Hs^Dvo'
    marker *= int(np.ceil(n/len(marker)))
    ## sort input arguments by value
    inds = np.argsort([n.get_sort_val('name', int) for n in nrds])
    nrds = np.asarray(nrds)[inds]
    ## initialize figure instance
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    for i, nrd in enumerate(nrds):
        fig = nrd.plot_shore_max_timeseries(
            figure_instance=fig,
            color=color[i],
            marker=marker[i],
            regex_string=regex_string
        )
    ax.legend(loc=legend_position, title=legend_title)
    ax.set_ylabel('Time [s]')
    ax.set_xlabel('Max shore position [m]')
    ax.set_ylim(top=nrd.t_f)
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
                                  xmin=0, uvelo='uvelo', vvelo='vvelo'):
        """
        streamline plot of velocity
        """
        U = getattr(self, uvelo)
        V = getattr(self, vvelo)
        if xmin:
            ind = self.x[0,:].searchsorted(xmin)
            U = U[:,ind:]
            V = V[:,ind:]
            X = self.x[:,ind:]
            Y = self.y[:,ind:]
        else:
            X = self.x
            Y = self.y
        speed = np.sqrt(U**2 + V**2)
        if return_fig:
            fig = plt.figure(figsize=figsize)
            p = plt.streamplot(X, Y, U, V, color=speed)
            return fig
        else:
            p = plt.streamplot(X, Y, U, V, color=speed)
            return p

    def plot_kinetic_energy(self, figsize=(12,7), cmap='viridis', xmin=0,
                            uvelo='uvelo', vvelo='vvelo', height='height',
                            bathy='bathymetry'):
        """
        pcolor plot of kinetic energy
        """
        U = getattr(self, uvelo)
        V = getattr(self, vvelo)
        H = getattr(self, height)
        B = getattr(self, bathy)
        if xmin:
            ind = self.x[0,:].searchsorted(xmin)
            U = U[:,ind:]
            V = V[:,ind:]
            H = H[:,ind:]
            B = B[:,ind:]
            X = self.x[:,ind:]
            Y = self.y[:,ind:]
        else:
            X = self.x
            Y = self.y
        Ek = 0.5 * (H - B) * (U**2 + V**2)
        fig = plt.figure(figsize=figsize)
        plt.pcolor(X, Y, Ek, cmap=cmap)
        cbar = plt.colorbar()
        cbar.set_label('Kinetic Energy [m^2/s^2]')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        return fig

    def plot_energy(self, figsize=(12,7), cmap='viridis', xmin=0,
                    height='height', bathy='bathymetry',
                    uvelo='uvelo', vvelo='vvelo'):
        """
        pcolor plot of total energy
        """
        U = getattr(self, uvelo)
        V = getattr(self, vvelo)
        H = getattr(self, height)
        B = getattr(self, bathy)
        if xmin:
            ind = self.x[0,:].searchsorted(xmin)
            U = U[:,ind:]
            V = V[:,ind:]
            H = H[:,ind:]
            X = self.x[:,ind:]
            Y = self.y[:,ind:]
        else:
            X = self.x
            Y = self.y
        g = 9.81
        Ek = 0.5 * (H - B) * (U**2 + V**2)
        Ep = g * H
        E = Ek + Ep
        fig = plt.figure(figsize=figsize)
        plt.pcolor(X, Y, E, cmap=cmap)
        cbar = plt.colorbar()
        cbar.set_label('Energy [m^2/s^2]')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        return fig

    def plot_bathy(self, figsize=(14,7), bathy='bathymetry', 
                   xmin=None, cmap='viridis'):
        """
        plot bathymetry as pcolor
        """
        B = getattr(self, bathy)
        X = self.x
        Y = self.y
        if xmin is not None:
            ind = X[0,:].searchsorted(xmin)
            B = B[:,ind:]
            X = X[:,ind:]
            Y = Y[:,ind:]
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.pcolor(X, Y, B, cmap=cmap)
        return fig

    def plot_bathy_3D(self, figsize=(14,7), xmin=None,
                     bathy='bathymetry', cmap='viridis'):
        """
        plot bathymetry as 3D surface
        """
        B = getattr(self, bathy)        
        X = self.x
        Y = self.y
        if xmin is not None:
            ind = X[0,:].searchsorted(xmin)
            B = B[:,ind:]
            X = X[:,ind:]
            Y = Y[:,ind:]
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        ax.invert_xaxis()
        ax.plot_surface(X, Y, B, rstride=1, cstride=1, cmap=cmap)
        return fig

    def plot_height_3D(self, figsize=(14,7), return_fig=True, ax_instance=None,
                       height='height', bathy='bathymetry', clims=None,
                       cmap='viridis'):
        """
        plot height and bathymetry as 2 3D surfaces
        """
        H = getattr(self, height)
        B = getattr(self, bathy)
        colormap = plt.get_cmap(cmap)
        if clims is not None:
            vmin = clims[0]
            vmax = clims[1]
        else:
            vmin = H.min()
            vmax = H.max()
        if return_fig:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_zlabel('z [m]')
            ax.invert_xaxis()
            ax.plot_surface(self.x, self.y, B, rstride=1, cstride=1)
        else:
            ax = ax_instance
        p = ax.plot_surface(self.x, self.y, H, cmap=colormap,
                            rstride=1, cstride=1, vmin=vmin, vmax=vmax)
        if return_fig:
            cb = fig.colorbar(p, shrink=.7)
            cb.set_label('Height [m]')
            return fig
        else:
            return [p]

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
            sort_val=None,
    ):
        self.t_f = t_f
        self.t_restart = t_restart
        self.run_dir_path = run_dir_path
        ## useful name for object
        if name is not None:
            self.name = name
        else:
            self.name = run_dir_path.split('-')[-1]
        ## value on which to sort in a sequence of NumaRunData objects
        self.sort_val = sort_val
        ## number of NumaCsvData files associated with object
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
        """
        data_obj_list = []
        for csv_file_name in csv_file_names:
            csv_file_path = os.path.join(self.run_dir_path, csv_file_name)
            data_obj_list.append(NumaCsvData(csv_file_path))
        return data_obj_list

    def _load_shore_data(self, shore_file_name):
        """
        load shore max timeseries from model run shore line max output data file
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

    def get_sort_val(self, sort_att='sort_val', sort_type=None):
        try:
            sort_val = getattr(self, sort_att)
        except AttributeError:
            sort_val = None
        if callable(sort_type):
            sort_val = sort_type(sort_val)
        return sort_val

    def legend_label(self, regex_string=None):
        """
        return a string to be used in a plot legend

        /carrier_three_obstacle_dx-90_dy-80.csv
        regex_string: 'dx-\d+_dy-\d+'
        returns "dx-90 dy-80"
        """
        if regex_string is None:
            return self.name
        else:
            p = re.compile(regex_string)
            m = p.search(self.run_dir_path)
            try:
                return m.group().replace("_", " ")
            except AttributeError as e:
                print("No match found for regex_string")
                raise e

    def animate_height_3D(self, save_file_path=None, figsize=(14,7),
                          height='height', cmap='jet', bathy='bathymetry',
                          interval=100):
        """
        create animation of NumaCsvData.plot_height_3D
        """
        if save_file_path is None:
            save_file_path = self.run_dir_path + '.mp4'
        ob0 = self.data_obj_list[0]
        H = getattr(ob0, height)
        clims = (H.min(), H.max())
        fig = ob0.plot_bathy_3D(figsize=figsize, bathy=bathy)
        fig.suptitle(os.path.split(self.run_dir_path)[-1])
        ax = fig.get_axes()[0]
        # create list of drawables to pass to animation
        list_plots = [ob.plot_height_3D(
            return_fig=False,
            ax_instance=ax,
            height=height,
            bathy=bathy,
            clims=clims,
            cmap=cmap
        ) for ob in self.data_obj_list]
        cb = fig.colorbar(list_plots[0][0])
        cb.set_label('Height [m]')
        ani = mpl.animation.ArtistAnimation(fig, list_plots, interval=interval)
        ani.save(save_file_path, dpi=300)

    def plot_max_energy_distance_time(self, figsize=(12,7),
                                      uvelo='uvelo', vvelo='vvelo',
                                      height='height', bathy='bathymetry'):
        """
        x axis time y axis distance marker size magnitude for maximum energy
        of wave
        """
        time = np.linspace(0, self.t_f, self.n_outputs)
        energy = []
        distance = []
        for ob in self.data_obj_list:
            U = getattr(ob, uvelo)
            V = getattr(ob, vvelo)
            H = getattr(ob, height)
            B = getattr(ob, bathy)
            E = 0.5 * (H - B) * (U**2 + V**2)
            ind = np.argmax(E)
            energy.append(E.flatten()[ind])
            distance.append(ob.x.flatten()[ind])
        energy = np.asarray(energy)
        fig = plt.figure(figsize=figsize)
        plt.plot(time, distance)
        plt.scatter(time, distance, s=energy/10, label=self.name)
        plt.xlabel('time [s]')
        plt.ylabel('distance [m]')
        leg = plt.legend(scatterpoints=1)
        leg.legendHandles[0]._sizes = [30]
        return fig
    
    def get_kinetic_energy(self, x_coord, y_coord='avg', return_max=True, 
                            uvelo='uvelo', vvelo='vvelo',
                            height='height', bathy='bathymetry'):
        """
        return the kinetic energy at (x_coord, y_coord) 
        (if y_coord is average, average along shore)
        if return_max, return the maximum instantaneous value 
        else return the time average value  
        """
        energy = []
        for ob in self.data_obj_list:
            ind = ob.x[0,:].searchsorted(x_coord)
            U = getattr(ob, uvelo)
            V = getattr(ob, vvelo)
            H = getattr(ob, height)
            B = getattr(ob, bathy)
            if y_coord == "avg":
                U = U[:,ind]
                V = V[:,ind]
                H = H[:,ind]
                B = B[:,ind]
                ## get average value across domain
                E = 0.5 * np.mean(H-B) * np.mean(U**2+V**2)
            else:
                if y_coord == "mid":
                    ind_y = int(np.ceil(ob.x.shape[0]/2.))                 
                else:
                    ind_y = ob.y[:,0].searchsorted(y_coord)
                U = U[ind_y,ind]
                V = V[ind_y,ind]
                H = H[ind_y,ind]
                B = B[ind_y,ind]            
                E = 0.5 * (H-B) * (U**2+V**2)
            energy.append(E)
        energy = np.asarray(energy)
        ## instantaneous max
        if return_max:
            return np.nanmax(energy)
        ## return the time average value
        else:
            return energy.mean()

    def get_velocity(self, x_coord, y_coord='avg', return_max=True, 
                            uvelo='uvelo', vvelo='vvelo'):
        """
        return the velocity at (x_coord, y_coord) 
        (if y_coord is average, average along shore)
        if return_max, return the maximum instantaneous value 
        else return the time average value  
        """
        velocity = []
        for ob in self.data_obj_list:
            ind = ob.x[0,:].searchsorted(x_coord)
            U = getattr(ob, uvelo)
            V = getattr(ob, vvelo)
            if y_coord == "avg":
                U = U[:,ind]
                V = V[:,ind]
                ## get average value across domain
                velocity.append(np.mean(U**2+V**2))
            else:
                if y_coord == "mid":
                    ind_y = int(np.ceil(ob.x.shape[0]/2.))                 
                else:
                    ind_y = ob.y[:,0].searchsorted(y_coord)
                U = U[ind_y,ind]
                V = V[ind_y,ind]
                velocity.append(U**2+V**2)
        velocity = np.asarray(velocity)
        ## instantaneous max
        if return_max:
            return np.nanmax(velocity)
        ## return the time average value
        else:
            return velocity.mean()

    def get_height(self, x_coord, y_coord='avg', return_max=True, 
                            height='height', bathy='bathymetry'):
        """
        return the kinetic energy at (x_coord, y_coord) 
        (if y_coord is average, average along shore)
        if return_max, return the maximum instantaneous value 
        else return the time average value  
        """
        hgt = []
        for ob in self.data_obj_list:
            ind = ob.x[0,:].searchsorted(x_coord)
            H = getattr(ob, height)
            B = getattr(ob, bathy)
            if y_coord == "avg":
                H = H[:,ind]
                B = B[:,ind]
                ## get average value across domain
                hgt.append(np.mean(H-B))
            else:
                if y_coord == "mid":
                    ind_y = int(np.ceil(ob.x.shape[0]/2.))                 
                else:
                    ind_y = ob.y[:,0].searchsorted(y_coord)
                H = H[ind_y,ind]
                B = B[ind_y,ind]            
                hgt.append(H-B)
        hgt = np.asarray(hgt)
        ## instantaneous max
        if return_max:
            return np.nanmax(hgt)
        ## return the time average value
        else:
            return hgt.mean()

    def plot_shore_max_timeseries(
            self,
            figsize=(12,7),
            figure_instance=None,
            color='k',
            marker='.',
            regex_string=None
    ):
        if figure_instance is None:
            figure_instance = plt.figure(figsize=figsize)
        p = plt.plot(self.shore_max, self.t, c=color, marker=marker,
                 figure=figure_instance, mew=0)[0]
        p.set_label(self.legend_label(regex_string=regex_string))
        return figure_instance


    def __repr__(self):
        return "NumaRunData({}, {}, {})".format(
            self.t_f,
            self.t_restart,
            self.run_dir_path
        )