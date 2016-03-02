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
import scipy.interpolate


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

def get_run_dirs(dir_prefix="failed_"):
    """
    returns list of all dir names in curdir`
    """
    ld = len(dir_prefix)
    xs = os.walk(os.curdir)
    return [x[0] for x in xs if (x[0] != '.' and x[0][:ld] != dir_prefix)]


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

class ObstacleOutlines:
    """
    class to hold obstacle outline data
    """
    def __init__(self, X, Y, B, x_range=(0,500)):
        """
        use bathy to find obstacle base contours using constant slope of domain
        """
        self.X = X
        self.Y = Y
        ## get a bathy with no hills
        min_x = X > x_range[0]
        max_x = X < x_range[1]
        filtr = np.invert(min_x*max_x)
        sub_X = X[filtr]
        sub_Y = Y[filtr]
        sub_B = B[filtr]
        ## interpolate
        shp = X.shape
        S = scipy.interpolate.griddata(
            (sub_X.flatten(), sub_Y.flatten()),
            sub_B.flatten(),
            (X.flatten(), Y.flatten())
        )
        S = S.reshape(shp)
        ## get obstacles
        O = B - S
        ## get rid of small deviations from 0
        O[O < .99] = 0
        self.outlines = O
        

def add_topo_contours(ax, o_o, ls="-", color="darkslategray", lw=2):
    """
    add shoreline and obstacle contours to the plot
    """
    ax.axvline(0, linestyle=ls, linewidth=lw, color=color, zorder=100)
    ax.contour(o_o.X, o_o.Y, o_o.outlines, [1], linestyles=ls, 
        linewidths=lw, colors=color, zorder=101)
    return ax

class GridSpec:
    """
    class to hold grid data for interpolation
    """
    def __init__(self, X, Y):
        """
        prepare X and Y to interpolate data onto evenly spaced grid using 
        scipy.interpolate.griddata
        """
        self.xi = np.linspace(X[0,0], X[0,-1], X.shape[1])
        self.yi = np.linspace(Y[0,0], Y[-1,0], Y.shape[0])
        self.Xi, self.Yi = np.meshgrid(self.xi, self.yi)
        self.shp = self.Xi.shape
        self.Xi_ = self.Xi.flatten()
        self.Yi_ = self.Yi.flatten()
        self.X_ = X.flatten()
        self.Y_ = Y.flatten()

    def out(self):
        return self.Xi_, self.Yi_, self.X_, self.Y_, self.shp

def remove_outliers(array, n_stds):
    """
    remove data greater than n_stds standard deviations from the mean
    """
    mn = array.mean()
    std = array.std()
    return array[np.abs(array) <= mn + n_stds*std]

def plot_function_val_spacing(nrds, x_coord, y_coord="avg", plot_max=True,
                                spacing='dist', function="get_kinetic_energy",
                                y_label="", sort_type=str, x_spacing=True,
                                filter_outliers=None, x_width=None,
                                colormap='viridis', regex_first_spacing_val_ind=1,
                                color_dict=None, marker_dict=None, ncol=2):
    """
    plot data values specified in kwarg function at location x_coord in
    run, plot by spacing found using regex_string

    pass in a color_dict and marker_dict to create custom symbology
    """
    if not color_dict or not marker_dict:
        ## set up color map and markers
        n = len(nrds)
        cmap_vals = np.linspace(0, 255, n, dtype=int)
        cmap = plt.get_cmap(colormap)
        colors = [cmap(val) for val in cmap_vals]
        markers = 'Hs^Dvo'
        markers *= int(np.ceil(n/len(marker)))
    ## sort input arguments by value
    inds = np.argsort([n.get_sort_val('name', sort_type) for n in nrds])
    nrds = np.asarray(nrds)[inds]
    hands, labs = [], []
    max_x, min_x = 0, -1
    fig = plt.figure(figsize=(18,8))
    ax = plt.subplot(121)
    ## loop over NumaRunData objects
    for i, nrd in enumerate(nrds):
        ## call function to calc y value
        y_val = getattr(nrd, function)(x_coord, y_coord, plot_max,
                                       filter_outliers, x_width)
        ## get spacing vals from run_dir_path attribute
        re_obj = re.compile('\d+')
        m = re_obj.findall(nrd.run_dir_path)
        m = np.asarray(m, dtype=float)
        ## choose which spacing characteristic to plot
        ind = regex_first_spacing_val_ind
        if spacing == "dist":
            x_val = np.sqrt(m[ind]**2 + m[ind+1]**2)
            x_label_prefix = "Total "
        elif spacing == "dx":
            x_val = m[ind]
            x_label_prefix = "Crossshore "
        elif spacing == "dy" and x_spacing:
            x_val = m[ind+1]
            x_label_prefix = "Alongshore "
        elif spacing == "dy" and not x_spacing:
            x_val = m[ind]
            x_label_prefix = "Alongshore "
        if color_dict and marker_dict:
            color = color_dict[m[ind]]
            if x_spacing:
                marker = marker_dict[m[ind+1]]
            else:
                marker = marker_dict[m[ind]]
        else:
            color = colors[i]
            marker = markers[i]
        p, = plt.plot(x_val, y_val,
                color=color,
                marker=marker,
            )  
        if x_val > max_x:
            max_x = x_val
        elif x_val < min_x or min_x < 0:
            min_x = x_val
        hands.append(p)  
        labs.append(nrd.legend_label())
    plt.xlim(left=min_x-(0.1*max_x), right=1.1*max_x)
    plt.title("x = {}".format(x_coord))
    plt.xlabel("{}distance between obstacle centers [m]".format(x_label_prefix))
    plt.ylabel(y_label)
    fig.legend(hands, labs, loc=5, numpoints=1, ncol=ncol)
    return fig

def plot_shore_max_timeseries(nrds, figsize=(18,8), colormap='viridis', sort_type=int):
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
    inds = np.argsort([n.get_sort_val('name', sort_type) for n in nrds])
    nrds = np.asarray(nrds)[inds]
    ## initialize figure instance
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(121)
    for i, nrd in enumerate(nrds):
        fig = nrd.plot_shore_max_timeseries(
            figure_instance=fig,
            color=color[i],
            marker=marker[i],
        )
    ax.set_ylabel('Time [s]')
    ax.set_xlabel('Max shore position [m]')
    ax.set_ylim(top=nrd.t_f)
    hands, labs = ax.get_legend_handles_labels()
    fig.legend(hands, labs, loc=5, numpoints=1, ncol=2)
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

    def plot_depth_velocity(self, figsize=(12,7), height='height', plot_every=1,
                                uvelo='uvelo', vvelo='vvelo', bathy="bathymetry", cmap="viridis",
                                xmin=None, boolean_depth=None, ax_instance=None, clims=None):
        """
        plot the depth
        overlay velocity vectors
        """
        U = getattr(self, uvelo)
        V = getattr(self, vvelo)
        H = getattr(self, height)
        B = getattr(self, bathy)
        if xmin is not None:
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
        depth = H - B
        ## convert the depth to 0 or 1 depending whether or not there is water there
        if boolean_depth is not None:
            filtr = depth > boolean_depth
            depth[filtr] = 1
            depth[np.invert(filtr)] = 0
        if ax_instance is None:
            fig = plt.figure(figsize=figsize)
            ax = plt.subplot(111)
            ob_ol = ObstacleOutlines(X, Y, B)
            ax = add_topo_contours(ax, ob_ol)
            # ax.contour(X, Y, B, linestyles='dashed', colors='DarkGray', vmin=0, vmax=50)
        else:
            ax = ax_instance
        if clims:
            P = ax.pcolor(X, Y, depth, cmap=cmap, vmin=clims[0], vmax=clims[1])
        else:
            P = ax.pcolor(X, Y, depth, cmap=cmap)
        Q = ax.quiver(X[:,::plot_every], Y[:,::plot_every], U[:,::plot_every],
                      V[:,::plot_every], color="k")
        qk = ax.quiverkey(Q, 0.9, 0.95, 0.5, r'$0.5 \frac{m}{s}$',
                   labelpos='E',
                   coordinates='figure',
                   fontproperties={'weight': 'bold'})
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        if ax_instance is None:
            return fig
        else:
            return P, Q

    def plot_devheight_velocity(self, figsize=(12,7), height='height', plot_every=1, 
                                uvelo='uvelo', vvelo='vvelo', bathy="bathymetry", cmap="viridis",
                                xmin=None, stillwater=None, ax_instance=None, clims=None):
        """
        plot the deviation in height from still water
        overlay velocity vectors
        """
        U = getattr(self, uvelo)
        V = getattr(self, vvelo)
        H = getattr(self, height)
        B = getattr(self, bathy)
        if xmin is not None:
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
        if stillwater is None:
            stillwater = B.copy()
            stillwater[B < 0] = 0
        devH = H - stillwater
        if ax_instance is None:
            fig = plt.figure(figsize=figsize)
            ax = plt.subplot(111)
            ob_ol = ObstacleOutlines(X, Y, B)
            ax = add_topo_contours(ax, ob_ol)
            # ax.contour(X, Y, B, linestyles='dashed', colors='DarkGray', vmin=0, vmax=50)
        else:
            ax = ax_instance
        if clims:
            P = ax.pcolor(X, Y, devH, cmap=cmap, vmin=clims[0], vmax=clims[1])
        else:
            P = ax.pcolor(X, Y, devH, cmap=cmap)
        Q = ax.quiver(X[:,::plot_every], Y[:,::plot_every], U[:,::plot_every], V[:,::plot_every], color="k")
        qk = ax.quiverkey(Q, 0.9, 0.95, 1, r'$1 \frac{m}{s}$',
                   labelpos='E',
                   coordinates='figure',
                   fontproperties={'weight': 'bold'})
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        if ax_instance is None:
            return fig
        else: 
            return P, Q

    def plot_velocity_streamlines(self, figsize=(12,7), density=1, norm=None,
                                  ax_instance=None, x_range=None, uvelo='uvelo',
                                  grid_spec=None, vvelo='vvelo', cmap="viridis",
                                  bathy="bathymetry", arrowsize=1):
        """
        streamline plot of velocity
        """
        U = getattr(self, uvelo)
        V = getattr(self, vvelo)
        ## for obstacle outlines only
        B = getattr(self, bathy)
        if x_range is not None:
            min_x = self.x[0,:].searchsorted(x_range[0])
            if x_range[1] == 'end':
                max_x = -1
            else:
                max_x = self.x[0,:].searchsorted(x_range[1])
            U = U[:,min_x:max_x]
            V = V[:,min_x:max_x]
            X = self.x[:,min_x:max_x]
            Y = self.y[:,min_x:max_x]
            B = B[:,min_x:max_x]
        else:
            X = self.x
            Y = self.y
            x_range = (X[0,0], X[0,-1])
        
        ## interpolate onto evenly spaced grid
        if grid_spec:
            ## option to pass in precomputed object of grid_for_interpolation function
            Xi_, Yi_, X_, Y_, shp = grid_spec.out()
        else:
            ## compute interpolated grid
            Xi_, Yi_, X_, Y_, shp = GridSpec(X, Y).out()
        Ui = scipy.interpolate.griddata((X_, Y_), U.flatten(), (Xi_, Yi_)).reshape(shp)
        Vi = scipy.interpolate.griddata((X_, Y_), V.flatten(), (Xi_, Yi_)).reshape(shp)  
        Xi = Xi_.reshape(shp)
        Yi = Yi_.reshape(shp)
        speedi = np.sqrt(Ui**2 + Vi**2)     
        if ax_instance is None:
            fig = plt.figure(figsize=figsize)
            ax = plt.subplot(111)
            obstacle_outlines = ObstacleOutlines(X, Y, B)
            ax = add_topo_contours(ax, obstacle_outlines)
            p = ax.streamplot(Xi[0,:], Yi[:,0], Ui, Vi, color=speedi, 
                norm=norm, cmap=cmap, arrowsize=arrowsize, density=density)
            # ax.set_xlim(left=x_range[0], right=x_range[1])
            return fig
        else:
            ax = ax_instance
            p = ax.streamplot(Xi[0,:], Yi[:,0], Ui, Vi, norm=norm,
                arrowsize=arrowsize, color=speedi, cmap=cmap, density=density)
            return p

    def plot_kinetic_energy(self, figsize=(12,7), cmap='viridis', xmin=None,
                            uvelo='uvelo', vvelo='vvelo', height='height',
                            bathy='bathymetry'):
        """
        pcolor plot of kinetic energy
        """
        U = getattr(self, uvelo)
        V = getattr(self, vvelo)
        H = getattr(self, height)
        B = getattr(self, bathy)
        if xmin is not None:
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

    def plot_energy(self, figsize=(12,7), cmap='viridis', xmin=None,
                    height='height', bathy='bathymetry',
                    uvelo='uvelo', vvelo='vvelo'):
        """
        pcolor plot of total energy
        """
        U = getattr(self, uvelo)
        V = getattr(self, vvelo)
        H = getattr(self, height)
        B = getattr(self, bathy)
        if xmin is not None:
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
                     bathy='bathymetry', cmap='viridis', stride=1):
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
        ## monocolor bathy for animation
        if cmap is None:
            ax.plot_surface(X, Y, B, rstride=stride, cstride=stride, color='b')
        else:
            ax.plot_surface(X, Y, B, rstride=stride, cstride=stride, cmap=cmap)
        return fig

    def plot_height_3D(self, figsize=(14,7), return_fig=True, ax_instance=None,
                       height='height', bathy='bathymetry', clims=None,
                       cmap='viridis', xmin=None, timestamp=None, stride=1):
        """
        plot height as 3D surface and color by depth
        (optionally) bathymetry as 3D surfaces
        """
        H = getattr(self, height)
        B = getattr(self, bathy)        
        X = self.x
        Y = self.y
        if xmin is not None:
            ind = X[0,:].searchsorted(xmin)
            H = H[:,ind:]
            B = B[:,ind:]
            X = X[:,ind:]
            Y = Y[:,ind:]
        ## set up color map
        depth = H - B
        colormap = plt.get_cmap(cmap)
        if clims is not None:
            vmin = clims[0]
            vmax = clims[1]
        else:
            vmin = depth.min()
            vmax = depth.max()
        ## array for colors
        norm_depth = depth/vmax
        norm_depth[norm_depth > 1] = 1.0
        norm_depth = norm_depth.astype(str)
        if return_fig:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_zlabel('z [m]')
            ax.invert_xaxis()
            ax.plot_surface(X, Y, B, rstride=stride, cstride=stride)
        ## ax instance passed in as argument
        else:
            ax = ax_instance
            # ax.text(.97, .97, "t = {} s".format(timestamp), transform=ax.transAxes)
        ## plot height surface regardless of return_fig
        p = ax.plot_surface(X, Y, H, cmap=colormap, facecolors=norm_depth,
                            rstride=stride, cstride=stride, vmin=vmin, vmax=vmax)
        if return_fig:
            cb = fig.colorbar(p, shrink=.7)
            cb.set_label('Depth [m]')
            return fig
        ## for use in animation function
        else:
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
            regex_string=None,
            sort_val=None,
    ):
        self.t_f = t_f
        self.t_restart = t_restart
        self.run_dir_path = run_dir_path
        ## useful name for object
        if regex_string is not None:    
            p = re.compile(regex_string)
            m = p.search(self.run_dir_path)
            try:
                self.name = m.group().replace("_", " ")
            except AttributeError as e:
                print("NumaRunData.__init__: No match found for regex_string")
                self.name = ""
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
            ncd = NumaCsvData(csv_file_path)
            data_obj_list.append(ncd)
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

    def animate_velocity_streamlines(self, save_file_path=None, figsize=(14,7), uvelo="uvelo",
                          vvelo="vvelo", cmap='plasma', bathy='bathymetry', arrowsize=1,
                          interval=200, x_range=None, density=1, t_range=None, clims=None):
        """
        create an animation of NumaCsvData.plot_velocity_streamlines
        """
        if save_file_path is None:
            save_file_path = self.run_dir_path + '_devheight_velocity.mp4'
        obs_to_plot = self.data_obj_list
        ## t_range is a tuple containing 2 indices into a self.data_obj_list
        if t_range is not None:
            start, end = t_range
            if start <= end and start >= 0:
                obs_to_plot = obs_to_plot[start:end]
        else:
            start = 0
        ob0 = obs_to_plot[0]
        B = getattr(ob0, bathy)
        ## set up for interpolated grid and obstacle outlines
        if x_range is not None:
            min_x = ob0.x[0,:].searchsorted(x_range[0])
            if x_range[1] == 'end':
                max_x = -1
            else:
                max_x = ob0.x[0,:].searchsorted(x_range[1])
            X = ob0.x[:,min_x:max_x]
            Y = ob0.y[:,min_x:max_x]
            B = B[:,min_x:max_x]
        else:
            X = ob0.x
            Y = ob0.y
            x_range = (X[0,0], X[0,-1])
        ## initialize GridSpec and ObstacleOutlines
        g_s = GridSpec(X, Y)
        o_o = ObstacleOutlines(X, Y, B)
        ## set colormap value limits
        if clims is not None:
            norm = mpl.colors.Normalize(clims[0], clims[1])
        else:
            norm = None
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)
        ax = add_topo_contours(ax, o_o)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ## quiver legend set; get handle and make colorbar
        sc = ob0.plot_velocity_streamlines(density=density, grid_spec=g_s, cmap=cmap,
            norm=norm, ax_instance=ax, x_range=x_range, arrowsize=arrowsize)
        cb = fig.colorbar(sc.lines, extend='max')
        cb.set_label("Velocity [m/s^2]")
        ## hide plotted stuff
        sc.lines.set_visible(False)
        for child in ax.get_children():
            if isinstance(child, mpl.patches.FancyArrowPatch):
                child.set_visible(False)
        # create list of drawables to pass to animation
        list_plots = []
        for i, ob in enumerate(obs_to_plot):
            ## hide old children
            for child in ax.get_children():
                if isinstance(child, mpl.patches.FancyArrowPatch):
                    child.set_visible(False)
            sc = ob.plot_velocity_streamlines(
                norm=norm,
                density=density,
                cmap=cmap,
                x_range=x_range,
                grid_spec=g_s,
                ax_instance=ax,
                arrowsize=arrowsize
            )
            ## mpl.streamplot is poorly written...
            artists = [sc.lines]
            for child in ax.get_children():
                ## SOO UGLY!!!
                if child.get_visible() and isinstance(child, mpl.patches.FancyArrowPatch):
                    artists.append(child)
            list_plots.append(artists)
        ani = mpl.animation.ArtistAnimation(fig, list_plots, interval=interval)
        ani.save(save_file_path, dpi=300)


    def animate_devheight_velocity(self, save_file_path=None, figsize=(14,7), uvelo="uvelo",
                          vvelo="vvelo", height='height', cmap='magma', bathy='bathymetry',
                          interval=200, xmin=None, plot_every=1, t_range=None, clims=None):
        """
        create an animation of NumaCsvData.plot_height_velocity
        """
        if save_file_path is None:
            save_file_path = self.run_dir_path + '_devheight_velocity.mp4'
        obs_to_plot = self.data_obj_list
        ## t_range is a tuple containing 2 indices into a self.data_obj_list
        if t_range is not None:
            start, end = t_range
            if start <= end and start >= 0:
                obs_to_plot = obs_to_plot[start:end]
        else:
            start = 0
        ob0 = obs_to_plot[0]
        B = getattr(ob0, bathy)
        H = getattr(ob0, height)        
        if xmin is not None:
            ind = ob0.x[0,:].searchsorted(xmin)
            B = B[:,ind:]
            H = H[:,ind:]
            X = ob0.x[:,ind:]
            Y = ob0.y[:,ind:]
        else:
            X = ob0.x
            Y = ob0.y
        ## calculate still water level
        stillwater = B.copy()
        stillwater[B < 0] = 0
        devH = H - stillwater
        ## caculate obstacle outlines
        o_o = ObstacleOutlines(X, Y, B)
        ## arbitrary max and min value for colorscaling
        if clims is None:        
            clims = (devH.min(), devH.max())
        fig = plt.figure(figsize=figsize)
        fig.suptitle(self.legend_label())
        ax = plt.subplot(111)
        ax = add_topo_contours(ax, o_o)
        ## quiver legend set; get handle and make colorbar
        P, Q = ob0.plot_devheight_velocity(ax_instance=ax, xmin=xmin, plot_every=plot_every, 
            stillwater=stillwater, clims=clims, cmap=cmap)
        cb = fig.colorbar(P, extend='both')
        cb.set_label("Height above stillwater [m]")
        # create list of drawables to pass to animation
        list_plots = []
        for i, ob in enumerate(obs_to_plot):
            list_plots.append(ob.plot_devheight_velocity(
                ax_instance=ax,
                xmin=xmin,
                plot_every=plot_every,
                stillwater=stillwater,
                clims=clims,
                cmap=cmap
            ))

        ## do animation
        ani = mpl.animation.ArtistAnimation(fig, list_plots, interval=interval)
        ani.save(save_file_path, dpi=300)

    def animate_height_3D(self, save_file_path=None, figsize=(14,7), stride=1,
                          height='height', cmap='viridis', bathy='bathymetry',
                          interval=100, xmin=None, t_range=None, regex_string=None):
        """
        create animation of NumaCsvData.plot_height_3D
        """
        if save_file_path is None:
            save_file_path = self.run_dir_path + '_height_3D.mp4'
        obs_to_plot = self.data_obj_list
        ## t_range is a tuple containing 2 indices into a self.data_obj_list
        if t_range is not None:
            start, end = t_range
            if start <= end and start >= 0:
                obs_to_plot = obs_to_plot[start:end]
        else:
            start = 0
        ob0 = obs_to_plot[0]
        H = getattr(ob0, height)
        B = getattr(ob0, bathy)
        if xmin is not None:
            ind = ob0.x[0,:].searchsorted(xmin)
            B = B[:,ind:]
            H = H[:,ind:]
        depth = H - B
        clims = (0, np.max(depth))
        fig = ob0.plot_bathy_3D(figsize=figsize, bathy=bathy, 
                                cmap=None, xmin=xmin, stride=stride)
        fig.suptitle(self.legend_label(regex_string))
        ax = fig.get_axes()[0]
        # create list of drawables to pass to animation
        list_plots = []
        for i, ob in enumerate(obs_to_plot):
            p = ob.plot_height_3D(
                return_fig=False,
                ax_instance=ax,
                height=height,
                bathy=bathy,
                clims=clims,
                cmap=cmap,
                xmin=xmin,
                stride=stride,
                timestamp=start+i*self.t_restart
            )
            list_plots.append([p])
        # cb = fig.colorbar(to_map)
        # cb.set_label('Depth [m]')
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
                           filter_outliers=None, x_width=None,
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
                ## to do average over area rather than line
                if x_width is not None:
                    ind2 = ob.x[0,:].searchsorted(x_coord+x_width)
                else:
                    ind2 = ind + 1
                U = U[:,ind:ind2]
                V = V[:,ind:ind2]
                H = H[:,ind:ind2]
                B = B[:,ind:ind2]
                ## get average value across domain
                E = 0.5 * (H-B) * (U**2+V**2)
                ## remove data more than filter_outliers standard deviations from mean
                if filter_outliers is not None:
                    E = remove_outliers(E, filter_outliers)
                E = np.mean(E)
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
                           filter_outliers=None, x_width=None,
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
                ## to do average over area rather than line
                if x_width is not None:
                    ind2 = ob.x[0,:].searchsorted(x_coord+x_width)
                else:
                    ind2 = ind + 1
                U = U[:,ind:ind2]
                V = V[:,ind:ind2]
                ## get average value across domain
                vel = np.sqrt(U**2+V**2)
                ## remove data more than filter_outliers standard deviations from mean
                if filter_outliers is not None:
                    vel = remove_outliers(vel, filter_outliers)
                vel = np.mean(vel)
            else:
                if y_coord == "mid":
                    ind_y = int(np.ceil(ob.x.shape[0]/2.))                 
                else:
                    ind_y = ob.y[:,0].searchsorted(y_coord)
                U = U[ind_y,ind]
                V = V[ind_y,ind]
                vel = np.sqrt(U**2+V**2)
            velocity.append(vel)
        velocity = np.asarray(velocity)
        ## instantaneous max
        if return_max:
            return np.nanmax(velocity)
        ## return the time average value
        else:
            return velocity.mean()

    def get_height(self, x_coord, y_coord='avg', return_max=True, 
                            filter_outliers=None, x_width=None,
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
                ## to do average over area rather than line
                if x_width is not None:
                    ind2 = ob.x[0,:].searchsorted(x_coord+x_width)
                else:
                    ind2 = ind + 1
                H = H[:,ind:ind2]
                B = B[:,ind:ind2]
                ## get average value across domain
                h = H - B
                ## remove data more than filter_outliers standard deviations from mean
                if filter_outliers is not None:
                    h = remove_outliers(h, filter_outliers)
                h = np.mean(h)
            else:
                if y_coord == "mid":
                    ind_y = int(np.ceil(ob.x.shape[0]/2.))                 
                else:
                    ind_y = ob.y[:,0].searchsorted(y_coord)
                H = H[ind_y,ind]
                B = B[ind_y,ind]
                h = H-B
            hgt.append(h)
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
                 figure=figure_instance, ls="None", mew=0)[0]
        p.set_label(self.legend_label(regex_string=regex_string))
        return figure_instance


    def __repr__(self):
        return "NumaRunData({}, {}, {})".format(
            self.t_f,
            self.t_restart,
            self.run_dir_path
        )