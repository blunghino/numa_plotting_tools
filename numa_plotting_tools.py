import csv
import os
import re
import pickle
import warnings
import sys

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.animation
import matplotlib as mpl
import scipy.interpolate
import scipy.fftpack

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
    ## this is a workaround for a bug in python 3.5 i/o on OS X https://bugs.python.org/issue24658
    max_bytes = 2**31 - 1
    pkl_bytes = pickle.dumps(obj)
    n_bytes = len(pkl_bytes)
    try:
        ## Open file and dump object
        with open(path, 'wb') as output:
            for idx in range(0, n_bytes, max_bytes):
                output.write(pkl_bytes[idx:idx+max_bytes])
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
    ## this is a workaround for a bug in python 3.5 i/o on OS X https://bugs.python.org/issue24658
    bytes_in = bytearray(0)
    input_size = os.path.getsize(path)
    max_bytes = 2**31 - 1
    ## open file and load object
    with open(path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    return pickle.loads(bytes_in)

def potential_energy(depth, g=9.81):
    """
    potential energy in m^3/s^2
    E_p = rho * A * h * g * (h/2)
    """
    return 0.5 * g * depth**2

def kinetic_energy(depth, u_velo, v_velo):
    """
    kinetic energy in m^3/s^2
    E_k = rho * A * v^2 * (h/2)
    """
    return 0.25 * depth * (u_velo**2 + v_velo**2)

def get_run_dirs(dir_prefix="failed_"):
    """
    returns list of all dir names in os.curdir
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
    def __init__(self, X, Y, B, x_range=(0,300)):
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

def plot_function_val_spacing(nrds, x_coord, y_coord="avg", plot_type='max',
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
        markers *= int(np.ceil(n/len(markers)))
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
        y_val = getattr(nrd, function)(x_coord, y_coord, plot_type, x_width,
                                       filter_outliers)
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
        if x_val < min_x or min_x < 0:
            min_x = x_val
        hands.append(p)  
        labs.append(nrd.legend_label())
    plt.xlim(left=min_x-(0.1*max_x), right=1.1*max_x)
    plt.title("x = {}".format(x_coord))
    plt.xlabel("{}distance between obstacle centers [m]".format(x_label_prefix))
    plt.ylabel(y_label)
    fig.legend(hands, labs, loc=5, numpoints=1, ncol=ncol)
    return fig

def plot_function_val_spatial(nrds, x_coord, x_width=None, n_pts_per_nrd=10,
                              colors=None, labels=None, markers=None, figsize=(12,8),
                              x_range=(-200,"end"), function="get_spatial_kinetic_energy"):
    """
    plot a map view of where spatially the maximum values occur for a set of runs
    """
    n_runs = len(nrds)
    ## set up colors
    if colors is None:
        cmap_vals = np.linspace(0, 255, n_runs, dtype=int)
        cmap = plt.get_cmap(colormap)
        colors = [cmap(val) for val in cmap_vals]
    if markers is None:
        markers = 'Hs^Dvo'
        markers *= int(np.ceil(n_runs/len(markers)))
    max_nrd = nrds[-1]
    max_y_nrd = np.max(nrds[-1].data_obj_list[0].y)
    ## loop over NumaRunData objects and plot results
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    plt.axhline(0, color='darkslategray')
    handles = []
    for i, nrd in enumerate(nrds):
        ## call function to calc y value
        val, x, y = getattr(nrd, function)(x_coord, x_width, n_pts_per_nrd)
        zorder = 100 - i
        # ax.plot(x, y, color=colors[i], marker=markers[i], ls="none",
        #         zorder=zorder, label=labels[i])
        handles.append(
            plt.scatter(x, y, c=colors[i], s=val, zorder=zorder)
        )
        ## get the domain size for plotting base map
        temp_max_y = np.max(nrd.data_obj_list[0].y)
        plt.axhline(temp_max_y, linestyle='--', color=colors[i])
        ## store the max domain size for plotting obstacle outlines
        if temp_max_y > max_y_nrd:
            max_y_nrd = temp_max_y
            max_nrd = nrd
    ## plot obstacle outlines
    max_nrd_data = max_nrd.data_obj_list[0]
    X = max_nrd_data.x
    Y = max_nrd_data.y
    B = max_nrd_data.bathymetry
    X, Y, B = subsample_arrays_in_x(x_range, X, Y, B)
    obstacle_outlines = ObstacleOutlines(X, Y, B)
    ax = add_topo_contours(ax, obstacle_outlines)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.legend(handles, labels, scatterpoints=1)
    return fig

def plot_function_val_timeseries(nrds, x_coord, x_width=None, x_label="[]",
                                 figsize=(18,8), t_0=0, t_f=None, show_legend=True,
                                 colormap='plasma', colors=None, markers=None,
                                 y_coord='avg', function="get_kinetic_energy",
                                 ylims=None, xlims=None, semilog_x=False, title="",
                                 plot_type="timeseries", filter_outliers=None, labels=None):
    """
    call the `function` method of each NumaRunData object in `nrds`
    plot timeseries on one axis and return fig
    """
    n_runs = len(nrds)
    ## set up colors
    if colors is None:
        cmap_vals = np.linspace(0, 255, n_runs, dtype=int)
        cmap = plt.get_cmap(colormap)
        colors = [cmap(val) for val in cmap_vals]
    if markers is None:
        markers = 'Hs^Dvo'
        markers *= int(np.ceil(n_runs/len(markers)))
    ## loop over NumaRunData objects and plot results
    fig = plt.figure(figsize=figsize)
    for i, nrd in enumerate(nrds):
        ## to normalize sums to deal with different numbers of grid points in different domain sizes
        if plot_type == 'sum_timeseries':
            normalization_denominator = nrd.get_initial_condition_total_potential_energy()
            # print(normalization_denominator)
        ## set up y_axis data
        if t_f is None:
            y_val = np.arange(0, nrd.t_f, nrd.t_restart)
        else:
            y_val = np.arange(0, t_f, nrd.t_restart)
        ind = y_val.searchsorted(t_0)
        y_val = y_val[ind:]
        n_vals = len(y_val)
        ## call function to calc y value
        x_val = getattr(nrd, function)(x_coord, y_coord, plot_type, x_width,
                                       filter_outliers)
        zorder = 100 - i
        if labels is not None:
            label = labels[i]
        else:
            label = nrd.legend_label()
        if plot_type == "timeseries_min_max":
            plt.plot(x_val[0,ind:ind+n_vals], y_val, zorder=zorder, color=colors[i],
                     marker=markers[i], label=label)
            plt.plot(x_val[1,ind:ind+n_vals], y_val, '--', zorder=zorder, color=colors[i])
            plt.plot(x_val[2,ind:ind+n_vals], y_val, zorder=zorder, color=colors[i])
        elif plot_type == "timeseries_max":
            plt.plot(x_val[2,ind:ind+n_vals], y_val, zorder=zorder, color=colors[i],
                     marker=markers[i], label=label)
        elif plot_type == "sum_timeseries":
            x_val = x_val / normalization_denominator
            plt.plot(x_val[ind:ind+n_vals], y_val, zorder=zorder, color=colors[i],
                 marker=markers[i], label=label)
        else:
            plt.plot(x_val[ind:ind+n_vals], y_val, zorder=zorder, color=colors[i],
                 marker=markers[i], label=label)
    ## customize figure
    loc = 'center right'
    if semilog_x:
        plt.xscale('log')
        loc = 'lower right'
    if show_legend:
        plt.legend(loc=loc, numpoints=1)
    if ylims is not None:
        plt.ylim(ylims)
    if xlims is not None:
        plt.xlim(xlims)
    plt.ylabel("Time [s]")
    plt.xlabel(x_label)
    plt.title(title)
    return fig

def plot_shore_max_timeseries(nrds, figsize=(18,8), show_legend='new_axis',
                              markers=None, colors=None, colormap='viridis',
                              sort_type=int, xlims=None, ylims=None, labels=None):
    """
    call the NumaRunData.plot_shore_max method for a list of NumaRunData objects
    plot all lines on one matplotlib.figure.Figure instance
    """
    ## set up color map and markers
    n = len(nrds)
    ls = "None"
    if colors is None:
        cmap_vals = np.linspace(0, 255, n, dtype=int)
        cmap = plt.get_cmap(colormap)
        colors = [cmap(val) for val in cmap_vals]
    if markers is None:
        markers = 'Hs^Dvo'
        markers *= int(np.ceil(n/len(markers)))
    elif markers is '-':
        ls = '-'
        markers = ["None"] * n
    ## sort input arguments by value
    if sort_type is not None:
        inds = np.argsort([n.get_sort_val('name', sort_type) for n in nrds])
        nrds = np.asarray(nrds)[inds]
    ## initialize figure instance
    fig = plt.figure(figsize=figsize)
    if show_legend == 'new_axis':
        ax = plt.subplot(121)
    else:
        ax = plt.subplot(111)
    for i, nrd in enumerate(nrds):
        fig = nrd.plot_shore_max_timeseries(
            figure_instance=fig,
            color=colors[i],
            marker=markers[i],
            ls=ls,
            zorder=100-i
        )
    ax.set_ylabel('Time [s]')
    ax.set_xlabel('Max shore position [m]')
    if ylims is not None:
        ax.set_ylim(ylims)
    else:
        ax.set_ylim(top=nrd.t_f)
    if xlims is not None:
        ax.set_xlim(xlims)
    hands, labs = ax.get_legend_handles_labels()
    if labels is not None:
        labs = labels
    if show_legend is 'new_axis':
        fig.legend(hands, labs, loc=5, numpoints=1, ncol=2)
    elif show_legend:
        plt.legend(hands, labs, loc='upper left', numpoints=1, ncol=1)        
    return fig

def pcolor_timeseries_spacing_runup(nrds, spacings, wavelength=2000,
                                    n_samples=300, t_min=0,
                                    att="shore_max", cmap='plasma'):
    """
    x axis - non-dimensionalized spacing,
    y axis - time,
    colorvalue - max runup
    """
    x = np.asarray(spacings) / wavelength
    y = np.linspace(t_min, nrds[0].t_f, n_samples)
    # y = nrds[0].t
    z = np.zeros((n_samples, len(x)))
    for j, nrd in enumerate(nrds):
        z[:,j] = nrd.subsample_shore_position_timeseries(n_samples,
                                                         t_min=t_min, att=att)
        # z[:,j] = getattr(nrd, att)
    fig = plt.figure()
    plt.pcolor(x, y, z, cmap=cmap)
    return fig

def plot_runup_distribution_timeseries(nrd, figsize=(12,8), title=None):
    """
    plot run up, max runup, min runup, and std on one axis
    """
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    plus_std = nrd.shore_mean + nrd.shore_std
    minus_std = nrd.shore_mean - nrd.shore_std
    plt.plot(nrd.t, nrd.shore_max, c='k', ls='--')
    plt.plot(nrd.t, nrd.shore_min, c='k', ls='--')
    plt.plot(nrd.t, plus_std, c='k', ls='-')
    plt.plot(nrd.t, minus_std, c='k', ls='-')
    plt.plot(nrd.t, nrd.shore_mean, c='k', marker='o', ls="None", mew=0)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Shore position [m]')
    if title is not None:
        plt.title(title)
    return fig

def plot_total_energy_timeseries(nrd, figsize=(12,8), uvelo='uvelo', vvelo='vvelo',
                            height='height', bathy='bathymetry', title=None):
    """
    for one nrd, plot Ek, Ep, and total energy through time
    """
    n = len(nrd.data_obj_list)
    t = np.arange(0, nrd.t_f, nrd.t_restart)
    potential = np.zeros(n)
    kinetic = np.zeros(n)
    fig = plt.figure(figsize=figsize)
    ## for each timestep calculate the total Ep and Ek in the domain
    for i, ob in enumerate(nrd.data_obj_list):
        U = getattr(ob, uvelo)
        V = getattr(ob, vvelo)
        H = getattr(ob, height)
        B = getattr(ob, bathy)
        depth = H - B
        Ek = kinetic_energy(depth, U, V)
        Ep = potential_energy(depth)
        potential[i] = np.sum(Ep)
        kinetic[i] = np.sum(Ek)
    plt.semilogy(t, potential, 'b-', label='Potential')
    plt.semilogy(t, kinetic, 'r-', label='Kinetic')
    plt.semilogy(t, potential+kinetic, 'k-', label='Total')
    plt.xlabel("Time [s]")
    plt.ylabel("Energy [m^3/s^2]")
    plt.legend(loc='center right')
    if title is not None:
        plt.title(title)
    return fig

def subsample_arrays_in_x(x_range, X, *arrays):
    """
    given x-coordinates `X` and a range of values to subsample `x_range`
    filter `X` and all arrays in `*arrays` by range `x_range`
    return list beginning with `X` and followed by all arrays in ` *arrays`
    """
    if x_range[0] == 'beginning':
        min_x = 0
    else:
        min_x = X[0,:].searchsorted(x_range[0])
    if x_range[1] == 'end':
        max_x = -1
    else:
        max_x = X[0,:].searchsorted(x_range[1])
    out = [X[:,min_x:max_x]]
    for arr in arrays:
        out.append(arr[:,min_x:max_x])
    return out

def calc_bottom_shear_stress(U, V, H, B,
                             von_karman=0.41, nu=1.36e-6, D_50=2e-4,
                             rho_0=1027, rho_s=2650, g=9.81):
    """
    calculate the bottom shear stress and critical bottom shear stress
    using formulations from Soulsby, 1997, Dynamics of Marine Sands, p104

    return `tau`, bottom shear stress values, of same type as input args
    and `tau_crit` single value for critical bottom shear stress
    """
    ## calculate bottom shear stress
    z_0 = D_50 / 12
    h = H - B
    C_D = ((1/von_karman) * (np.log(h/z_0) + (z_0/h) - 1))**(-2)
    tau = rho_0 * C_D * (U**2 + V**2)
    ## calculate critical bottom shear stress for fluid/sediment characteristics
    G = rho_s / rho_0
    D_str = (g*(G-1)/nu**2)**(1/3) * D_50
    shields_crit = 0.3/(1+1.2*D_str) + 0.055*(1-np.exp(-0.02*D_str))
    tau_crit = shields_crit * g * D_50 * (rho_s-rho_0)
    return tau, tau_crit

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
            ## create velo_mag attribute for velocity magnitude
            # try:
            #     u = getattr(self, 'uvelo')
            #     v = getattr(self, 'vvelo')
            #     setattr(self, 'velo_mag', np.sqrt(u**2 + v**2))
            #     setattr(self, 'kinetic_energy', )
            # except AttributeError as e:
            #     print(e)
            #     raise warnings.warning(
            #         "unable to calculate velo_mag: couldn't find uvelo and/or vvelo")
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
        load csv data
        """
        return np.loadtxt(csv_file_path, skiprows=1)

    def get_free_surface_slice(self, y_ind, x_range, height='height'):
        """
        return the x values and water height along a transect
        """
        X, H = subsample_arrays_in_x(x_range, self.x, getattr(self, height))
        return X[y_ind,:], H[y_ind,:]

    def plot_free_surface_slice(self, y_ind="mid", x_range=("beginning", "end"),
                                figsize=(12,8), title=None):
        """
        plot a slice of the free surface through y_ind
        """
        ## find midpoint idx
        if y_ind is "mid":
            y_ind = np.floor(self.y.shape[0]/2)
        ## get bathy profile
        B = getattr(self, "bathymetry")
        H = getattr(self, "height")
        X, B, H = subsample_arrays_in_x(x_range, self.x, B, H)
        fig = plt.figure(figsize=figsize)
        plt.plot(X[y_ind,:], B[y_ind,:], 'k')
        plt.plot(X[y_ind,:], H[y_ind,:], 'b')
        plt.xlabel('x [m]')
        plt.ylabel('h [m]')
        if title is None:
            plt.title(self.csv_file_path)
        else:
            plt.title(title)
        return fig

    def get_wavelength_spectrum(self, height="height"):
        """
        return the wavelength spectrum and frequencies
        """
        ## get spacing for interpolation
        x = self.x[0,:]
        dx = np.min(np.diff(x))
        xi = np.arange(np.min(x), np.max(x), dx)
        n = len(xi)
        ## get frequencies
        freq = scipy.fftpack.rfftfreq(n, d=dx)
        ## interpolate height to regular grid
        H = getattr(self, height)
        m = H.shape[0]
        Hi = np.zeros((m,n))
        ## apply interpolation for each row
        for i in range(m):
            f = scipy.interpolate.interp1d(x, H[i,:])
            Hi[i,:] = f(xi)
        ## compute spectrum along each crossshore transect
        S_all = scipy.fftpack.rfft(Hi, axis=1)
        ## sum into one spectrum
        S = np.mean(S_all, axis=0)
        return S, freq

    def plot_wavelength_spectrum(self, figsize=(12,9), xlims=None, ylims=None,
                                 return_line=False, logscale=False):
        """
        plot wavelength spectrum
        """
        S, freq = self.get_wavelength_spectrum()
        ## new figure and axis
        fig = plt.figure(figsize=figsize)
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.xlabel('Wavenumber [1/m]')
        plt.ylabel('||')
        if logscale:
            p = plt.semilogy(freq, S)[0]
        else:
            p = plt.plot(freq, S)[0]
        if return_line:
            return fig, p
        else:
            return fig

    def plot_velocity(self, figsize=(12,7), uvelo='uvelo', vvelo='vvelo'):
        """
        quiver plot of velocities
        """
        U = getattr(self, uvelo)
        V = getattr(self, vvelo)
        fig = plt.figure(figsize=figsize)
        plt.quiver(self.x, self.y, U, V)
        return fig

    def plot_height(self, figsize=(12,7), height='height', title=None):
        """
        pcolormesh plot of height
        """
        H = getattr(self, height)
        fig = plt.figure(figsize=figsize)
        plt.pcolormesh(self.x, self.y, H, cmap='viridis')
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        if title is not None:
            plt.title(title)
        cb = plt.colorbar()
        cb.set_label("h [m]")
        return fig

    def plot_depth(self, figsize=(12,7), height='height', bathy='bathymetry'):
        """
        pcolormesh plot of water depth
        """
        H = getattr(self, height)
        B = getattr(self, bathy)
        fig = plt.figure(figsize=figsize)
        plt.pcolormesh(self.x, self.y, H-B, cmap='viridis')
        plt.colorbar()
        return fig

    def plot_depth_velocity(self, figsize=(12,7), height='height', plot_every=1,
                                uvelo='uvelo', vvelo='vvelo', bathy="bathymetry", cmap="viridis",
                                x_range=None, boolean_depth=None, ax_instance=None, clims=None):
        """
        plot the depth
        overlay velocity vectors
        """
        U = getattr(self, uvelo)
        V = getattr(self, vvelo)
        H = getattr(self, height)
        B = getattr(self, bathy)
        X = self.x
        Y = self.y
        depth = H - B
        if x_range is not None:
            X, Y, U, V, depth = subsample_arrays_in_x(x_range, X, Y, U, V, depth)
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
            P = ax.pcolormesh(X, Y, depth, cmap=cmap, vmin=clims[0], vmax=clims[1])
        else:
            P = ax.pcolormesh(X, Y, depth, cmap=cmap)
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
                                x_range=None, stillwater=None, ax_instance=None, clims=None):
        """
        plot the deviation in height from still water
        overlay velocity vectors
        """
        U = getattr(self, uvelo)
        V = getattr(self, vvelo)
        H = getattr(self, height)
        B = getattr(self, bathy)
        X, Y = self.x, self.y
        if x_range is not None:
            X, Y, U, V, H, B = subsample_arrays_in_x(x_range, X, Y, U, V, H, B)
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
            P = ax.pcolormesh(X, Y, devH, cmap=cmap, vmin=clims[0], vmax=clims[1])
        else:
            P = ax.pcolormesh(X, Y, devH, cmap=cmap)
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
        X, Y = self.x, self.y
        if x_range is not None:
            X, Y, U, V, B = subsample_arrays_in_x(x_range, X, Y, U, V, B)
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
            return fig
        else:
            ax = ax_instance
            p = ax.streamplot(Xi[0,:], Yi[:,0], Ui, Vi, norm=norm,
                arrowsize=arrowsize, color=speedi, cmap=cmap, density=density)
            return p

    def plot_shear_stress(self, figsize=(12,7), cmap='plasma', x_range=None,
                          uvelo='uvelo', vvelo='vvelo', height='height',
                          bathy='bathymetry'):
        """
        pcolormesh plot of bed shear stress
        """
        U = getattr(self, uvelo)
        V = getattr(self, vvelo)
        H = getattr(self, height)
        B = getattr(self, bathy)
        X, Y = self.x, self.y
        if x_range is not None:
            X, Y, U, V, H, B = subsample_arrays_in_x(x_range, X, Y, U, V, H, B)
        ## calculations
        tau, tau_crit = calc_bottom_shear_stress(U, V, H, B)
        ## NaNs occur in C_D when dividing by zero height
        tau[np.isnan(tau)] = 0
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)
        ## add outline of bathymetry
        obstacle_outlines = ObstacleOutlines(X, Y, B)
        ax = add_topo_contours(ax, obstacle_outlines)
        pcm = ax.pcolormesh(X, Y, tau, cmap=cmap, vmin=tau_crit, norm=LogNorm())
        cbar = plt.colorbar(pcm)
        cbar.set_label('Shear Stress [N/m^2]')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        plt.title(self.csv_file_path)
        return fig

    def plot_kinetic_energy(self, figsize=(12,7), cmap='plasma', x_range=None,
                            uvelo='uvelo', vvelo='vvelo', height='height',
                            bathy='bathymetry'):
        """
        pcolormesh plot of kinetic energy
        """
        U = getattr(self, uvelo)
        V = getattr(self, vvelo)
        H = getattr(self, height)
        B = getattr(self, bathy)
        X, Y = self.x, self.y
        if x_range is not None:
            X, Y, U, V, H, B = subsample_arrays_in_x(x_range, X, Y, U, V, H, B)
        Ek = 0.25 * (H - B) * (U**2 + V**2)
        fig = plt.figure(figsize=figsize)
        plt.pcolormesh(X, Y, Ek, cmap=cmap)
        cbar = plt.colorbar()
        cbar.set_label('Kinetic Energy [m^2/s^2]')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        return fig

    def plot_energy(self, figsize=(12,7), cmap='viridis', x_range=None,
                    height='height', bathy='bathymetry',
                    uvelo='uvelo', vvelo='vvelo'):
        """
        pcolormesh plot of total energy
        """
        U = getattr(self, uvelo)
        V = getattr(self, vvelo)
        H = getattr(self, height)
        B = getattr(self, bathy)
        X, Y = self.x, self.y
        if x_range is not None:
            X, Y, U, V, H, B = subsample_arrays_in_x(x_range, X, Y, U, V, H, B)
        g = 9.81
        Ek = 0.25 * (H - B) * (U**2 + V**2)
        Ep = 0.5 * g * (H-B)**2
        E = Ek + Ep
        fig = plt.figure(figsize=figsize)
        plt.pcolormesh(X, Y, E, cmap=cmap)
        cbar = plt.colorbar()
        cbar.set_label('Energy [m^2/s^2]')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        return fig

    def plot_bathy(self, figsize=(14,7), bathy='bathymetry', 
                   x_range=None, cmap='viridis'):
        """
        plot bathymetry as pcolormesh
        """
        B = getattr(self, bathy)
        X = self.x
        Y = self.y
        if x_range is not None:
            X, Y, B = subsample_arrays_in_x(x_range, X, Y, B)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.pcolormesh(X, Y, B, cmap=cmap)
        return fig

    def plot_bathy_3D(self, figsize=(14,7), x_range=None,
                     bathy='bathymetry', cmap='viridis', stride=1):
        """
        plot bathymetry as 3D surface
        """
        B = getattr(self, bathy)        
        X, Y = self.x, self.y
        if x_range is not None:
            X, Y, B = subsample_arrays_in_x(x_range, X, Y, B)
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
                       cmap='viridis', x_range=None, timestamp=None, stride=1):
        """
        plot height as 3D surface and color by depth
        (optionally) bathymetry as 3D surfaces
        """
        H = getattr(self, height)
        B = getattr(self, bathy)        
        X, Y = self.x, self.y
        if x_range is not None:
            X, Y, H, B = subsample_arrays_in_x(x_range, X, Y, H, B)
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
        norm = plt.Normalize()
        norm_colors = plt.cm.get_cmap(cmap)(norm(norm_depth))
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
        p = ax.plot_surface(X, Y, H, cmap=colormap, facecolors=norm_colors,
                            rstride=stride, cstride=stride, vmin=vmin, vmax=vmax)
        if return_fig:
            # cb = fig.colorbar(p, shrink=.7)
            # cb.set_label('Depth [m]')
            return fig
        ## for use in animation function
        else:
            return p

    def __repr__(self):
        return "NumaCsvData({})".format(self.csv_file_path)

class UnstNumaCsvData(NumaCsvData):
    """
    sub class of numa_plotting_tools.NumaCsvData
      modifies __init__ to interpolate data on to a regular grid

      `delta_x` sets interpolation grid spacing in x in the low res area
        offshore and far onshore
      `delta_y` sets interpolation grid spacing in y everywhere and in x in the
        high res area at the shore
    """
    def __init__(self, csv_file_path, xcoord_header='xcoord',
                 ycoord_header='ycoord', high_res_range=[-500, 500],
                 delta_x=50., delta_y=5.):
        ## load data
        self.csv_file_path = csv_file_path
        self.headers = self._get_headers(csv_file_path)
        self.data = self._load_csv_data(csv_file_path)
        for j, att in enumerate(self.headers):
            setattr(self, att, self.data[:,j])

        ## reshape inferring shape from data
        if xcoord_header in self.headers and ycoord_header in self.headers:
            self.headers.remove(xcoord_header)
            self.headers.remove(ycoord_header)
            x_unst = getattr(self, xcoord_header)
            y_unst = getattr(self, ycoord_header)
            x_st = np.hstack((
                np.arange(np.min(x_unst), high_res_range[0], delta_x),
                np.arange(high_res_range[0], high_res_range[1], delta_y),
                np.arange(high_res_range[1], np.max(x_unst), delta_x)
            ))
            y_st = np.arange(np.min(y_unst), np.max(y_unst), delta_y)
            self.nelx = len(x_st)
            self.nely = len(y_st)
            self.x, self.y = np.meshgrid(x_st, y_st)
            ## for each attribute interpolate onto grid
            for att in self.headers:
                temp = getattr(self, att)
                temp = scipy.interpolate.griddata(
                    (x_unst, y_unst),
                    temp,
                    (self.x, self.y)
                )
                setattr(self, att, temp)
            ## create velo_mag attribute for velocity magnitude
            # try:
            #     u = getattr(self, 'uvelo')
            #     v = getattr(self, 'vvelo')
            #     setattr(self, 'velo_mag', np.sqrt(u**2 + v**2))
            # except AttributeError as e:
            #     print(e)
            #     raise warnings.warning(
            #         "unable to calculate velo_mag: couldn't find uvelo and/or vvelo")
        else:
            ## can't detect x and/or y coord data
            raise warnings.warning(
                'NO X and/or Y data found using headers {} and {} \n{}'.format(
                    xcoord_header,
                    ycoord_header,
                    csv_file_path
                )
            )

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
            unstructured=False,
            suffix='',
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
            csv_file_names = ['{}_{}_{:04d}.csv'.format(
                csv_file_root, suffix, i) for i in range(self.n_outputs)]
            self.data_obj_list = self._load_numa_csv_data(csv_file_names,
                                                          unstructured=unstructured)
        self.t, \
        self.shore_max, \
        self.shore_min, \
        self.shore_mean, \
        self.shore_std = self._load_shore_data(shore_file_name)

    def _load_numa_csv_data(self, csv_file_names, unstructured=False):
        """
        load data from model output csvs
        """
        data_obj_list = []
        for csv_file_name in csv_file_names:
            csv_file_path = os.path.join(self.run_dir_path, csv_file_name)
            if unstructured:
                ncd = UnstNumaCsvData(csv_file_path)
            else:
                ncd = NumaCsvData(csv_file_path)
            data_obj_list.append(ncd)
        return data_obj_list

    def _load_shore_data(self, shore_file_name):
        """
        load shore max timeseries from model run shore line max output data file
        """
        t = []
        shore_max = []
        shore_min = []
        shore_mean = []
        shore_std = []
        shore_file_path = os.path.join(self.run_dir_path, shore_file_name)
        with open(shore_file_path, 'r') as file:
            for i, r in enumerate(file):
                r = r.strip().lstrip().split()
                ## bug workaround, some files were written with the mean value
                ## wrapping onto a second row... in that case we only get the max
                try:
                    shore_max.append(float(r[1]))
                    t.append(float(r[0]))
                except IndexError:
                    continue
                try:
                    shore_mean.append(float(r[3]))
                    shore_min.append(float(r[2]))
                    try:
                        shore_std.append(np.sqrt(float(r[4])))
                    except IndexError:
                        shore_std.append(np.nan)
                except IndexError:
                    pass
                ## sometimes appends '*******' == NaN?
                except ValueError:
                    shore_mean.append(np.nan)
                    shore_min.append(np.nan)

        return (np.asarray(t), np.asarray(shore_max), np.asarray(shore_min),
                np.asarray(shore_mean), np.asarray(shore_std))

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

    def get_initial_condition_total_potential_energy(self, x_range=["beginning", 0],
                                                     height="height", bathy="bathymetry"):
        """
        return the sum of the potential energy values at each offshore grid point
        """
        ncd0 = self.data_obj_list[0]
        H = getattr(ncd0, height)
        B = getattr(ncd0, bathy)
        X = ncd0.x
        depth = H - B
        subsample_arrays_in_x(x_range, X, depth)
        Ep = potential_energy(depth)
        return np.mean(Ep)

    def animate_free_surface_slice(self, x_range=("beginning", "end"),
                                   save_file_path=None,
                                   t_range=None, y_ind="mid",
                                   figsize=(12,8), interval=300):
        """
        create an animation of the evolution of the wavelength spectrum through time
        """
        if save_file_path is None:
            save_file_path = self.run_dir_path + '_free_surface_slice.mp4'
        obs_to_plot = self.data_obj_list
        ## t_range is a tuple containing 2 indices into a self.data_obj_list
        if t_range is not None:
            start, end = t_range
            if start <= end and start >= 0:
                obs_to_plot = obs_to_plot[start:end]
        ob0 = obs_to_plot[0]
        ## find midpoint idx
        if y_ind is "mid":
            y_ind = np.floor(ob0.y.shape[0]/2)
        ## get bathy profile
        B = getattr(ob0, "bathymetry")
        max_B = np.max(B[y_ind,:])
        X, B = subsample_arrays_in_x(x_range, ob0.x, B)
        b = B[y_ind,:]
        ## get initial data
        x, h = ob0.get_free_surface_slice(y_ind, x_range)
        fig = plt.figure(figsize=figsize)
        plt.plot(x, b, 'k')
        plt.plot(x, b, 'kx')
        line = plt.plot(x, h)[0]
        plt.xlabel('x [m]')
        plt.ylabel('h [m]')
        plt.ylim(top=max_B)
        plt.title(self.name)

        def animate(ob):
            x, h = ob.get_free_surface_slice(y_ind, x_range)
            line.set_ydata(h)
            return line,

        def init():
            line.set_ydata(np.ma.array(x, mask=True))
            return line,
        ## do animation
        ani = mpl.animation.FuncAnimation(fig, animate, obs_to_plot, init_func=init,
                                          blit=True, interval=interval)
        ani.save(save_file_path, dpi=300)


    def animate_wavelength_spectrum(self, save_file_path=None, interval=300,
                                    t_range=None, figsize=(12,9), logscale=False,
                                    xlims=None, ylims=None):
        """
        create an animation of the evolution of the wavelength spectrum through time
        """
        if save_file_path is None:
            save_file_path = self.run_dir_path + '_wavelength_spectrum.mp4'
        obs_to_plot = self.data_obj_list
        ## t_range is a tuple containing 2 indices into a self.data_obj_list
        if t_range is not None:
            start, end = t_range
            if start <= end and start >= 0:
                obs_to_plot = obs_to_plot[start:end]
        ob0 = obs_to_plot[0]
        _, freq = ob0.get_wavelength_spectrum()
        fig, line = ob0.plot_wavelength_spectrum(xlims=xlims, ylims=ylims,
                                                 figsize=figsize,
                                                 return_line=True, logscale=logscale)
        def animate(ob):
            S, freq = ob.get_wavelength_spectrum()
            line.set_ydata(S)
            return line,

        def init():
            line.set_ydata(np.ma.array(freq, mask=True))
            return line,

        ## do animation
        ani = mpl.animation.FuncAnimation(fig, animate, obs_to_plot, init_func=init,
                                          blit=True, interval=interval)
        ani.save(save_file_path, dpi=300)

    def animate_velocity_streamlines(self, save_file_path=None, figsize=(14,7), uvelo="uvelo",
                          vvelo="vvelo", cmap='plasma', bathy='bathymetry', arrowsize=1,
                          interval=200, x_range=None, density=1, t_range=None, clims=None):
        """
        create an animation of NumaCsvData.plot_velocity_streamlines
        """
        if save_file_path is None:
            save_file_path = self.run_dir_path + '_velocity_streamlines.mp4'
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
        X, Y = ob0.x, ob0.y
        ## set up for interpolated grid and obstacle outlines
        if x_range is not None:
            X, Y, B = subsample_arrays_in_x(x_range, X, Y, B)
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
        ## in case you have no velocity streamlines
        try:
            cb = fig.colorbar(sc.lines, extend='max')
            cb.set_label("Velocity [m/s^2]")
        except TypeError:
            pass

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
                          interval=200, x_range=None, plot_every=1, t_range=None, clims=None):
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
        X, Y = ob0.x, ob0.y
        if x_range is not None:
            X, Y, H, B = subsample_arrays_in_x(x_range, X, Y, H, B)
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
        P, Q = ob0.plot_devheight_velocity(ax_instance=ax, x_range=x_range, plot_every=plot_every,
            stillwater=stillwater, clims=clims, cmap=cmap,
            height=height, bathy=bathy, uvelo=uvelo, vvelo=vvelo)
        cb = fig.colorbar(P, extend='both')
        cb.set_label("Height above stillwater [m]")
        # create list of drawables to pass to animation
        list_plots = []
        for i, ob in enumerate(obs_to_plot):
            list_plots.append(ob.plot_devheight_velocity(
                ax_instance=ax,
                x_range=x_range,
                plot_every=plot_every,
                stillwater=stillwater,
                clims=clims,
                cmap=cmap,
                height=height,
                bathy=bathy,
                uvelo=uvelo,
                vvelo=vvelo
            ))

        ## do animation
        ani = mpl.animation.ArtistAnimation(fig, list_plots, interval=interval)
        ani.save(save_file_path, dpi=300)

    def animate_height_3D(self, save_file_path=None, figsize=(14,7), stride=1,
                          height='height', cmap='viridis', bathy='bathymetry',
                          interval=100, x_range=None, t_range=None, regex_string=None):
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
        X, Y = ob0.x, ob0.y
        if x_range is not None:
            X, Y, H, B = subsample_arrays_in_x(x_range, X, Y, H, B)
        depth = H - B
        clims = (0, np.max(depth))
        fig = ob0.plot_bathy_3D(figsize=figsize, bathy=bathy, 
                                cmap=None, x_range=x_range, stride=stride)
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
                x_range=x_range,
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
            E = 0.25 * (H - B) * (U**2 + V**2)
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

    def plot_energy_distance_time(self, figsize=(12,7), x_range=None,
                                  plot_type='Mean', cmap='plasma', cnorm='log',
                                      uvelo='uvelo', vvelo='vvelo',
                                      height='height', bathy='bathymetry'):
        """
        x axis x-location y axis time, pcolormesh of mean along each x location
        of magnitude of kinetic energy
        """
        time = np.linspace(0, self.t_f, self.n_outputs)
        X = self.data_obj_list[0].x
        if x_range is not None:
            x = subsample_arrays_in_x(x_range, X)[0][0,:]
        else:
            x = X[0,:]
        energy = np.zeros((time.shape[0], x.shape[0]), float) * np.nan
        for i, ob in enumerate(self.data_obj_list):
            U = getattr(ob, uvelo)
            V = getattr(ob, vvelo)
            H = getattr(ob, height)
            B = getattr(ob, bathy)
            if x_range is not None:
                U, V, H, B = subsample_arrays_in_x(x_range, X, U, V, H, B)[1:]
            E = 0.25 * (H - B) * (U**2 + V**2)
            if plot_type == 'Mean':
                energy[i,:] = np.mean(E, axis=0)
            elif plot_type == 'Max':
                energy[i,:] = np.nanmax(E, axis=0)
        fig = plt.figure(figsize=figsize)
        if cnorm == 'log':
            plt.pcolormesh(x, time, energy[1:,1:], cmap=cmap, norm=LogNorm())
        else:
            plt.pcolormesh(x, time, energy[1:,1:], cmap=cmap)
        cbar = plt.colorbar()
        cbar.set_label("{} kinetic energy [m^3/s^2]".format(plot_type))
        plt.ylabel('time [s]')
        plt.xlabel('x location [m]')
        return fig

    def plot_speed_distance_time(self, figsize=(12,7), x_range=None,
                                  plot_type='Mean', cmap='plasma', cnorm='log',
                                      uvelo='uvelo', vvelo='vvelo',):
        """
        x axis x-location y axis time, pcolormesh of mean along each x location
        of magnitude of kinetic energy
        """
        time = np.linspace(0, self.t_f, self.n_outputs)
        X = self.data_obj_list[0].x
        if x_range is not None:
            x = subsample_arrays_in_x(x_range, X)[0][0,:]
        else:
            x = X[0,:]
        speed = np.zeros((time.shape[0], x.shape[0]), float) * np.nan
        for i, ob in enumerate(self.data_obj_list):
            U = getattr(ob, uvelo)
            V = getattr(ob, vvelo)
            if x_range is not None:
                U, V = subsample_arrays_in_x(x_range, X, U, V)[1:]
            S = np.sqrt(U**2 + V**2)
            if plot_type == 'Mean':
                speed[i,:] = np.mean(S, axis=0)
            elif plot_type == 'Max':
                speed[i,:] = np.max(S, axis=0)
        fig = plt.figure(figsize=figsize)
        if cnorm == 'log':
            plt.pcolormesh(x, time, speed[1:,1:], cmap=cmap, norm=LogNorm())
        else:
            plt.pcolormesh(x, time, speed[1:,1:], cmap=cmap)
        cbar = plt.colorbar()
        cbar.set_label("{} velocity magnitude [m/s]".format(plot_type))
        plt.ylabel('time [s]')
        plt.xlabel('x location [m]')
        return fig

    def get_shear_stress(self, x_coord, y_coord='avg', return_type='max',
                            x_width=None, filter_outliers=None,
                         von_karman=0.41, rho_0=1000, z_0=2e-5,
                            uvelo='uvelo', vvelo='vvelo',
                            height='height', bathy='bathymetry'):
        """
        return the shear stress at (x_coord, y_coord)
        (if y_coord is average, average along shore)
        if return_max, return the maximum instantaneous value
        else return the time average value
        """
        tau = []
        min_tau = []
        max_tau = []
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
                T, T_critical = calc_bottom_shear_stress(U, V, H, B)
                ## remove data more than filter_outliers standard deviations from mean
                if filter_outliers is not None:
                    T = remove_outliers(T, filter_outliers)
                if return_type[:11] == "timeseries_":
                    min_T = np.nanmin(T)
                    max_T = np.nanmax(T)
                T = np.nanmean(T)
            ## single y_coord at midpoint of domain
            else:
                if y_coord == "mid":
                    ind_y = int(np.ceil(ob.x.shape[0]/2.))
                else:
                    ind_y = ob.y[:,0].searchsorted(y_coord)
                U = U[ind_y,ind]
                V = V[ind_y,ind]
                H = H[ind_y,ind]
                B = B[ind_y,ind]
                T, T_critical = calc_bottom_shear_stress(U, V, H, B)

            tau.append(T)
            if return_type[:11] == "timeseries_" and y_coord == 'avg':
                min_tau.append(min_T)
                max_tau.append(max_T)

        ## instantaneous max
        if return_type == 'max':
            return np.nanmax(tau)
        ## return the time average value
        elif return_type == 'mean':
            return np.mean(tau)
        elif return_type == 'timeseries':
            return np.asarray(tau)
        elif return_type[:11] == "timeseries_":
            return np.asarray((tau, min_tau, max_tau))

    def get_kinetic_energy(self, x_coord, y_coord='avg', return_type='max',
                            x_width=None, filter_outliers=None,
                            uvelo='uvelo', vvelo='vvelo',
                            height='height', bathy='bathymetry'):
        """
        return the kinetic energy at (x_coord, y_coord)
        (if y_coord is average, average along shore)
        if return_type == max, return the maximum instantaneous value 
        else return the time average value  
        also can return timeseries of max or mean values
        """
        energy = []
        min_energy = []
        max_energy = []
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
                E = kinetic_energy(H-B, U, V)
                ## remove data more than filter_outliers standard deviations from mean
                if filter_outliers is not None:
                    E = remove_outliers(E, filter_outliers)
                ## record min and max as well as mean
                if return_type[:11] == "timeseries_":
                    min_E = np.nanmin(E)
                    max_E = np.nanmax(E)
                ## calc the total kinetic energy
                if return_type == 'sum_timeseries':
                    E = np.sum(E)
                ## calc the mean kinetic energy
                else:
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
                E = kinetic_energy(H-B, U, V)

            energy.append(E)
            ## record min and max as well as mean
            if return_type[:11] == "timeseries_" and y_coord == 'avg':
                min_energy.append(min_E)
                max_energy.append(max_E)

        ## instantaneous max
        if return_type == 'max':
            return np.nanmax(energy)
        ## return the time average value
        elif return_type == 'mean':
            return np.mean(energy)
        elif return_type == 'timeseries' or return_type == 'sum_timeseries':
            return np.asarray(energy)
        elif return_type[:11] == "timeseries_":
            return np.asarray((energy, min_energy, max_energy))

    def get_velocity(self, x_coord, y_coord='avg', return_type='max',
                            x_width=None, filter_outliers=None,
                            uvelo='uvelo', vvelo='vvelo'):
        """
        return the velocity at (x_coord, y_coord) 
        (if y_coord is average, average along shore)
        if return_max, return the maximum instantaneous value 
        else return the time average value  
        """
        velocity = []
        min_velocity = []
        max_velocity = []
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
                if return_type[:11] == "timeseries_":
                    min_v = np.nanmin(vel)
                    max_v = np.nanmax(vel)
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
            if return_type[:11] == "timeseries_" and y_coord == 'avg':
                min_velocity.append(min_v)
                max_velocity.append(max_v)
        ## instantaneous max
        if return_type == 'max':
            return np.nanmax(velocity)
        ## return the time average value
        elif return_type == 'mean':
            return np.mean(velocity)
        elif return_type == 'timeseries':
            return np.asarray(velocity)
        elif return_type[:11] == "timeseries_":
            return np.asarray((velocity, min_velocity, max_velocity))

    def get_height(self, x_coord, y_coord='avg', return_type='max',
                             x_width=None, filter_outliers=None,
                           height='height', bathy='bathymetry'):
        """
        return the shear stress at (x_coord, y_coord)
        (if y_coord is average, average along shore)
        if return_max, return the maximum instantaneous value 
        else return the time average value  
        """
        hgt = []
        min_hgt = []
        max_hgt = []
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
                if return_type[:11] == "timeseries_":
                    min_h = np.nanmin(h)
                    max_h = np.nanmax(h)
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
            if return_type[:11] == "timeseries_" and y_coord == 'avg':
                min_hgt.append(min_h)
                max_hgt.append(max_h)
        ## instantaneous max
        if return_type == 'max':
            return np.nanmax(hgt)
        ## return the time average value
        elif return_type == 'mean':
            return np.mean(hgt)
        elif return_type == 'timeseries':
            return np.asarray(hgt)
        elif return_type[:11] == "timeseries_":
            return np.asarray((hgt, min_hgt, max_hgt))

    def get_spatial_kinetic_energy(self, x_coord, x_width=None, n_pts=10,
                            uvelo='uvelo', vvelo='vvelo',
                            height='height', bathy='bathymetry'):
        """
        get the max Ek values and their spatial coordinates
        in the box defined by `x_coord`, `y_coord`, `x_width`
        return values from the top `n_pts` timesteps
        """
        energy = np.zeros((n_pts,1), dtype=float)
        x_out = np.zeros_like(energy)
        y_out = np.zeros_like(energy)
        min_E_old = 0.
        for ob in self.data_obj_list:
            ind = ob.x[0,:].searchsorted(x_coord)
            U = getattr(ob, uvelo)
            V = getattr(ob, vvelo)
            H = getattr(ob, height)
            B = getattr(ob, bathy)
            ## to do average over area rather than line
            if x_width is not None:
                ind2 = ob.x[0,:].searchsorted(x_coord+x_width)
            else:
                ind2 = ind + 1
            X = ob.x[:,ind:ind2].flatten()
            Y = ob.y[:,ind:ind2].flatten()
            U = U[:,ind:ind2]
            V = V[:,ind:ind2]
            H = H[:,ind:ind2]
            B = B[:,ind:ind2]
            ## compute Ek
            E = kinetic_energy(H-B, U, V)
            max_E_new = np.max(E)
            if max_E_new > min_E_old:
                ## find location to insert new value
                idx_out = np.argmin(energy)
                ## find location of max value in flattened array
                idx_xy = np.argmax(E)
                ## place data in output array
                energy[idx_out] = max_E_new
                x_out[idx_out] = X[idx_xy]
                y_out[idx_out] = Y[idx_xy]
                min_E_old = np.min(energy)
        return energy, x_out, y_out

    def get_spatial_velocity(self, x_coord, x_width=None, n_pts=10,
                            uvelo='uvelo', vvelo='vvelo'):
        """
        get the max velocity values and their spatial coordinates
        in the box defined by `x_coord`, `y_coord`, `x_width`
        return values from the top `n_pts` timesteps
        """
        velocity = np.zeros((n_pts,1), dtype=float)
        x_out = np.zeros_like(velocity)
        y_out = np.zeros_like(velocity)
        min_V_old = 0.
        for ob in self.data_obj_list:
            ind = ob.x[0,:].searchsorted(x_coord)
            U = getattr(ob, uvelo)
            V = getattr(ob, vvelo)
            ## to do average over area rather than line
            if x_width is not None:
                ind2 = ob.x[0,:].searchsorted(x_coord+x_width)
            else:
                ind2 = ind + 1
            X = ob.x[:,ind:ind2].flatten()
            Y = ob.y[:,ind:ind2].flatten()
            U = U[:,ind:ind2]
            V = V[:,ind:ind2]
            ## compute velocity
            V = np.sqrt(U**2+V**2)
            max_V_new = np.max(V)
            if max_V_new > min_V_old:
                ## find location to insert new value
                idx_out = np.argmin(velocity)
                ## find location of max value in flattened array
                idx_xy = np.argmax(V)
                ## place data in output array
                velocity[idx_out] = max_V_new
                x_out[idx_out] = X[idx_xy]
                y_out[idx_out] = Y[idx_xy]
                min_V_old = np.min(velocity)
        return velocity, x_out, y_out

    def subsample_shore_position_timeseries(self, n_samples,
                                            t_min=0, att="shore_max"):
        """
        return n_samples evenly spaced from shore position timeseries data
        """
        pos = getattr(self, att)
        n = len(pos)
        idx_min = self.t.searchsorted(t_min)
        interval = np.ceil(n / n_samples)
        return pos[idx_min:-1:interval]

    def plot_shore_max_timeseries(
            self,
            figsize=(12,7),
            figure_instance=None,
            color='k',
            marker='.',
            ls="None",
            zorder=2,
            regex_string=None
    ):
        if figure_instance is None:
            figure_instance = plt.figure(figsize=figsize)
        try:
            p = plt.plot(self.shore_max, self.t, c=color, marker=marker,
                         zorder=zorder, figure=figure_instance, ls=ls, mew=0)[0]
        ## big ugly hack to get around https://github.com/scikit-learn/scikit-learn/issues/5040
        except ValueError:
            p = plt.scatter(self.shore_max, self.t, c=color, zorder=zorder)
        #     filtr = np.logical_not(np.isnan(self.shore_max))
        #     shore_max = self.shore_max[filtr]
        #     t = self.t[filtr]
        #     try:
        #         fig_temp = plt.figure("temp")
        #         plt.plot(shore_max, t)
        #         plt.close("temp")
        #     except ValueError:
        #         pass
        #     ## set current figure back to the one you want
        #     plt.figure(figure_instance.number)
        #     p = plt.plot(shore_max, t, c=color, marker=marker,
        #                  ls="None", mew=0)[0]

        p.set_label(self.legend_label(regex_string=regex_string))
        return figure_instance


    def __repr__(self):
        return "NumaRunData({}, {}, {})".format(
            self.t_f,
            self.t_restart,
            self.run_dir_path
        )

def load_nrds(run_dirs, from_pickle=1, regex_string='h-\d+'):
    """
    function to be used in data processing script to load in a set of NumaRunData objects
    can load the data from .pkl files (DEFAULT)
    OR  can load the data from a directory of .csv files
        - this option SAVES a .pkl for faster future loads
    returns a list of NumaRunData objects
    """
    nrds = []
    ## initialize objects from csv files
    if not from_pickle:
        ## load all data into list of NumaRunData objects
        for rd in run_dirs:
            try:
                nrd = NumaRunData(t_f, t_restart, rd,
                                      regex_string=regex_string,
                                      load_csv_data=True, unstructured=False)
                ## save to pickle file
                saveobj(nrd, rd)
                nrds.append(nrd)
            except FileNotFoundError as e:
                print(e)
    ## load object list from pickle file
    else:
        for rd in run_dirs:
            try:
                nrds.append(openobj(rd))
            except FileNotFoundError as e:
                print(e)
    return nrds
# commit 08f7904ca37416113d7ab43ffc36be7722d70470