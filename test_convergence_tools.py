import numpy as np
import scipy.interpolate
from scipy.stats import linregress
from matplotlib import pyplot as plt


def interp_to_mesh(fine_x, fine_y, nrd, ind=None, att='height'):
    """
    interpolate vals at grid defined by x and y onto grid
    defined by fine_x and fine_y
    ie interpolate the coarse grid onto the fine grid points

    use ind to only interpolate one data point from nrd
    """
    shapef = fine_x.shape
    fine_x = fine_x.flatten()
    fine_y = fine_y.flatten()
    zi_list = []
    if ind is None:
        for ob in nrd.data_obj_list:
            z = getattr(ob, att)
            z = z.flatten()
            x = ob.x.flatten()
            y = ob.y.flatten()
            zi = scipy.interpolate.griddata((x,y), z, (fine_x, fine_y))
            zi_list.append(zi.reshape(shapef))
    else:
        ob = nrd.data_obj_list[ind]
        z = getattr(ob, att)
        z = z.flatten()
        x = ob.x.flatten()
        y = ob.y.flatten()
        zi = scipy.interpolate.griddata((x,y), z, (fine_x, fine_y))
        zi_list.append(zi.reshape(shapef))
    return zi_list

def calc_error(thr, act):
    """
    thr and act are lists of ndarrays
    return a list of len(thr) with the ith entry being the error for
    thr[i] and act[i]
    """
    err = []
    for t, a in zip(thr, act):
        err.append(np.nanmean(np.sqrt(a**2 - t**2)))
    return err

def calc_l2_error(thr, act):
    """
    thr and act are lists of ndarrays
    return a list of len(thr) with the ith entry being the error for
    thr[i] and act[i]
    """
    err = []
    for t, a in zip(thr, act):
        err.append(np.sqrt(np.nansum((t-a)**2)))
    return err

def calc_l1_error(thr, act):
    """
    thr and act are lists of ndarrays 
    return a list of len(thr) with the ith entry being the error for
    thr[i] and act[i]    
    """
    err = []
    for t, a in zip(thr, act):
        err.append(np.nansum(np.abs(t - a)))
    return err

def calc_linf_error(thr, act):
    """
    thr and act are lists of ndarrays
    return a list of len(thr) with the ith entry being the error for
    thr[i] and act[i]
    """
    err = []
    for t, a in zip(thr, act):
        err.append(np.nanmax(np.abs(t - a)))
    return err    

def plot_error(nrd_f, nrd_c, func=calc_l2_error, att='height'):
    """
    plot the error at each timestep for two NumaRunData objects
    """
    xi = nrd_f.data_obj_list[0].x
    yi = nrd_f.data_obj_list[0].y
    zf = [getattr(ob, att) for ob in nrd_f.data_obj_list]
    zi = interp_to_mesh(xi, yi, nrd_c, ind, att)
    err = func(zf, zi)
    N = len(nrd_f.data_obj_list)
    t = np.linspace(0, N, N)
    fig = plt.figure()
    plt.plot(t, err)
    plt.xlabel("timestep")
    plt.ylabel("error")
    plt.title("{}: error in {}".format(nrd_c.name, att))
    return fig

def calc_error_norms(nrds, ind=None, att="velo_mag"):
    """
    calc error with 3 different norms as a function of 
    h for a set of model runs
    the FINEST GRID is the LAST nrd in nrds
    """
    ## initialize output structures
    l1_err = np.zeros(len(nrds)-1)
    l2_err = l1_err.copy()
    linf_err = l1_err.copy()
    ## get fine grid and values
    nrd_f = nrds[-1]
    xi = nrd_f.data_obj_list[0].x
    yi = nrd_f.data_obj_list[0].y
    ob = nrd_f.data_obj_list[ind]
    zf = [getattr(ob, att)]
    zero = np.zeros_like(zf)
    l1_f = calc_l1_error(zf, zero)[0]
    l2_f = calc_l2_error(zf, zero)[0]
    li_f = calc_linf_error(zf, zero)[0]
    ## loop over all coarse nodes
    for j, nrd_c in enumerate(nrds[:-1]):
        ## calculate interpolated values
        zi = interp_to_mesh(xi, yi, nrd_c, ind, att)
        l1_err[j] = calc_l1_error(zf, zi)[0] / l1_f
        l2_err[j] = calc_l2_error(zf, zi)[0] / l2_f
        linf_err[j] = calc_linf_error(zf, zi)[0] / li_f
    return l1_err, l2_err, linf_err


def plot_error_norms_with_h(h_vals, nrds, ind=0, att="velo_mag", xlims=None, base=10):
    """
    plot error calculated with 3 different norms as a function of 
    h for a set of model runs
    the FINEST GRID is the LAST nrd in nrds
    """
    fig = plt.figure(figsize=(12,9))
    l1, l2, linf = calc_error_norms(nrds, ind, att)
    x = 1/h_vals[:-1]
    xlog = np.log10(x)
    m1, b1 = linregress(xlog, np.log10(l1))[:2]
    m2, b2 = linregress(xlog, np.log10(l2))[:2]
    mi, bi = linregress(xlog, np.log10(linf))[:2]
    x_ = np.linspace(xlog.min(), xlog.max(), 25)
    x_10 = 10**x_
    lstr = "m = {:.02f}"
    plt.loglog(x, l1, 'ks', label=r"$L_1$", basex=base, basey=base)
    plt.plot(x_10, 10**(m1*x_+b1), 'k--', label=lstr.format(m1))
    plt.loglog(x, l2, 'ro', label=r"$L_2$", basex=base, basey=base)
    plt.plot(x_10, 10**(m2*x_+b2), 'r--', label=lstr.format(m2))
    plt.loglog(x, linf, 'b<', label=r"$L_{\infty}$", basex=base, basey=base)
    plt.plot(x_10, 10**(mi*x_+bi), 'b--', label=lstr.format(mi))
    if xlims is not None:
        plt.xlim(xlims)
    plt.legend(numpoints=1)
    plt.xlabel("1/h")
    plt.ylabel("Error norm")
    return fig

