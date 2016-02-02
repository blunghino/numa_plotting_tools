import numpy as np
import scipy.interpolate
from matplotlib import pyplot as plt


def interp_to_mesh(fine_x, fine_y, nrd, att='height'):
	"""
	interpolate vals at grid defined by x and y onto grid
	defined by fine_x and fine_y
	"""
	shapef = fine_x.shape
	fine_x = fine_x.flatten()
	fine_y = fine_y.flatten()
	zi_list = []
	for ob in nrd.data_obj_list:
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

def plot_error(nrd_f, nrd_c, att='height'):
	"""
	plot the error at each timestep for two NumaRunData objects
	"""
	xi = nrd_f.data_obj_list[0].x
	yi = nrd_f.data_obj_list[0].y
	zf = [getattr(ob, att) for ob in nrd_f.data_obj_list]
	zi = interp_to_mesh(xi, yi, nrd_c, att)
	err = calc_error(zf, zi)
	N = len(nrd_f.data_obj_list)
	t = np.linspace(0, N, N)
	fig = plt.figure()
	plt.plot(t, err)
	plt.xlabel("timestep")
	plt.ylabel("mean error")
	plt.title("{}: error in {}".format(nrd_c.name, att))
	return fig
