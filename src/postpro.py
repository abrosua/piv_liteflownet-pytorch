import numpy as np
from scipy.signal import convolve2d


def calc_vorticity(flow):
	# Extract each velocity component
	u = flow[:, :, 0]
	v = flow[:, :, 1]

	# Add boundaries
	u_temp = np.vstack([u[0, :], u, u[-1, :]])
	u = np.hstack([u_temp[:, 0], u_temp, u_temp[:, -1]])
	v_temp = np.vstack([v[0, :], v, v[-1, :]])
	v = np.hstack([v_temp[:, 0], v_temp, v_temp[:, -1]])

	# calculate the kernel
	kernel = np.array([[-1, 0, 1],
					   [-2, 0, 2],
					   [-1, 0, 1]]) / 8.0

	# calculate derivative (should be checked later!)
	uy = convolve2d(u, kernel, mode='same')
	vx = convolve2d(v, kernel, mode='same')

	# calculate vorticity and erase the boundaries
	vort = vx - uy
	vort = vort[1:-1, 1:-1]

	return vort