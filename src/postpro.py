import numpy as np
from scipy.signal import convolve2d


def calc_vorticity(flow, calib: float = 1.0):
	# Extract each velocity component
	u = flow[:, :, 0]  # [m/s]
	v = flow[:, :, 1]  # [m/s]

	# calculate the kernel
	kernel = np.array([[1, 0, -1],
					   [2, 0, -2],
					   [1, 0, -1]]) / (8.0 * calib)

	# Calculate spatial derivatives
	du = convolve2d(u, -kernel.T, mode='same', boundary="symm")
	dv = convolve2d(v, kernel, mode='same', boundary="symm")

	# calculate vorticity and erase the boundaries
	vort = dv - du
	shear = dv + du
	normal = -(dv + du)

	return vort, shear, normal


def de_vort(flow, calib: float = 1.0):
	h, w, _ = flow.shape  # Configure the Height and Width
	uy, vx = np.zeros([h, w]), np.zeros([h, w])

	# Extract each velocity component
	u = flow[:, :, 0]  # [m/s]
	v = flow[:, :, 1]  # [m/s]

	u_pad = np.pad(u, 1, mode="edge")
	v_pad = np.pad(v, 1, mode="edge")

	for it in range(h):
		for jt in range(w):
			i, j = it+1, jt+1
			vx[it, jt] = ((v_pad[i+1, j+1] + 2*v_pad[i, j+1] + v_pad[i-1, j+1]) -
						  (v_pad[i+1, j-1] + 2*v_pad[i, j-1] + v_pad[i-1, j-1])
						  ) / (8 * calib)

			uy[it, jt] = ((u_pad[i-1, j-1] + 2*u_pad[i-1, j] + u_pad[i-1, j+1]) -
						  (u_pad[i+1, j-1] + 2*u_pad[i+1, j] + u_pad[i+1, j+1])
						  ) / (8 * calib)

	vort = vx - uy
	return vort, uy, vx


if __name__ == "__main__":
	pass
