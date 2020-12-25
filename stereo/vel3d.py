import numpy as np


def willert(flow, theta, beta):
	"""
	Using method from Willert, 1997.
	*note: argument 0 for Left Camera, and 1 for Right Camera
	Args:
		u		:
		v		:
		theta	: is off-axis half angle (radians)
		beta	: is off-axis half angle in y-z plane (radians)
	Returns:
		U, V, W	: 2D3C velocity array
	"""
	# Init. variable
	u, v = [flo[:, :, 0] for flo in flow], [flo[:, :, 1] for flo in flow]

	u_3c = (u[1] * np.tan(theta[0]) - u[0] * np.tan(theta[1])) / (np.tan(theta[0]) - np.tan(theta[1]))
	v_3c = (v[0] + v[1]) / 2 + (u[1] - u[0]) * (np.tan(beta[1]) - np.tan(beta[0])) / (
			np.tan(theta[0]) - np.tan(theta[1])) / 2
	w_3c = (u[1] - u[0]) / (np.tan(theta[0]) - np.tan(theta[1]))

	return np.dstack([u_3c, v_3c, w_3c])
