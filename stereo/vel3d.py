import numpy as np


def willert(u, v, theta, beta):
	"""
	Using method from Willert, 1997.
	*note: argument 0 for Left Camera, and 1 for Right Camera
	Args:
		u		:
		v		:
		theta	: is off-axis half angle
		beta	: is off-axis half angle in y-z plane
	Returns:
		U, V, W	: 2D3C velocity array
	"""

	U = (u[1] * np.tan(theta[0]) - u[0] * np.tan(theta[1])) / (np.tan(theta[0]) - np.tan(theta[1]))
	V = (v[0] + v[1]) / 2 + (u[1] - u[0]) * (np.tan(beta[1]) - np.tan(beta[0])) / (
			np.tan(theta[0]) - np.tan(theta[1])) / 2
	W = (u[1] - u[0]) / (np.tan(theta[0]) - np.tan(theta[1]))

	return U, V, W
