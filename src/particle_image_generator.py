import numpy as np
import scipy.interpolate as itp
import os
from skimage.io import imsave

from test_scripts.utils_eval import flow_scalling


class ParticleImageGen:
	def __init__(self, density=0.05, avg_diameter=1.5, std_diameter=1, laser_thickness=1):
		# INPUT CHECKING!
		if avg_diameter < 0 or std_diameter < 0:
			raise ValueError(f'Input particle average diameter ({avg_diameter}) and '
							 f'particle deviation diameter {std_diameter} must be a positive number!')

		# INPUT Initialization
		self.lt = laser_thickness
		self.p_density = density
		self.p_avg_d = avg_diameter
		self.p_std_d = std_diameter

	def write_image(self, img, dir, floname, idx, ext='.bmp'):
		namelist = [floname, str(self.p_density), str(self.p_avg_d), str(self.p_std_d), str(idx)]
		filename = '_'.join(namelist) + ext
		filepath = os.path.join(dir, filename)

		# save image
		imsave(filepath, img)

	def random_particle(self, cartesian_shape=(256, 256, 1)):
		# INPUT CHECKING!
		if cartesian_shape[0] < 0 or cartesian_shape[1] < 0 or cartesian_shape[2] < 0:
			raise ValueError(f'Input shape ({cartesian_shape}) must be all in positive number!')

		xrange, yrange, zrange = cartesian_shape  # (x, y, z)
		p_count = np.floor(self.p_density * yrange * xrange)

		# Generating random particles
		x = xrange * np.random.rand(p_count)
		y = yrange * np.random.rand(p_count)
		z = zrange * np.random.rand(p_count) - 0.5 * zrange
		d = self.p_avg_d + self.p_std_d * np.random.rand(p_count)

		return x, y, z, d

	def generate_image(self, x_par, y_par, z_par, d_par, cartesian_shape=(256, 256, 1)):
		xrange, yrange, zrange = cartesian_shape
		x, y = np.meshgrid(range(1, xrange+1), range(1, yrange+1))

		# Calculating intensity level of each random particle
		I = 240 * np.exp(-(z_par ** 2) / (self.lt ** 2))

		# Generating particle image
		im = np.zeros(x.shape)
		for p in range(len(x_par)):
			im += I[p] * np.exp(-((x - x_par[p]) ** 2 + (y - y_par[p]) ** 2) / ((d_par[p] / 2) ** 2))

		return np.uint8(im)  # convert to 8-bit data before returning the value

	def create_pair_from_flow(self, dir, floname, ref_coord, ref_flow, cartesian_shape=(256, 256, 1),
							  pad=None, method='cubic'):
		# Init.
		if pad is None:
			extension = np.ceil(0.5 * (np.maximum(cartesian_shape[0], cartesian_shape[1])) * (np.sqrt(2) - 1))
		else:
			extension = np.ceil(pad)

		_, _, zrange = cartesian_shape
		xrange, yrange = cartesian_shape[0] + 2 * extension, cartesian_shape[1] + 2 * extension

		# Generate random particle
		x_par, y_par, z_par, d_par = self.random_particle(cartesian_shape=[xrange, yrange, zrange])

		# 1st Image
		im1_raw = self.generate_image([yrange, xrange], x_par, y_par, z_par, d_par)  # HW
		im1_clean = im1_raw[extension:-extension, extension:-extension]
		self.write_image(im1_clean, dir=dir, floname=floname, idx=0)

		# Interpolate the displacement
		target_flow = flow_scalling([x_par, yrange - y_par], ref_coord, ref_flow, method=method)

		# 2nd Image
		x_new, y_new = x_par + u, y_par + v
		im2_raw = self.generate_image([yrange, xrange], x_new, y_new, z_par, d_par)  # HW
		im2_clean = im2_raw[extension:-extension, extension:-extension]
		self.write_image(im2_clean, dir=dir, floname=floname, idx=1)
