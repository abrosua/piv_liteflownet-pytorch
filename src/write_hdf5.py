import h5py
from tqdm import tqdm
from torchvision import transforms

import os
import json
from glob import glob

from src.utils_data import read_gen


def import_dataset(json_paths):
	dataset = {}

	for json_path in json_paths:
		# Set directory
		subdir = os.path.basename(json_path).rsplit('_', 1)[0]
		directory = os.path.join(os.path.dirname(os.path.dirname(json_path)), subdir)

		with open(json_path) as json_files:
			json_file = json.load(json_files)

			for key, flonames in json_file.items():
				if flonames is not None:
					flopath = [os.path.join(directory, floname) for floname in flonames]

					if key not in dataset.keys():
						dataset[key] = flopath
					else:
						dataset[key].extend(flopath)

	return dataset


def imname_modifier(floname: str, idx: int = 1):
	for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.ppm']:
		imname = str(floname.rsplit('_', 1)[0]) + f'_img{idx}' + ext

		if os.path.isfile(imname):
			return imname


def write_hdf5(dataset, filename):
	transformer = transforms.ToTensor()

	# Define placeholder
	with h5py.File(filename, "w") as out:
		for key, value in dataset.items():
			# Label placeholder
			# label_shape = transformer(read_gen(value[0])).size()
			out.create_dataset(f"X_{key}", (len(value),), dtype='u1')

			# Image placeholder
			# imname = imname_modifier(value[0], root)
			# im_shape = transformer(read_gen(imname)).size()
			out.create_dataset(f"Y_{key}", (len(value),), dtype='u1')

	# Define dataset
	with h5py.File(filename, "a") as out:
		for key, val in dataset.items():
			images, labels = [], []

			for floname in val:
				# Define label
				labels.append([read_gen(floname)])

				# Define image
				tmp_val = []
				for i in range(2):
					imname = imname_modifier(floname, idx=i+1)
					tmp_val.append(read_gen(imname))
				images.append(tmp_val)

			out[f"X_{key}"] = images
			out[f"Y_{key}"] = labels


if __name__ == "__main__":
	# INPUT
	h5file = 'piv_cai2018.h5'
	root = '../../piv_datasets/cai2018/ztest_hdf5'
	json_root = os.path.join(os.path.dirname(root), 'ztest_trainval')

	if not os.path.isdir(root):
		os.makedirs(root)
	filepath = os.path.join(root, h5file)

	# Define json paths
	json_paths = sorted(glob(os.path.join(json_root, f'*.json')))

	# Gather dataset from jsonfiles
	raw_dataset = import_dataset(json_paths)
	for key in raw_dataset.keys():
		tqdm.write(f"{key} dataset length = {len(raw_dataset[key])}")

	# Write to hdf5 file
	write_hdf5(raw_dataset, filepath)

	tqdm.write('DONE!')
