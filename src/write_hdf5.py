import h5py
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

import os
import json
from glob import glob
from typing import List, Dict, Tuple

from src.utils_data import read_gen


class DatasetH5(Dataset):
	def __init__(self, dataset_list: List[str]) -> None:
		self.dataset_list = dataset_list
		self.dataset_length = len(dataset_list)

	def __len__(self) -> int:
		return self.dataset_length

	def __getitem__(self, idx: int) -> Tuple[List[np.array], np.array]:
		# Call via indexing
		floname = self.dataset_list[idx]
		imnames = [imname_modifier(floname, i+1) for i in range(2)]

		# Instantiate the images and flow objects
		flo = read_gen(floname)  # Flow
		images = [np.array(read_gen(imname)) for imname in imnames]

		return images, flo


def import_dataset(json_paths: List[str]) -> Dict[str, List[str]]:
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


def import_single_set(floname: str) -> Dict[str, List[str]]:
	return {"single": [floname]}


def imname_modifier(floname: str, idx: int = 1) -> str:
	for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.ppm']:
		imname = str(floname.rsplit('_', 1)[0]) + f'_img{idx}' + ext

		if os.path.isfile(imname):
			return imname


def write_hdf5(dataset_dict: Dict[str, List[str]], filename: str) -> None:
	dataloader = {}

	# Define placeholder
	with h5py.File(filename, "w") as out:
		for key, value in dataset_dict.items():
			# Define the shape
			file_shape = read_gen(value[0]).shape

			# Image(s) placeholder
			out.create_dataset(f"{key}/data1", (len(value), file_shape[0], file_shape[1], 3),
							   dtype=h5py.h5t.STD_U8BE)
			out.create_dataset(f"{key}/data2", (len(value), file_shape[0], file_shape[1], 3),
							   dtype=h5py.h5t.STD_U8BE)

			# Label placeholder
			out.create_dataset(f"{key}/label", (len(value), file_shape[0], file_shape[1], file_shape[2]),
							   dtype=h5py.h5t.IEEE_F32BE)

			# Instatiate dataloader
			dataloader[key] = DataLoader(DatasetH5(value), batch_size=1, shuffle=False, num_workers=8)

	# Define dataset variables
	with h5py.File(filename, "a") as out:
		for key, dataload in tqdm(dataloader.items(), ncols=100, desc='Iterate over DataLoader'):
			for i, (images, flow) in enumerate(tqdm(dataload, ncols=100, desc=f"{key} dataset", unit="set")):
				out[f"{key}/data1"][i] = images[0]
				out[f"{key}/data2"][i] = images[1]
				out[f"{key}/label"][i] = flow


if __name__ == "__main__":
	# INPUT
	h5file = 'piv_cai2018.h5'
	# root = '..\..\piv_datasets\cai2018\ztest_hdf5'
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

	# Gather single dataset
	singlepath = os.path.join(root, "single.h5")
	single_dataset = import_single_set("/home/faber/thesis/thesis_faber/images/demo/DNS_turbulence_flow.flo")

	# Write to hdf5 file
	write_hdf5(raw_dataset, filepath)
	write_hdf5(single_dataset, singlepath)

	tqdm.write('DONE!')
