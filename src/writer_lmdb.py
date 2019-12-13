import lmdb
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image

import os
import pickle
import json
from glob import glob
from typing import List, Dict

from src.utils_data import read_gen


class ImagesFlo:
    def __init__(self, images: list, flo: np.array) -> None:
        # Dimensions of image for reconstruction - not really necessary
        # for this dataset, but some datasets may include images of
        # varying sizes
        self.imshape = np.array(images[0]).shape
        self.floshape = flo.shape

        self.images = [image.tobytes() for image in images]
        self.label = flo.tobytes()

    def get_images(self) -> List[np.array]:
        """ Returns the image as a numpy array. """
        images = [np.frombuffer(image, dtype=np.uint8).reshape(self.imshape) for image in self.images]
        return images

    def get_flo(self) -> np.array:
        """ Returns the ground truth flow as a numpy array"""
        flo = np.frombuffer(self.label, dtype=np.float32).reshape(self.floshape)
        return flo


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


def store_many_lmdb(dataset_dict: Dict[str, List[str]], lmdb_dir: str) -> None:
    """ Stores an array of images to LMDB.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """

    for key, val in dataset_dict.items():

        # Init. variable
        num_images = len(val)
        map_size = num_images * read_gen(val[0]).nbytes * 10

        # Create a new LMDB DB for all the images
        env = lmdb.open(os.path.join(lmdb_dir, f"{key}_{num_images}_lmdb"), map_size=map_size)

        # Same as before — but let's write all the images in a single transaction
        with env.begin(write=True) as txn:
            for i, flo in enumerate(val):
                images = [read_gen(imname_modifier(flo, i+1)) for i in range(2)]
                flow = read_gen(flo)

                # All key-value pairs need to be Strings
                value = ImagesFlo(images, flow)
                key = f"{i:08}"
                txn.put(key.encode("ascii"), pickle.dumps(value))
        env.close()


if __name__ == "__main__":
    # INPUT
    # lmdb_dir = '..\..\piv_datasets\cai2018\ztest_lmdb'
    lmdb_dir = '../../piv_datasets/cai2018/ztest_lmdb'
    json_root = os.path.join(os.path.dirname(lmdb_dir), 'ztest_trainval')

    if not os.path.isdir(lmdb_dir):
        os.makedirs(lmdb_dir)

    # Define json paths
    json_paths = sorted(glob(os.path.join(json_root, f'*.json')))

    # Gather dataset from jsonfiles
    raw_dataset = import_dataset(json_paths)
    for key in raw_dataset.keys():
        tqdm.write(f"{key} dataset length = {len(raw_dataset[key])}")

    # Gather single dataset
    single_dataset = import_single_set("/home/faber/thesis/thesis_faber/images/demo/DNS_turbulence_flow.flo")

    # Write to hdf5 file
    # store_many_lmdb(raw_dataset, lmdb_dir)
    store_many_lmdb(single_dataset, lmdb_dir)

    tqdm.write('DONE!')
