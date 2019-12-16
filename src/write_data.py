import h5py
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import lmdb

import os
import pickle
import json
from glob import glob
import pyarrow as pa
from typing import List, Dict, Tuple

from src.utils_data import read_gen


class FromList(Dataset):
    def __init__(self, dataset_list: List[str], raw_reading: bool = False) -> None:
        self.dataset_list = dataset_list
        self.dataset_length = len(dataset_list)
        self.raw_reading = raw_reading

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[List[np.array], np.array, str, List[int]]:
        # Call via indexing
        floname = self.dataset_list[idx]
        imnames = [imname_modifier(floname, i+1) for i in range(2)]
        filename = str(os.path.splitext(os.path.basename(floname))[0].rsplit('_', 1)[0])

        # Instantiate the images and flow objects
        flo = read_gen(floname)  # Flow
        fshape = list(flo.shape[:-1])

        if self.raw_reading:
            flo = pickle.dumps(flo)
            images = [raw_reader(imname) for imname in imnames]
        else:
            images = [Image.open(imname) for imname in imnames]

        return images, flo, filename, fshape

def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


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
            out.create_dataset(f"{key}/data1", (len(value), file_shape[0], file_shape[1], 1), dtype=np.uint8)
            out.create_dataset(f"{key}/data2", (len(value), file_shape[0], file_shape[1], 1), dtype=np.uint8)

            # Label placeholder
            out.create_dataset(f"{key}/label", (len(value), file_shape[0], file_shape[1], file_shape[2]),
                               dtype=np.float32)
            out.close()

            # Instatiate dataloader
            dataset = FromList(value)
            dataloader[key] = DataLoader(dataset, shuffle=False, num_workers=16, collate_fn=lambda x: x)

    # Define dataset variables
    with h5py.File(filename, "a") as out:
        for key, dataload in tqdm(dataloader.items(), ncols=100, desc='Iterate over DataLoader'):
            for i, data in enumerate(tqdm(dataload, ncols=100, desc=f"{key} dataset", unit="set")):
                images, flow, fname, fshape = data[0]

                out[f"{key}/data1"][i] = np.array(images[0])
                out[f"{key}/data2"][i] = np.array(images[1])
                out[f"{key}/label"][i] = flow
        out.close()


def write_lmdb(dataset_dict: Dict[str, List[str]], filename: str, write_frequency: int = 1000):
    if not os.path.isdir(filename):
        os.makedirs(filename)

    # Iterate over the dataset dictionary
    for dname, dval in dataset_dict.items():
        dataset = FromList(dval, raw_reading=True)
        data_loader = DataLoader(dataset, shuffle=False, num_workers=16, collate_fn=lambda x: x)

        lmdb_path = os.path.join(filename, f"{dname}.lmdb")
        isdir = os.path.isdir(lmdb_path)

        print("\nGenerate LMDB to %s" % lmdb_path)
        db = lmdb.open(lmdb_path, subdir=isdir,
                       map_size=1099511627776 * 2, readonly=False,
                       meminit=False, map_async=True)

        fshape = [0, 0]
        txn = db.begin(write=True)

        for idx, data in enumerate(data_loader):
            # print(type(data), data)
            images, label, fname, fshape = data[0]
            txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow((images[0], images[1], label)))
            if idx % write_frequency == 0 or idx == len(data_loader)-1:
                print("[%d/%d]" % (idx+1, len(data_loader)))
                txn.commit()
                txn = db.begin(write=True)

        # finish iterating through dataset
        txn.commit()
        keys = [u'{}'.format(k).encode('ascii') for k in range(len(data_loader))]
        with db.begin(write=True) as txn:
            txn.put(b'__keys__', dumps_pyarrow(keys))
            txn.put(b'__len__', dumps_pyarrow(len(keys)))
            txn.put(b'__shape__', dumps_pyarrow(fshape))

        print("Flushing database ...")
        db.sync()
        db.close()


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


if __name__ == "__main__":
    # INPUT
    savedir = 'ztest_hdf5'
    # savedir = 'ztest_lmdb'
    savefile = 'piv_cai2018'

    # root = '..\..\piv_datasets\cai2018\ztest_hdf5'
    root = '../../piv_datasets/cai2018'
    json_root = os.path.join(root, 'ztest_trainval')
    save_root = os.path.join(root, savedir)

    if not os.path.isdir(save_root):
        os.makedirs(save_root)
    filepath = os.path.join(save_root, savefile)

    # Define json paths
    json_paths = sorted(glob(os.path.join(json_root, f'*.json')))

    # Gather dataset from jsonfiles
    raw_dataset = import_dataset(json_paths)
    for key in raw_dataset.keys():
        tqdm.write(f"{key} dataset length = {len(raw_dataset[key])}")

    # Gather single dataset
    singlepath = os.path.join(root, "single.h5")
    single_dataset = import_single_set("/home/faber/thesis/thesis_faber/images/demo/DNS_turbulence_flow.flo")

    # ---- Write to hdf5 file ----
    write_hdf5(raw_dataset, filepath)
    # write_hdf5(single_dataset, singlepath)

    # ---- write to LMDB file ----
    write_lmdb(raw_dataset, filepath, write_frequency=300)

    tqdm.write('DONE!')
