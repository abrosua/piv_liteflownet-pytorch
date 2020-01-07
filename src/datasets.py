import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import lmdb
import six
import pyarrow as pa

import json, h5py
import pickle
import os
from glob import glob
from typing import Optional, List, Tuple

from src.utils_data import image_files_from_folder, flo_files_from_folder, read_gen
import src.flow_transforms as f_transforms


class PIVH5(Dataset):
    def __init__(self, args, is_cropped: bool = False, root: str = '', replicates: int = 1, mode: str = 'train',
                 transform: Optional[object] = None, load_data: bool = False, data_cache_size: int = 3) -> None:
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.transform = transform

        self.replicates = replicates
        self.set_type = mode

        # Caching
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size

        dataset_list = sorted(glob(os.path.join(root, f'*.h5')))

        for h5dataset_fp in dataset_list:
            self._add_data_infos(str(h5dataset_fp), load_data)

        self.size = len(self.get_data_infos('label'))

        if self.size > 0:
            self.frame_size = list(self.data_info[0]['shape'])

            if (self.render_size[0] < 0) or (self.render_size[1] < 0) or \
                    (self.frame_size[0] % 64) or (self.frame_size[1] % 64):
                self.render_size[0] = ((self.frame_size[0])//64) * 64
                self.render_size[1] = ((self.frame_size[1])//64) * 64

            args.inference_size = self.render_size
        else:
            self.frame_size = None

        # Sanity check on the number of image pair and flow
        assert len(self.get_data_infos('data1')) \
               == len(self.get_data_infos('data2')) \
               == len(self.get_data_infos('label'))
        print('DONE')

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        index = index % self.size

        img1 = Image.fromarray(self.get_data('data1', index)).convert('RGB')
        img2 = Image.fromarray(self.get_data('data2', index)).convert('RGB')
        flow = self.get_data('label', index)
        data = [[img1, img2], [flow]]

        if self.is_cropped:
            crop_type = 'rand'
            csize = self.crop_size
        else:
            crop_type = 'center'
            csize = self.render_size

        # Instantiate the transformer
        if self.transform is None:
            transformer = f_transforms.Compose([
                f_transforms.Crop(csize, crop_type=crop_type),
                f_transforms.ModToTensor()
            ])
        else:
            transformer = self.transform

        res_data = tuple(transformer(*data))
        return res_data

    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path) as h5_file:
            # Walk through a specific groups, extracting datasets
            group = h5_file[self.set_type]

            for dname, dset in group.items():
                for ds in dset[()]:
                    idx = -1  # if data is not loaded its cache index is -1

                    if load_data:
                        # add data to the data cache
                        idx = self._add_to_cache(ds, file_path)

                    self.data_info.append(
                        {'file_path': file_path, 'type': dname, 'shape': ds.shape, 'cache_idx': idx})

    def _load_data(self, file_path):
        """
        Load data to the cache given the file path and update the cache index in the data_info structure.
        """
        with h5py.File(file_path) as h5_file:
            # Walk through a specific groups, extracting datasets
            group = h5_file[self.set_type]

            for dname, dset in group.items():
                for ds in dset[()]:
                    # add data to the data cache and retrieve the cache index
                    idx = self._add_to_cache(ds, file_path)

                    # find the beginning index of the hdf5 file we are looking for
                    file_idx = next(i for i, v in enumerate(self.data_info) if v['file_path'] == file_path)

                    # the data info should have the same index since we loaded it in the same way
                    self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])

            # remove invalid cache_idx
            self.data_info = [
                {'file_path': di['file_path'],
                 'type': di['type'],
                 'shape': di['shape'],
                 'cache_idx': -1
                 }
                if di['file_path'] == removal_keys[0] else di
                for di in self.data_info
            ]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """
        Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        """
        Call this function anytime you want to access a chunk of data from the dataset.
        This will make sure that the data is loaded in case it is not part of the data cache.
        """
        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)

        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]


class PIVLMDB(Dataset):
    def __init__(self, args, is_cropped: bool = False, root: str = '', replicates: int = 1, mode: str = 'train',
                 transform: Optional[object] = None) -> None:
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.transform = transform

        self.replicates = replicates
        self.set_type = mode

        self.db_path = os.path.join(root, f"{mode}.lmdb")
        self.env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
                             readonly=True, lock=False, readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.size = int(loads_pyarrow(txn.get(b'__len__')))
            self.keys = list(loads_pyarrow(txn.get(b'__keys__')))
            self.frame_size = list(loads_pyarrow(txn.get(b'__shape__')))

        if self.size > 0:
            if (self.render_size[0] < 0) or (self.render_size[1] < 0) or \
                    (self.frame_size[0] % 64) or (self.frame_size[1] % 64):
                self.render_size[0] = ((self.frame_size[0])//64) * 64
                self.render_size[1] = ((self.frame_size[1])//64) * 64

            args.inference_size = self.render_size
        else:
            self.frame_size = None

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Notes for IO object: Use each COUNTERPART method to read/write the object!
        """
        index = index % self.size

        # Init. buffer files
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = list(loads_pyarrow(byteflow))

        # Generate images
        img1, img2 = self.get_data(unpacked[0]), self.get_data(unpacked[1])
        flow = pickle.loads(unpacked[2])
        data = [[img1, img2], [flow]]

        if self.is_cropped:
            crop_type = 'rand'
            csize = self.crop_size
        else:
            crop_type = 'center'
            csize = self.render_size

        # Instantiate the transformer
        if self.transform is None:
            transformer = f_transforms.Compose([
                f_transforms.Crop(csize, crop_type=crop_type),
                f_transforms.ModToTensor()
            ])
        else:
            transformer = self.transform

        res_data = tuple(transformer(*data))
        return res_data

    @staticmethod
    def get_data(unpacked, imdata=True):
        imgbuf = unpacked
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)

        if imdata:
            data = Image.open(buf).convert('RGB')
        else:
            data = np.load(buf, allow_pickle=True)

        return data


class PIVData(Dataset):
    def __init__(self, args, is_cropped: bool = False, root: str = '', replicates: int = 1, mode: str = 'train',
                 transform: Optional[object] = None) -> None:
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.transform = transform

        self.replicates = replicates
        self.set_type = mode

        exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.ppm']
        dataset_list = sorted(glob(os.path.join(root, f'*.json')))

        self.flow_list = []
        self.image_list = []

        for dataset_file in dataset_list:
            flonames = json_pickler(dataset_file, self.set_type, self.replicates)

            for flo in flonames:
                if 'test' in flo:
                    continue

                fbase = os.path.splitext(flo)[0]
                fbase = fbase.rsplit('_', 1)[0]

                img1, img2 = None, None
                for ext in exts:
                    img1 = str(fbase) + '_img1' + ext
                    img2 = str(fbase) + '_img2' + ext
                    if os.path.isfile(img1):
                        break

                if not os.path.isfile(img1) or not os.path.isfile(img2) or not os.path.isfile(flo):
                    continue

                self.image_list.append([img1, img2])
                self.flow_list.append(flo)

        self.size = len(self.image_list)

        if self.size > 0:
            self.frame_size = read_gen(self.image_list[0][0]).size

            if (self.render_size[0] < 0) or (self.render_size[1] < 0) or \
                    (self.frame_size[0] % 64) or (self.frame_size[1] % 64):
                self.render_size[0] = ((self.frame_size[0])//64) * 64
                self.render_size[1] = ((self.frame_size[1])//64) * 64

            args.inference_size = self.render_size
        else:
            self.frame_size = None

        # Sanity check on the number of image pair and flow
        assert (len(self.image_list) == len(self.flow_list))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        index = index % self.size

        img1 = read_gen(self.image_list[index][0])
        img2 = read_gen(self.image_list[index][1])
        flow = read_gen(self.flow_list[index])
        data = [[img1, img2], [flow]]

        if self.is_cropped:
            crop_type = 'rand'
            csize = self.crop_size
        else:
            crop_type = 'center'
            csize = self.render_size

        # Instantiate the transformer
        if self.transform is None:
            transformer = f_transforms.Compose([
                f_transforms.Crop(csize, crop_type=crop_type),
                f_transforms.ModToTensor()
            ])
        else:
            transformer = self.transform

        res_data = tuple(transformer(*data))
        return res_data


class InferenceRun(Dataset):
    def __init__(self, inference_size: Tuple = (-1, -1), root: str = '', pair: bool = True, use_stereo: bool = False
                 ) -> None:
        self.render_size = list(inference_size)
        if use_stereo:
            file_list = [image_files_from_folder(x[0], pair=pair) for x in os.walk(root)
                         if os.path.basename(x[0]) != os.path.basename(root)]
            assert len(file_list[0]) == len(file_list[1])
        else:
            file_list = image_files_from_folder(root, pair=pair)
        prev_file = None

        self.image_list, self.name_list = [], []
        for files in file_list:
            tmp_image_list, tmp_name_list = [], []

            for file in files:
                if 'test' in file:
                    continue

                if pair:  # Using paired images
                    imbase, imext = os.path.splitext(os.path.basename(str(file)))
                    fbase = imbase.rsplit('_', 1)[0]

                    img1 = file
                    img2 = os.path.join(root, str(fbase) + '_img2' + imext)

                else:  # Using sequential images
                    if prev_file is None:
                        prev_file = file
                        continue
                    else:
                        img1, img2 = prev_file, file
                        prev_file = file
                        fbase = os.path.splitext(os.path.basename(str(img1)))[0]
                        fbase = fbase.rsplit('_', 1)[0] if use_stereo else fbase

                if not os.path.isfile(img1) or not os.path.isfile(img2):
                    continue

                tmp_image_list += [[img1, img2]]
                tmp_name_list += [fbase]

            self.image_list.append(tmp_image_list)
            self.name_list.append(tmp_name_list)

        assert len(self.image_list[0]) == len(self.image_list[1]) and \
               len(self.name_list[0]) == len(self.name_list[1])
        self.size = len(self.image_list[0])

        if self.size > 0:
            img_tmp = self.image_list[0][0][0]
            self.frame_size = read_gen(img_tmp).size

            if (self.render_size[0] < 0) or (self.render_size[1] < 0) or \
                    (self.frame_size[0] % 64) or (self.frame_size[1] % 64):
                self.render_size[0] = ((self.frame_size[0]) // 64) * 64
                self.render_size[1] = ((self.frame_size[1]) // 64) * 64

        else:
            self.frame_size = None

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[str]]:
        # Init.
        index = index % self.size
        im_name = [name_list[index] for name_list in self.name_list]

        # Cropper and totensor tranformer for the images
        transformer = transforms.Compose([
            transforms.CenterCrop(self.render_size),
            transforms.ToTensor(),
        ])

        # Read and transform file into tensor
        imgs = []
        for im_list in self.image_list:
            for i, imname in enumerate(im_list[index]):
                imgs.append(transformer(read_gen(imname)))

        return imgs, im_name


class InferenceEval(Dataset):
    def __init__(self, inference_size: Tuple = (-1, -1), root: str = '', set_type: Optional[str] = None) -> None:
        self.render_size = list(inference_size)
        exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.ppm']

        self.flow_list = []
        self.image_list = []

        root_ext = os.path.splitext(root)[1]
        if root_ext and set_type is not None:
            if root_ext == '.json':
                flo_list = json_pickler(root, set_type=set_type)
            else:
                raise ValueError(f'Only json format is currently supported! Change the input path ({root}).')
        else:
            flo_list = flo_files_from_folder(root)

        for flo in flo_list:
            if 'test' in flo:
                # print file
                continue

            fbase = os.path.splitext(flo)[0]
            fbase = fbase.rsplit('_', 1)[0]

            img1, img2 = None, None
            for ext in exts:
                img1 = str(fbase) + '_img1' + ext
                img2 = str(fbase) + '_img2' + ext
                if os.path.isfile(img1):
                    break

            if not os.path.isfile(img1) or not os.path.isfile(img2) or not os.path.isfile(flo):
                continue

            self.image_list += [[img1, img2]]
            self.flow_list += [flo]

        self.size = len(self.image_list)

        if self.size > 0:
            self.frame_size = read_gen(self.image_list[0][0]).size

            if (self.render_size[0] < 0) or (self.render_size[1] < 0) or \
                    (self.frame_size[0] % 64) or (self.frame_size[1] % 64):
                self.render_size[0] = ((self.frame_size[0]) // 64) * 64
                self.render_size[1] = ((self.frame_size[1]) // 64) * 64

        else:
            self.frame_size = None

        # Sanity check on the number of image pair and flow
        assert (len(self.image_list) == len(self.flow_list))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # Init.
        index = index % self.size

        img1 = read_gen(self.image_list[index][0])
        img2 = read_gen(self.image_list[index][1])
        flow = read_gen(self.flow_list[index])
        data = [[img1, img2], [flow]]

        # Cropper and totensor tranformer for the images and flow
        transformer = f_transforms.Compose([
            f_transforms.Crop(self.render_size, crop_type='center'),
            f_transforms.ModToTensor(),
        ])

        res_data = tuple(transformer(*data))
        return res_data


# -------------------- Transformer --------------------
def get_transform(args):
    """
    The default transformer for the training script.
    WARNING:
    Args:
        args: argument parser.
    Returns:
        Both training and validation data augmentation transformer.
    """

    # Training set transformer (Data Augmentation)
    train_tranformer = f_transforms.Compose([
        f_transforms.RandomTranslate(16),  # 16 or 20
        # f_transforms.RandomRotate(17, diff_angle=3),
        f_transforms.RandomScale([0.95, 1.45]),
        f_transforms.RandomHorizontalFlip(),
        f_transforms.RandomVerticalFlip(),
        f_transforms.Crop(args.crop_size, crop_type='rand'),
        f_transforms.ModToTensor(),
        f_transforms.RandomPhotometric(
            min_noise_stddev=0.0,
            max_noise_stddev=0.04,
            min_contrast=-0.8,
            max_contrast=0.4,
            brightness_stddev=0.2,
            min_color=0.5,
            max_color=2.0,
            min_gamma=0.7,
            max_gamma=1.5),
    ])

    # Validation set transformer (Data Augmentation)
    val_tranformer = f_transforms.Compose([
        f_transforms.RandomHorizontalFlip(),
        f_transforms.RandomVerticalFlip(),
        f_transforms.Crop(args.crop_size, crop_type='rand'),
        f_transforms.ModToTensor(),
    ])

    return train_tranformer, val_tranformer


# -------------------- Helper --------------------
def json_pickler(set_dir: str, set_type: str, replicates: int = 1) -> List[str]:
    modes = ['train', 'val', 'test']
    if set_type not in modes:
        raise ValueError(f'Unknown input of mode ({set_type})! Choose between {(" or ").join(modes)} only!')

    subdir = os.path.basename(set_dir).rsplit('_', 1)[0]
    directory = os.path.join(os.path.dirname(os.path.dirname(set_dir)), subdir)

    datanames = []
    with open(set_dir) as json_file:
        data = json.load(json_file)[set_type]

        for line in data:
            bname = os.path.basename(line)
            filename = os.path.join(directory, bname)

            if os.path.isfile(filename):
                for i in range(replicates):
                    datanames.append(filename)

    return datanames


def df_pickler(set_dir: str, replicate: int = 1, mode='json') -> List[str]:
    subdir = os.path.basename(set_dir).rsplit('_', 1)[0]
    directory = os.path.join(os.path.dirname(set_dir), subdir)

    if mode == 'csv':
        df_label = pd.read_csv(set_dir)
    elif mode == 'json':
        df_label = pd.read_json(set_dir)
    else:
        raise ValueError(f'Unknown input occured (mode: {mode}! Choose between "csv" or "json" only!')

    datanames = []
    for index, row in df_label.iterrows():
        filename = os.path.join(directory, row['floname'])

        if os.path.isfile(filename):
            for i in range(replicate):
                datanames.append(filename)

    return datanames


def txt_pickler(set_dir: str, replicate: int = 1) -> List[str]:
    subdir = os.path.basename(set_dir).rsplit('_', 1)[0]
    directory = os.path.join(os.path.dirname(set_dir), subdir)

    datanames = []
    with open(set_dir) as fp:
        for line in fp:
            filename = os.path.join(directory, line)

            if os.path.isfile(filename):
                for i in range(replicate):
                    datanames.append(filename)

    return datanames


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)
