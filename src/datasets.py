import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import json
import os
from glob import glob
from typing import Optional, List, Tuple

from src.utils_data import image_files_from_folder, flo_files_from_folder, read_gen
import src.flow_transforms as f_transforms


# Mean augmentation global variable
MEAN = ((0.173935, 0.180594, 0.192608), (0.172978, 0.179518, 0.191300))  # PIV-LiteFlowNet-en (Cai, 2019)
# MEAN = ((0.411618, 0.434631, 0.454253), (0.410782, 0.433645, 0.452793))  # LiteFlowNet (Hui, 2018)


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

        # Instantiate norm augmentation
        norm_aug = f_transforms.Normalize(mean=MEAN)

        res_data = tuple(transformer(*data))
        res_data = tuple(norm_aug(*res_data))
        return res_data


class InferenceRun(Dataset):
    def __init__(self, inference_size: Tuple = (-1, -1), root: str = '', pair: bool = True) -> None:
        self.render_size = list(inference_size)
        file_list = image_files_from_folder(root, pair=pair)
        prev_file = None

        self.image_list = []
        for file in file_list:
            if 'test' in file:
                continue

            if pair:  # Using paired images
                imbase, imext = os.path.splitext(os.path.basename(file))
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

            if not os.path.isfile(img1) or not os.path.isfile(img2):
                continue

            self.image_list += [[img1, img2]]

        self.size = len(self.image_list)

        if self.size > 0:
            self.frame_size = read_gen(self.image_list[0][0]).size

            if (self.render_size[0] < 0) or (self.render_size[1] < 0) or \
                    (self.frame_size[0] % 64) or (self.frame_size[1] % 64):
                self.render_size[0] = ((self.frame_size[0]) // 64) * 64
                self.render_size[1] = ((self.frame_size[1]) // 64) * 64

        else:
            self.frame_size = None

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], str]:
        # Init.
        index = index % self.size
        im_name = self.image_list[index][0]

        # Cropper and totensor tranformer for the images
        transformer = transforms.Compose([
            transforms.CenterCrop(self.render_size),
            transforms.ToTensor(),
            f_transforms.Normalize(mean=MEAN)
        ])

        # Read and transform file into tensor
        imgs = []
        for imname in self.image_list[index]:
            img = transformer(read_gen(imname))
            imgs.append(img)

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
            f_transforms.Normalize(mean=MEAN)
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
        f_transforms.RandomTranslate(20),  # 16 or 20
        f_transforms.RandomRotate(17, diff_angle=3),
        f_transforms.RandomScale([0.95, 1.45]),
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
