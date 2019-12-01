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
            flonames = self._json_pickler(dataset_file)

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

        self.frame_size = read_gen(self.image_list[0][0]).size

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or \
                (self.frame_size[0] % 64) or (self.frame_size[1] % 64):
            self.render_size[0] = ((self.frame_size[0])//64) * 64
            self.render_size[1] = ((self.frame_size[1])//64) * 64

        args.inference_size = self.render_size

        assert (len(self.image_list) == len(self.flow_list))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
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

    def _json_pickler(self, set_dir: str) -> List[str]:
        modes = ['train', 'val', 'test']
        if self.set_type not in modes:
            raise ValueError(f'Unknown input of mode ({self.set_type})! Choose between {(" or ").join(modes)} only!')

        subdir = os.path.basename(set_dir).rsplit('_', 1)[0]
        directory = os.path.join(os.path.dirname(os.path.dirname(set_dir)), subdir)

        datanames = []
        with open(set_dir) as json_file:
            data = json.load(json_file)[self.set_type]

            for line in data:
                bname = os.path.basename(line)
                filename = os.path.join(directory, bname)

                if os.path.isfile(filename):
                    for i in range(self.replicates):
                        datanames.append(filename)

        return datanames


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

        self.frame_size = read_gen(self.image_list[0][0]).size

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or \
                (self.frame_size[0] % 64) or (self.frame_size[1] % 64):
            self.render_size[0] = ((self.frame_size[0]) // 64) * 64
            self.render_size[1] = ((self.frame_size[1]) // 64) * 64

        # Sanity check on the number of image pair and flow
        self.size = len(self.image_list)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], str]:
        # Init.
        index = index % self.size
        im_name = self.image_list[index][0]

        # Cropper and totensor tranformer for the images
        transformer = transforms.Compose([
            transforms.CenterCrop(self.render_size),
            transforms.ToTensor()
        ])

        # Read and transform file into tensor
        imgs = []
        for imname in self.image_list[index]:
            img = transformer(read_gen(imname))
            imgs.append(img)

        return imgs, im_name


class InferenceEval(Dataset):
    def __init__(self, inference_size: Tuple = (-1, -1), root: str = '') -> None:
        self.render_size = list(inference_size)

        self.flow_list = []
        self.image_list = []


        file_list = flo_files_from_folder(root)
        for file in file_list:
            if 'test' in file:
                # print file
                continue

            imbase, imext = os.path.splitext(os.path.basename(file))
            fbase = imbase.rsplit('_', 1)[0]

            img1 = file
            img2 = os.path.join(root, str(fbase) + '_img2' + imext)
            flow = os.path.join(root, str(fbase) + '_flow' + '.flo')

            if not os.path.isfile(img1) or not os.path.isfile(img2) or not os.path.isfile(flow):
                continue

            self.image_list += [[img1, img2]]
            self.flow_list += [flow]

        self.size = len(self.image_list)

        self.frame_size = read_gen(self.image_list[0][0]).size

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or \
                (self.frame_size[0] % 64) or (self.frame_size[1] % 64):
            self.render_size[0] = ((self.frame_size[0]) // 64) * 64
            self.render_size[1] = ((self.frame_size[1]) // 64) * 64

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

        # Cropper and totensor tranformer for the images and flow
        transformer = transforms.Compose([
            transforms.CenterCrop(self.render_size),
            transforms.ToTensor()
        ])

        # Transform into tensor
        img1 = transformer(img1)
        img2 = transformer(img2)
        flow = transformer(flow)

        return [img1, img2], [flow]


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
        f_transforms.RandomTranslate(20),
        f_transforms.RandomRotate(17, diff_angle=3),
        f_transforms.RandomScale([0.9, 2.0]),
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
