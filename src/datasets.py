import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import json
import os, random
from glob import glob
from typing import Dict, Optional, List, Tuple, Union, Iterable, Any

from src.utils_data import image_files_from_folder, read_gen


class PIVData(Dataset):
    def __init__(self, args, is_cropped: bool = False, root: str = '', replicates: int = 1, mode: str = 'train'
                 ) -> None:
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.replicates = replicates

        exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.ppm']
        dataset_list = sorted(glob(os.path.join(root, f'*.json')))

        self.flow_list = []
        self.image_list = []

        for dataset_file in dataset_list:
            flonames = json_pickler(dataset_file, replicate=self.replicates, set_type=mode)

            for flo in flonames:
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

                self.image_list.append([img1, img2])
                self.flow_list.append(flo)

        self.size = len(self.image_list)

        self.frame_size = read_gen(self.image_list[0][0]).shape

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

        # Instantiate the image cropper
        image_size = img1.shape[:2]
        if self.is_cropped:
            cropper = StaticRandomCrop(image_size, self.crop_size)
        else:
            cropper = StaticCenterCrop(image_size, self.render_size)

        # Instantiate the transformer
        transformer = transforms.Compose([
            cropper,
            transforms.ToTensor()
        ])

        img1 = transformer(img1)
        img2 = transformer(img2)
        flow = transformer(flow)

        return [img1, img2], flow


class InferenceRun(Dataset):
    def __init__(self, root: str = '', pair: bool = True) -> None:

        file_list = image_files_from_folder(root, pair=pair)
        prev_file = None

        self.image_list = []
        for file in file_list:
            if 'test' in file:
                # print file
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

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], str]:
        # Init.
        index = index % self.size

        img1 = read_gen(self.image_list[index][0])
        img2 = read_gen(self.image_list[index][1])
        im_name = self.image_list[index][0]

        # Transform into tensor
        img1 = transforms.ToTensor()(img1)
        img2 = transforms.ToTensor()(img2)

        return [img1, img2], im_name


class InferenceEval(Dataset):
    def __init__(self, inference_size: Tuple = (-1, -1), root: str = '') -> None:
        self.render_size = list(inference_size)
        file_list = image_files_from_folder(root, pair=True)

        self.flow_list = []
        self.image_list = []
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

        self.frame_size = read_gen(self.image_list[0][0]).shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or \
                (self.frame_size[0] % 64) or (self.frame_size[1] % 64):
            self.render_size[0] = ((self.frame_size[0])//64) * 64
            self.render_size[1] = ((self.frame_size[1])//64) * 64

        # Sanity check on the number of image pair and flow
        assert (len(self.image_list) == len(self.flow_list))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        # Init.
        index = index % self.size

        img1 = read_gen(self.image_list[index][0])
        img2 = read_gen(self.image_list[index][1])
        flow = read_gen(self.flow_list[index])
        image_size = img1.shape[:2]

        # Cropper and totensor tranformer the image and flow
        transformer = transforms.Compose([
            StaticCenterCrop(image_size, self.render_size),
            transforms.ToTensor()
        ])

        # Transform into tensor
        img1 = transformer(img1)
        img2 = transformer(img2)
        flow = transformer(flow)

        return [img1, img2], flow


# -------------------- Helper --------------------
class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw), :]


class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size

    def __call__(self, img):
        return img[(self.h - self.th)//2:(self.h + self.th)//2, (self.w - self.tw)//2:(self.w + self.tw)//2, :]


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


def json_pickler(set_dir: str, replicate: int = 1, set_type: str = 'train') -> List[str]:
    modes = ['train', 'val', 'test']
    if set_type not in modes:
        raise ValueError(f'Unknown input of mode ({set_type})! Choose between {(" or ").join(modes)} only!')

    subdir = os.path.basename(set_dir).rsplit('_', 1)[0]
    directory = os.path.join(os.path.dirname(os.path.dirname(set_dir)), subdir)

    datanames = []
    with open(set_dir) as json_file:
        data = json.load(json_file)[set_type]

        for line in data:
            filename = os.path.join(directory, line)

            if os.path.isfile(filename):
                for i in range(replicate):
                    datanames.append(filename)

    return datanames
