import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageFilter, ImageOps
import scipy.ndimage as ndimage

import math
import numbers
import random
import collections
from typing import List, Tuple, Optional, Union, Any

from src.utils_plot import resize_flow, horizontal_flip_flow, vertical_flip_flow


# Pixel VALUE transformer
class RandomPhotometric(object):
    """
    Applies photometric augmentations to a list of image tensors
        i.e., Contrast, Brightness, Color mult., Noise and Gamma.
    Each image in the list is augmented in the same way (Sequential process).
    """
    def __init__(self,
                 min_noise_stddev: float = 0.0,
                 max_noise_stddev: float = 0.0,
                 min_contrast: float = 0.0,
                 max_contrast: float = 0.0,
                 brightness_stddev: float = 0.0,
                 min_color: float = 1.0,
                 max_color: float = 1.0,
                 min_gamma: float = 1.0,
                 max_gamma: float = 1.0
                 ) -> None:

        self.noise_stddev = np.random.uniform(min_noise_stddev, max_noise_stddev)
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast
        self.brightness_stddev = brightness_stddev
        self.min_color = min_color
        self.max_color = max_color
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma

    def __call__(self, ims: List[torch.Tensor], label_list: Optional[List[np.array]] = None
                 ) -> Tuple[List[torch.Tensor], Optional[List[np.array]]]:
        """
        Args:
            ims: list of 3-channel images normalized to [0, 1].
        Returns:
            normalized images with photometric augmentations. Has the same
            shape as the input.
        """
        # Contrast aug.
        contrast = np.random.uniform(self.min_contrast, self.max_contrast)

        # Gamma aug.
        gamma = np.random.uniform(self.min_gamma, self.max_gamma)
        gamma_inv = 1.0 / gamma

        # Color aug.
        color = torch.from_numpy(np.random.uniform(self.min_color, self.max_color, 3)).float()

        # Noise aug.
        if self.noise_stddev > 0.0:
            noise = np.random.normal(scale=self.noise_stddev)
        else:
            noise = 0
        # Brightness aug.
        if self.brightness_stddev > 0.0:
            brightness = np.random.normal(scale=self.brightness_stddev)
        else:
            brightness = 0

        # Perform data augmentation
        out = []
        for im in ims:
            im_re = im.permute(1, 2, 0)
            im_re = (im_re * (contrast + 1.0) + brightness) * color
            im_re = torch.clamp(im_re, min=0.0, max=1.0)
            im_re = torch.pow(im_re, gamma_inv)
            im_re += noise

            im = im_re.permute(2, 0, 1)
            out.append(im)

        return out, label_list


class RandomGaussianBlur(object):
    """Add gaussian blur to the image with binary occurrence probability"""
    def __init__(self, radius: float = 2.0) -> None:
        self.radius = radius

    def __call__(self, img_list: List[Image.Image], label_list: Optional[List[np.array]] = None
                 ) -> Tuple[List[Image.Image], Optional[List[np.array]]]:
        if random.random() < 0.5:
            img_list = [
                img.filter(ImageFilter.GaussianBlur(self.radius))
                for img in img_list
            ]
        return img_list, label_list


class ChromaticAugment(object):
    """Perform chromatic augmentation process, based on the ImageNet PCA value
    Each image in the list is augmented in the same way (Sequential process).
    """
    def __init__(self) -> None:
        pass

    def __call__(self, img_list: List[Image.Image], label_list: List[np.array]
                 ) -> Tuple[List[Image.Image], List[np.array]]:
        """
        Args:
            img_list: list of 3-channel images normalized to [0, 1].
        Returns:
            Augmented images with chromatic augmentations.
            Has the same shape as the input.
        """
        assert len(img_list) == 2 and len(label_list) <= 1

        return img_list, label_list


# SPATIAL Transformer
class RandomTranslate(object):
    """
    Translation Augmentation: Translate both the image and the respective flow.
    """
    def __init__(self, translation) -> None:
        if isinstance(translation, numbers.Number):  # Uniform translation between axis
            self.translation = (int(translation), int(translation))
        else:  # Same translation for both axis
            self.translation = translation

    def __call__(self, img_list: List[Image.Image], label_list: List[np.array]
                 ) -> Tuple[List[Image.Image], List[np.array]]:
        """
        Args:
            img_list: Pair of input images (img1 and img2)
            label_list: THe respective ground truth flow field
        Returns:
            Translated pair of image and the respective flow.
        """
        assert len(img_list) == 2 and len(label_list) <= 1

        # Initial shape and the translation
        w, h = img_list[0].size
        th, tw = self.translation
        tw = int(random.uniform(-tw, tw) * w / 100)
        th = int(random.uniform(-th, th) * h / 100)

        if tw == 0 and th == 0:  # NO TRANSLATION!
            return img_list, label_list

        # compute x1_a, x2_a, y1_a, y2_a for img1 and target
        #   and x1_b,x2_b,y1_b,y2_b for img2
        x1_a, x2_a, x1_b, x2_b = max(0, tw), min(w + tw, w), max(0, -tw), min(w - tw, w)
        y1_a, y2_a, y1_b, y2_b = max(0, th), min(h + th, h), max(0, -th), min(h - th, h)

        img_list[0] = img_list[0].crop((x1_a, y1_a, x2_a, y2_a))
        img_list[1] = img_list[1].crop((x1_b, y1_b, x2_b, y2_b))

        if len(label_list) == 1:
            label_list[0] = label_list[0][y1_a:y2_a, x1_a:x2_a]
            label_list[0][:, :, 0] += tw
            label_list[0][:, :, 1] += th

        return img_list, label_list


class RandomRotate(object):
    """
    Random rotation of the image from -angle to angle (in degrees)
    This is useful for dataAugmentation, especially for geometric problems such as FlowEstimation
    Args:
        angle: max angle of the rotation
        diff_angle: Default: 0. Must stay less than 10 degrees, or linear approximation of flowmap will be off.
        interpolation order: Default: 2 (bilinear)
        reshape: Default: false. If set to true, image size will be set to keep every pixel in the image.
    Returns:
        Rotated pair of image and the respective flow.
    """
    def __init__(self, angle: float, diff_angle: float = 0.0, order: int = 2, reshape: bool = False) -> None:
        self.angle = angle
        self.diff_angle = diff_angle if diff_angle < 10.0 else diff_angle % 10.0

        self.reshape = reshape
        self.order = order

    def __call__(self, img_list: List[Image.Image], label_list: List[np.array]
                 ) -> Tuple[List[Image.Image], List[np.array]]:
        assert len(img_list) == 2 and len(label_list) <= 1

        # Variable init.
        applied_angle = random.uniform(-self.angle, self.angle)
        diff = random.uniform(-self.diff_angle, self.diff_angle)

        angle1 = applied_angle - diff / 2
        angle2 = applied_angle + diff / 2
        angle1_rad = angle1 * np.pi / 180

        img_list[0] = ndimage.interpolation.rotate(np.asarray(img_list[0]), angle1, reshape=self.reshape,
                                                   order=self.order)
        img_list[1] = ndimage.interpolation.rotate(np.asarray(img_list[1]), angle2, reshape=self.reshape,
                                                   order=self.order)
        img_list[0] = Image.fromarray(img_list[0])
        img_list[1] = Image.fromarray(img_list[1])

        if len(label_list) == 1:
            target = label_list[0]
            h, w, c = target.shape
            assert c == 2

            def rotate_flow(i, j, k):
                return -k * (j - w / 2) * (diff * np.pi /
                                           180) + (1 - k) * (i - h / 2) * (
                                               diff * np.pi / 180)

            rotate_flow_map = np.fromfunction(rotate_flow, target.shape)
            target += rotate_flow_map
            target = ndimage.interpolation.rotate(
                target, angle1, reshape=self.reshape, order=self.order)
            # flow vectors must be rotated too! careful about Y flow which is upside down
            target_ = np.copy(target)
            target[:, :, 0] = np.cos(angle1_rad) * target_[:, :, 0] + np.sin(
                angle1_rad) * target_[:, :, 1]
            target[:, :, 1] = -np.sin(angle1_rad) * target_[:, :, 0] + np.cos(
                angle1_rad) * target_[:, :, 1]
            label_list = [target]

        return img_list, label_list


class RandomScale(object):
    """
    Randomly resize image & label with scale factor in [scale_min, scale_max]
    *** WARNING *** Scale operation is dangerous for sparse optical flow ground truth
    """
    def __init__(self, scale, aspect_ratio=None, method: str = 'bilinear') -> None:
        assert (isinstance(scale, collections.Iterable) and len(scale) == 2)

        # Input scale restrictions
        if isinstance(scale, collections.Iterable) and len(scale) == 2 \
                and isinstance(scale[0], numbers.Number) and isinstance(scale[1], numbers.Number) \
                and 0 < scale[0] < scale[1]:
            self.scale = scale
        else:
            raise (RuntimeError("transforms.RandScale() scale param error.\n"))

        # Input aspect ratio restrictions
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif isinstance(aspect_ratio, collections.Iterable) and len(aspect_ratio) == 2 \
                and isinstance(aspect_ratio[0], numbers.Number) and isinstance(aspect_ratio[1], numbers.Number) \
                and 0 < aspect_ratio[0] < aspect_ratio[1]:
            self.aspect_ratio = aspect_ratio
        else:
            raise RuntimeError("transforms.RandScale() aspect_ratio param error.\n")

        self.method = method

    def __call__(self, img_list: List[Image.Image], label_list: List[np.array]
                 ) -> Tuple[List[Image.Image], List[np.array]]:
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        temp_aspect_ratio = 1.0

        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)

        scale_factor_w = temp_scale * temp_aspect_ratio
        scale_factor_h = temp_scale / temp_aspect_ratio

        w, h = img_list[0].size
        new_w = int(w * scale_factor_w)
        new_h = int(h * scale_factor_h)

        img_list = [img.resize((new_w, new_h), Image.BILINEAR) for img in img_list]
        label_list = [resize_flow(label, new_w, new_h, self.method) for label in label_list]

        return img_list, label_list


class Crop(object):
    """
    Crops the given PIL Image with Center or Random crop method.
    """

    def __init__(self, size, crop_type: str = 'center', padding=None) -> None:
        """
        Args:
        size (sequence or int): Desired output size of the crop.
            If size is an int instead of sequence like (h, w), a square crop (size, size) is made.
        crop_type: cropping method to choose (center or random crop)
        padding (Optional): add padding to the input,
            necessary for cropping with larger resolution than the original size.
        """
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise RuntimeError("crop size error.\n")

        if crop_type == 'center' or crop_type == 'rand':
            self.crop_type = crop_type
        else:
            raise RuntimeError("crop type error: rand | center\n")

        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise RuntimeError("padding in Crop() should be a number list\n")

            if len(padding) != 3:
                raise RuntimeError("padding channel is not equal with 3\n")
        else:
            raise RuntimeError("padding in Crop() should be a number list\n")

    def __call__(self, img_list: List[Image.Image], label_list: List[np.array]
                 ) -> Tuple[List[Image.Image], List[np.array]]:
        w, h = img_list[0].size
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)

        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)

        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise RuntimeError("flow_transforms.Crop() need padding while padding argument is None\n")

            border = (pad_w_half, pad_h_half, pad_w - pad_w_half, pad_h - pad_h_half)

            # Image padding
            img_list = [
                ImageOps.expand(img, border=border, fill=tuple([int(item) for item in self.padding]))
                for img in img_list
            ]
            # Label padding
            if len(label_list) > 0:
                if label_list[0].shape[2] == 3:
                    label_list = [
                        np.pad(label,
                               ((pad_h_half, pad_h - pad_h_half), (pad_w_half, pad_w - pad_w_half), (0, 0)),
                               'constant') for label in label_list
                    ]
                else:
                    raise RuntimeError("Cropping to larger size not supported for optical flow without mask.\n")

        w, h = img_list[0].size
        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = (h - self.crop_h) // 2
            w_off = (w - self.crop_w) // 2

        # Image padding
        img_list = [
            img.crop((w_off, h_off, w_off + self.crop_w, h_off + self.crop_h))
            for img in img_list
        ]
        # Label padding
        label_list = [
            label[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w, :]
            for label in label_list
        ]
        return img_list, label_list


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, img_list: List[Image.Image], label_list: List[np.array]
                 ) -> Tuple[List[Image.Image], List[np.array]]:
        if random.random() < 0.5:
            img_list = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_list]
            label_list = [horizontal_flip_flow(label) for label in label_list]

        return img_list, label_list


class RandomVerticalFlip(object):
    """Randomly vertically flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, img_list: List[Image.Image], label_list: List[np.array]
                 ) -> Tuple[List[Image.Image], List[np.array]]:
        if random.random() < 0.5:
            img_list = [img.transpose(Image.FLIP_TOP_BOTTOM) for img in img_list]
            label_list = [vertical_flip_flow(label) for label in label_list]

        return img_list, label_list


# GLOBAL Transformer
class Normalize(object):
    """
    Given mean: (R, G, B) and std: (R, G, B), will normalize each channel of the torch Tensor in [C, H, W] format!
    i.e. channel = (channel - mean) / std
    """
    def __init__(self, mean: Tuple[Tuple[float, ...], ...], std: Tuple[Tuple[float, ...], ...] = ((1.0, 1.0, 1.0),)
                 ) -> None:
        # Mean and St. Deviation initialization for the Normalization transformer
        if len(mean) == 1:
            self.mean = (mean[0], mean[0])
        elif len(mean) == 2:
            self.mean = mean
        else:
            raise ValueError(f'Unknown mean input format ({mean})!')

        if len(std) == 1:
            self.std = (std[0], std[0])
        elif len(std) == 2:
            self.std = std
        else:
            raise ValueError(f'Unknown standard deviation (std) input format ({std})!')

    def __call__(self, img_list: List[torch.Tensor], label_list: List[np.array]
                 ) -> Tuple[List[torch.Tensor], List[np.array]]:

        norm_img_list = []
        for i, img in enumerate(img_list):
            norm = transforms.Normalize(self.mean[i], self.std[i])
            norm_img_list.append(norm(img))

        return norm_img_list, label_list


class Resize(object):
    """
    Resize the input PIL Image to the given size.
    'size' is a 2-element tuple or list in the order of (w, h)
    """
    def __init__(self, size, method: str = 'bilinear') -> None:
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.method = method

    def __call__(self, img_list: List[Image.Image], label_list: List[np.array]
                 ) -> Tuple[List[Image.Image], List[np.array]]:
        img_list = [img.resize(self.size, Image.BILINEAR) for img in img_list]
        label_list = [resize_flow(label, self.size[0], self.size[1], self.method) for label in label_list]

        return img_list, label_list


# MODIFIER
class ModToTensor(object):
    """
    Modification of the original ToTensor transformer by torchvision, to handle the specific dataset input.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __call__(self, img_list: List[Image.Image], label_list: List[np.array]
                 ) -> Tuple[List[torch.Tensor], List[np.array]]:
        img_list = [transforms.ToTensor()(img) for img in img_list]
        label_list = [transforms.ToTensor()(label) for label in label_list]

        return img_list, label_list


class Compose(object):
    """
    Composes several transforms together.
    """
    def __init__(self, transformers: List) -> None:
        self.transformers = transformers

    def __call__(self, *args):
        for t in self.transformers:
            args = t(*args)

        return args
