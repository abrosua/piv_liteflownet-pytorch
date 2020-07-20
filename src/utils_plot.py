import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as itp
from PIL import Image

import os, io, time, re
from glob import glob
from typing import Union, List, Tuple, Optional

from src.utils_color import compute_color


TAG_STRING = 'PIEH'
TAG_FLOAT = 202021.25
flags = {
    'debug': False
}

"""
flowIO.h
"""
UNKNOWN_FLOW_THRESH = 1e9


def read_flow(filename: str, crop_window: Union[int, Tuple[int, int, int, int]] = 0) -> np.array:
    """
        Read a .flo file (Middlebury format).
        Parameters
        ----------
        filename : str
            Filename where the flow will be read. Must have extension .flo.
        Returns
        -------
        flow : ndarray, shape (height, width, 2), dtype float32
            The read flow from the input file.
        """

    if not isinstance(filename, io.BufferedReader):
        if not isinstance(filename, str):
            raise AssertionError(
                "Input [{p}] is not a string".format(p=filename))
        if not os.path.isfile(filename):
            raise AssertionError(
                "Path [{p}] does not exist".format(p=filename))
        if not filename.split('.')[-1] == 'flo':
            raise AssertionError(
                "File extension [flo] required, [{f}] given".format(f=filename.split('.')[-1]))

        flo = open(filename, 'rb')
    else:
        flo = filename

    tag = np.frombuffer(flo.read(4), np.float32, count=1)[0]
    if not TAG_FLOAT == tag:
        raise AssertionError("Wrong Tag [{t}]".format(t=tag))

    width = np.frombuffer(flo.read(4), np.int32, count=1)[0]
    if not (width > 0 and width < 100000):
        raise AssertionError("Illegal width [{w}]".format(w=width))

    height = np.frombuffer(flo.read(4), np.int32, count=1)[0]
    if not (width > 0 and width < 100000):
        raise AssertionError("Illegal height [{h}]".format(h=height))

    n_bands = 2
    size = n_bands * width * height
    tmp = np.frombuffer(flo.read(n_bands * width * height * 4), np.float32, count=size)
    flow = np.resize(tmp, (int(height), int(width), int(n_bands)))
    flo.close()

    return array_cropper(flow, crop_window=crop_window)


def read_flow_collection(dirname: str, start_at: int = 0, num_images: int = -1,
                         crop_window: Union[int, Tuple[int, int, int, int]] = 0) -> Tuple[np.array, List[str]]:
    """
    Load a collection of .flo files.
    An example directory may look like:
        dirname
            - frame_0001.flo
            - frame_0002.flo
            - ...
    Parameters
    ----------
    dirname : str
        Directory containing .flo files.
    Returns
    -------
    flows : ndarray, shape (N, H, W, 2)
        Sequence of flow components.
    """
    pattern = re.compile('\d+')

    files = []

    allfiles = [f for f in os.listdir(dirname) if f.endswith('.flo')]
    for f in allfiles:
        match = pattern.findall(f)
        if len(match) > 0:
            frame_index = int(match[-1])
            filepath = os.path.join(dirname, f)
            files.append((frame_index, filepath))

    files = sorted(files, key=lambda x: x[0])
    files_sliced = files[start_at:] if num_images < 0 else files[start_at:start_at+num_images]

    flos, flonames = [], []
    for frame_index, filepath in files_sliced:
        flo_frame = read_flow(filepath, crop_window=crop_window)
        flos.append(flo_frame)
        flonames.append(filepath)

    flos = np.array(flos)

    return flos, flonames


def write_flow(flow: np.ndarray, filename: str, norm: bool = False):
    """
    Write a .flo file (Middlebury format).
    Parameters
    ----------
    flow : ndarray, shape (height, width, 2), dtype float32
        Flow to save to file.
    filename : str
        Filename where flow will be saved. Must have extension .flo.
    norm : bool
        Logical option to normalize the input flow or not.
    Returns
    -------
    None
    """

    assert type(filename) is str, "file is not str (%r)" % str(filename)
    assert filename[-4:] == '.flo', "file ending is not .flo (%r)" % filename[-4:]

    height, width, n_bands = flow.shape
    assert n_bands == 2, "Number of bands = %r != 2" % n_bands

    # Extract the u and v velocity
    if norm:  # use flow normalization
        u, v = _normalize_flow(flow)
    else:
        u = flow[:, :, 0]
        v = flow[:, :, 1]
        w = flow[:, :, 2] if flow.shape[2] > 2 else None

    assert u.shape == v.shape, "Invalid flow shape"
    height, width = u.shape

    with open(filename, 'wb') as f:
        tag = np.array([TAG_FLOAT], dtype=np.float32)  # assign ASCCII tag for float 202021.25 (TAG_FLOAT)
        tag.tofile(f)
        np.array([width], dtype=np.int32).astype(np.int32).tofile(f)  # assign width size to ASCII
        np.array([height], dtype=np.int32).tofile(f)  # assign height size to ASCII
        flow.tofile(f)  # assign the array value (u, v)


def quiver_plot(flow: np.ndarray, coord: Optional[np.ndarray] = None, filename: Optional[str] = None,
                norm: bool = False, show: bool = False):
    if norm:  # use flow normalization
        u, v = _normalize_flow(flow)
    else:
        u = flow[:, :, 0]
        v = flow[:, :, 1]

    # Setting up the quiver plot
    if coord is None:
        h, w = u.shape
        x = np.arange(0, w) + 0.5
        y = np.arange(0, h)[::-1] + 0.5
        xp, yp = np.meshgrid(x, y)
    else:
        xp, yp = coord[:, :, 0], coord[:, :, 1]

    # interpolate over the actual points
    # u_itp = itp.RectBivariateSpline(x, y, u)  # x and y must be a 1-D arrays of coordinates in ascending order
    # v_itp = itp.RectBivariateSpline(x, y, v)

    # ploting the result
    plt.quiver(xp, yp, u, v)
    plt.axis('equal')
    if show:
        plt.show()
    if filename is not None:
        assert type(filename) is str, "File is not str (%r)" % str(filename)
        assert filename[-4:] == '.png', "File extension is not an image format (%r)" % filename[-4:]
        plt.savefig(filename)

    plt.clf()


def vorticity_plot(vort, filename):
    pass


def motion_to_color(flow, maxmotion=None, verbose=False, original_color: bool = False):
    """
    Parameters (adopted from the original color_flow.cpp)
    ----------
    flow : ndarray, dtype float, shape (height, width, 2) OR (length, height, width, 2)
        Array of vector components. Can be either a single array or a sequence of arrays of vector components.
    maxmotion : float
        Maximum value to normalize by.
    Returns
    -------
    colim : ndarray, shape (height, width, 3) or (length, height, width, 3), dtype uint8
        Colored image.
    """
    if flow.ndim == 3:
        motim = flow[None, ...]
    else:
        motim = flow

    if motim.ndim != 4 or motim.shape[-1] != 2:
        quit('motim must be a (length, height, width, 2) array')

    length, height, width, _ = motim.shape
    colim = np.zeros((length, height, width, 3), dtype=np.uint8)

    # determine motion range
    # maxx, maxy = motim[..., 0].max(), motim[..., 1].max()
    # minx, miny = motim[..., 0].min(), motim[..., 1].min()
    fx = motim[:, :, :, 0]
    fy = motim[:, :, :, 1]
    rad = np.sqrt(fx ** 2 + fy ** 2)
    maxrad = rad.max()
    # print("max motion: {:.4f}   motion range: u = {:.3f} .. {:.3f};  v = {:.3f} .. {:.3f}".format(
    #     maxrad, minx, maxx, miny, maxy
    # ))

    if maxmotion is not None:
        maxrad = maxmotion

    if maxrad == 0:
        maxrad = 1

    if verbose:
        print("normalizing by {}".format(maxrad))

    for i in range(length):
        fx = motim[i, :, :, 0]
        fy = motim[i, :, :, 1]
        compute_color(fx / maxrad, fy / maxrad, colim[i], original_color=original_color)

    fx = motim[:, :, :, 0]
    fy = motim[:, :, :, 1]
    idx = _unknown_flow(fx, fy)
    colim[idx] = 0

    if flow.ndim == 3:
        return colim[0]

    return colim


# ---------------------- Subroutines ----------------------
def _normalize_flow(flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # UNKNOWN_FLOW = 1e10

    height, width, n_bands = flow.shape
    if not n_bands == 2:
        raise AssertionError("Image must have two bands. [{h},{w},{nb}] shape given instead".format(
            h=height, w=width, nb=n_bands))

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    # Fix unknown flow
    idx_unknown = np.where(np.logical_or(
        abs(u) > UNKNOWN_FLOW_THRESH,
        abs(v) > UNKNOWN_FLOW_THRESH
    ))
    u[idx_unknown] = 0
    v[idx_unknown] = 0

    maxu = max([-999, np.max(u)])
    maxv = max([-999, np.max(v)])
    minu = max([999, np.min(u)])
    minv = max([999, np.min(v)])

    rad = np.sqrt(np.multiply(u, u) + np.multiply(v, v))
    maxrad = max([-1, np.max(rad)])

    if flags['debug']:
        print("Max Flow : {maxrad:.4f}. Flow Range [u, v] -> [{minu:.3f}:{maxu:.3f}, {minv:.3f}:{maxv:.3f}] ".format(
            minu=minu, minv=minv, maxu=maxu, maxv=maxv, maxrad=maxrad
        ))

    eps = np.finfo(np.float32).eps
    u = u / (maxrad + eps)
    v = v / (maxrad + eps)

    return u, v


def _unknown_flow(u: np.ndarray, v: np.ndarray):
    """
    Adopted from the original flowIO.cpp
    :param u: x-axis flow
    :param v: y-axis flow
    :return:
    """
    return (np.fabs(u) > UNKNOWN_FLOW_THRESH) | (np.fabs(v) > UNKNOWN_FLOW_THRESH) | np.isnan(u) | np.isnan(v)


# ---------------------- UTILITIES ----------------------
def flowname_modifier(indir: str, outdir: str, ext: str = '_out.flo', pair: bool = True) -> str:
    out_name = os.path.splitext(os.path.basename(indir))[0]
    if pair:
        out_name = str(out_name.rsplit('_', 1)[0]) + ext
    else:
        out_name += ext

    out_name = os.path.join(outdir, out_name)
    return out_name


def resize_flow(flow, des_width, des_height, method='bilinear'):
    """Utility function to resize the flow array, used by RandomScale transformer.
    WARNING: improper for sparse flow!
    Args:
        flow: the flow array
        des_width: Target width
        des_height: Target height
        method: interpolation method to resize the flow
    Returns:
        the resized flow
    """
    src_height = flow.shape[0]
    src_width = flow.shape[1]

    if src_width == des_width and src_height == des_height:  # Sanity check, if resizing is a necessary
        return flow

    ratio_height = float(des_height) / float(src_height)
    ratio_width = float(des_width) / float(src_width)

    if method == 'bilinear':
        flow = cv2.resize(flow, (des_width, des_height), interpolation=cv2.INTER_LINEAR)
    elif method == 'nearest':
        flow = cv2.resize(flow, (des_width, des_height), interpolation=cv2.INTER_NEAREST)
    else:
        raise Exception('Invalid resize flow method!')

    flow[:, :, 0] = flow[:, :, 0] * ratio_width
    flow[:, :, 1] = flow[:, :, 1] * ratio_height

    return flow


def horizontal_flip_flow(flow):
    flow = np.copy(np.fliplr(flow))
    flow[:, :, 0] *= -1
    return flow


def vertical_flip_flow(flow):
    flow = np.copy(np.flipud(flow))
    flow[:, :, 1] *= -1
    return flow


def array_cropper(array, crop_window: Union[int, Tuple[int, int, int, int]] = 0):
    # Cropper init.
    s = array.shape  # Create image cropper
    crop_window = (crop_window,) * 4 if type(crop_window) is int else crop_window
    assert len(crop_window) == 4

    # Cropping the array
    return array[crop_window[0] : s[0]-crop_window[1], crop_window[2] : s[1]-crop_window[3]]


# ---------------------- TESTING ----------------------
if __name__ == '__main__':
    tic = time.time()
    print('START!')

    flodir = '../images/test_images'
    floname = glob(os.path.join(flodir, '*.flo'))

    if isinstance(floname, list):
        for floi in floname:
            if os.path.isfile(floi):
                tic_in = time.time()
                vec_flow = read_flow(floi)
                tmpname, _ = floi.rsplit('.', 1)

                # quiver plot
                qname = tmpname + '_quiver.png'
                quiver_plot(vec_flow, filename=qname, show=False)

                # save with different name
                new_floi = tmpname + '_tmp.flo'
                # write_flow(vec_flow, new_floi)

                toc_in = time.time()
                print(f'\tInner processing time: {float("{0:.2f}".format(toc_in - tic_in))} s')

            else:
                raise AssertionError(f'File {floi} is not found!')

    else:
        raise AssertionError('Wrong input!')

    toc = time.time()
    print(f'Done with total processing time: {float("{0:.2f}".format(toc - tic))} s')
