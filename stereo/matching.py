import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib
from matplotlib import pyplot as plt

import os, sys
import argparse
import json

from stereo.dewarp import Guess, map_coeff, warp
# matplotlib.use('Qt5Agg')


# ------------------ CLI ------------------
parser = argparse.ArgumentParser(description='2D reconstruction method for Stereoscopic PIV calibration')
parser.add_argument('--root', '-r', default='./imgs', type=str, help='root directory for the input images')
parser.add_argument('--name', '-n', default='30-5_0', type=str, help='stereo image input names')
parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')

main_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(main_dir)


def gen_template(TC: int = 5, HC: int = 25, LC: int = 25) -> np.array:
    """
    Generate template image.
    Args:
        TC  : (int) cross thickness
        HC  : (int) cross global height
        LC  : (int) cross global length/width
    Returns:
        temp;ate    : (np.array) template image, in greyscale uint8 format.
    """
    LC2, HC2, TC2 = np.ceil(LC / 2), np.ceil(HC / 2), np.floor(TC / 2)
    template = np.zeros([HC, LC])

    if TC2 != np.ceil(TC / 2):  # if TC is odd.
        template[np.int(HC2 - TC2 - 1):np.int(HC2 + TC2), 0:np.int(LC)] = np.ones([TC, LC])  # horizontal cross.
        template[0:np.int(HC), np.int(LC2 - TC2 - 1):np.int(LC2 + TC2)] = np.ones([HC, TC])  # vertical cross.
    else:  # if TC is even.
        template[HC2 - TC2 - 1:HC2 + TC2 - 1, 0:LC] = np.ones([TC, LC])  # horizontal cross.
        template[0:HC, LC2 - TC2 - 1:LC2 + TC2] = np.ones([HC, TC])  # vertical cross.

    template = np.array(template * 255, dtype=np.uint8)

    return template


def template_matching(gray_img: np.array, template: np.array, threshold: float = 0.0) -> np.array:
    """
    Perform template matching between the main image (gray_img) and the template image.
    Args:
        gray_img    : (np.array) bitwise main image.
        template    : (np.array) bitwise template image.
        threshold   : (float) threshold value for the correlation map.
    Returns:
        Processed correlation map.
    """
    # Padding init.
    pad = [int((template.shape[0] - 1) / 2), int((template.shape[1] - 1) / 2)]
    pad_gray = np.zeros([gray_img.shape[0] + 2 * pad[0], gray_img.shape[1] + 2 * pad[1]], dtype=np.uint8)

    pad_gray[pad[0]:-pad[0], pad[1]:-pad[1]] = gray_img
    res = cv2.matchTemplate(pad_gray, template, cv2.TM_CCOEFF_NORMED)

    # Threshold: 0.0 (without threshold); 0.7 (default config.)
    thres_mask = res > threshold
    res_thres = res * thres_mask

    # Add gaussian blur
    result = cv2.blur(res_thres, (2, 2))

    return result


def findLocalMax(image: np.array):
    """
    Finding local maxima from dots pattern.
    Args:
        image   : (np.array) W, H post-processed correlation map image.
    Returns:
        coords  : (np.array) N, 2 array with (x, y) config.
    """
    lbl = ndimage.label(image)
    points = ndimage.measurements.center_of_mass(
        image, lbl[0], [i + 1 for i in range(lbl[1])]
    )

    # Change into (x, y) format!
    coord = np.fliplr(np.array(points))

    return coord


def select_ref(coords):
    """
    Choosing the 4 nearest reference points
    using Left - Right - Down - Left (L-R-D-L) direction.
    Args:
        select_ref  : (N, 2) The original coordinate points.
    """
    n_point_ref = 4
    points_selected = []

    for i in range(n_point_ref):
        point_ref = plt.ginput(1, timeout=-1, show_clicks=True)[0]
        print(f'\t{i+1}. Clicked at {point_ref}')

        # in (x, y) coordinate format
        distance = np.linalg.norm(coords - np.array(point_ref), axis=1)
        min_distance_arg = np.argmin(distance)
        plt.plot(coords[min_distance_arg, 0], coords[min_distance_arg, 1], 'yo')  # Plotting each selected point

        points_selected.append(min_distance_arg)

    # Plotting the line
    points_ref = np.zeros([4, 2])
    for i in range(n_point_ref):
        j = i + 1 if i < n_point_ref - 1 else 0
        i_pt, j_pt = points_selected[i], points_selected[j]

        plt.plot([coords[i_pt, 0], coords[j_pt, 0]],
                 [coords[i_pt, 1], coords[j_pt, 1]], 'r-')

        points_ref[i, :] = coords[i_pt, :]

    # Calculating the center point
    c_x = (np.abs(points_ref[1, 0] - points_ref[0, 0]) + np.abs(points_ref[3, 0] - points_ref[2, 0])) * 0.5
    c_y = (np.abs(points_ref[3, 1] - points_ref[0, 1]) + np.abs(points_ref[2, 1] - points_ref[1, 1])) * 0.5

    c_point = [c_x, c_y]
    # c_point = points_ref.mean(axis=0)
    return points_ref, points_selected, c_point


def read_image(root: str, name: str):
    """
    Utility function to read the input stereo images name,
    given the respective root path and basename.
    Args:
        root    : (str) the respective root directory.
        name    : (str) stereo image basename, without specifying either from left or right camera.
    Returns:
        imnames : (List) List of the left and right image names
    """
    assert os.path.isdir(root)

    exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.ppm']
    imnames = []

    for ext in exts:
        imname = [os.path.join(root, f"{name}{idcam}{ext}") for idcam in ['-L', '-R']]

        if os.path.isfile(imname[0]) and os.path.isfile(imname[1]):
            imnames.extend(imname)
            break

    assert len(imnames) == 2
    print("Reading stereo calibration images from:")
    [print(f"\t{idcam.upper()}\t: {imnames[i]}") for i, idcam in enumerate(['Left', 'Right'])]

    return imnames


def write_coeff(save: str, name: str, coeffdict: dict):
    """
    Utility function to write the stereo mapping coefficients.
    Args:
        save        : (str) the respective saving directory.
        name        : (str) stereo image basename, without specifying either from left or right camera.
        coeffdict   : (dict) left and right mapping coefficients, stored as a dictionary.
    Returns:
        write the mapping coefficients.
        coeffdf : (pd.DataFrame) coeffdict in DataFrame format
    """
    assert len(coeffs) == 2  # Mapping coefficients for left and right camera
    os.makedirs(save) if not os.path.isdir(save) else None

    coeffname = os.path.join(save, f"{name}.json")
    coeffdf = pd.DataFrame(coeffdict)

    with open(coeffname, 'w') as fp:
        json.dump(coeffdict, fp, indent=4)

    print(f"\nWriting the mapping coefficients to {coeffname}")
    print(coeffdf)

    return coeffdf


if __name__ == '__main__':
    # -------------------- Debugging mode here --------------------
    debug_input = [
        'matching.py',
        '--root', './imgs',
        '--name', '30-5_0',
        '--save', './outputs'
    ]
    # sys.argv = debug_input  # Uncomment for debugging

    # -------------------- INPUT Init. --------------------
    args = parser.parse_args()
    imnames = read_image(args.root, args.name)
    images = [cv2.imread(imname) for imname in imnames]
    imshape = images[0].shape[:2]

    # Template input
    template = gen_template(TC=5, HC=25, LC=25)

    # Storing init.
    gray_images, matched_imgs, dewarped_images = [], [], []
    old_coords, new_coords = [], []
    naming = ['Left', 'Right']
    coeffs = {}

    # Iterate over the left and right camera images to guess the calibration coord.
    fig, axes = plt.subplots(1, 2)
    for i, image in enumerate(images):
        gray = cv2.bitwise_not(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        matched_img = template_matching(gray, template, threshold=0.7)
        coord = findLocalMax(matched_img)

        # Plot the results
        plt.subplot(axes[i])
        plt.xlim((0, imshape[1]))
        plt.ylim((imshape[0], 0))
        plt.imshow(gray, cmap='gray')
        plt.plot(coord[:, 0], coord[:, 1], 'r.')

        # Results gathering
        gray_images.append(gray)
        matched_imgs.append(matched_img)
        old_coords.append(coord)

    # Select 4 reference points to find the dewarping coordinates!
    for i, old_pts in enumerate(old_coords):
        title = f"\nChoose 4 reference points from the {naming[i].upper()} camera!"
        print(title)

        plt.subplot(axes[i])
        axes[i].title.set_text(title)
        ref_points, select_points, c_points = select_ref(old_pts)

        # Perform dewarping using perspective transform
        # new_pts = dewarping(old_pts, ref_points, c_points)
        new_pts = Guess(old_pts, c_points, select_points[0])()

        # Calculating the mapping coefficient
        A = map_coeff(old_pts, new_pts, select_points[0])
        dewarped_img = warp(matched_imgs[i], old_pts, select_points[0], A)

        # Results gathering
        new_coords.append(new_pts)
        dewarped_images.append(dewarped_img)
        coeffs[naming[i]] = A.tolist()

    plt.show()

    # Plot the old and new coordinates
    fig, axes = plt.subplots(1, 2)
    for i, ax in enumerate(axes):
        plt.subplot(ax)
        ax.set_title(f"Guessed coordinate result for the {naming[i].upper()} camera")
        plt.xlim((0, imshape[1]))
        plt.ylim((imshape[0], 0))
        plt.plot(old_coords[i][:, 0], old_coords[i][:, 1], 'cx')
        plt.plot(new_coords[i][:, 0], new_coords[i][:, 1], 'rx')

    plt.show()

    # Plot the calibrated result
    plt.figure()
    dot_type = ['cx', 'rx']
    for i, dot in enumerate(dot_type):
        plt.xlim((0, imshape[1]))
        plt.ylim((imshape[0], 0))

        plt.subplot(1, 2, 1).set_title(f"Dewarped coordinate results")
        plt.plot(new_coords[i][:, 0], new_coords[i][:, 1], dot)
        plt.legend(naming)

    plt.subplot(1, 2, 2).set_title(f"Dewarped image")
    plot_dewarped = np.zeros([imshape[0], imshape[1], 3])
    plot_dewarped[:, :, 0], plot_dewarped[:, :, 2] = dewarped_images[0], dewarped_images[1]
    plt.imshow(dewarped_images[0])

    plt.show()

    coeffdf = write_coeff(args.save, args.name, coeffs)
    print("DONE!")
