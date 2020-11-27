import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd

import os, sys
import json
import argparse

from stereo.matching import gen_template, template_matching, findLocalMax, select_ref
from stereo.dewarp import Guess, map_coeff, warp
# matplotlib.use('Qt5Agg')


# ------------------ CLI ------------------
parser = argparse.ArgumentParser(description='2D reconstruction method for Stereoscopic PIV calibration')
parser.add_argument('--root', '-r', default='./imgs', type=str, help='root directory for the input images')
parser.add_argument('--name', '-n', default='30-5_0', type=str, help='stereo image input names')
parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')

main_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(main_dir)


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
        coeffdict   : (dict) left and right mapping coefficients.
        calibdict   : (dict) left and right calibration point.
    Returns:
        write the mapping coefficients.
        coeffdf : (pd.DataFrame) stereo coefficient and calibration report in DataFrame format.
    """
    # assert len(coeffdict) == 2  # Mapping coefficients for left and right camera
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
        'stero_cal.py',
        '--root', './images/test-stereo',
        '--name', '30-5',
        '--save', './test-output/PIV-LiteFlowNet-en/test-stereo'
    ]
    sys.argv = debug_input  # Uncomment for debugging

    # -------------------- INPUT Init. --------------------
    args = parser.parse_args()
    imnames = read_image(args.root, args.name)
    images = [cv2.imread(imname) for imname in imnames]
    imshape = images[0].shape[:2]

    # Template input
    template = gen_template(TC=11, HC=25, LC=25)

    # Storing init.
    gray_images, matched_imgs, dewarped_images = [], [], []
    old_coords, new_coords, calib_x = [], [], []
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

        # Calibration point
        calib_x.append(np.abs(new_pts[select_points[0], 0] - new_pts[select_points[1], 0]))

        # Calculating the mapping coefficient
        A = map_coeff(old_pts, new_pts, select_points[0])
        dewarped_img = warp(matched_imgs[i], old_pts, select_points[0], A)

        # Results gathering
        new_coords.append(new_pts)
        dewarped_images.append(dewarped_img)
        coeffs[naming[i]] = A.tolist()

    coeffs["calib"] = np.mean(calib_x)
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
