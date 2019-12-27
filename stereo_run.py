import numpy as np

import os, sys
import argparse
import json

from stereo.dewarp import nl_trans


def willert(u, v, theta, beta):
    # Using method from Willert, 1997

    #Theta is off-axis half angle
    #Beta is off-axis half angle in y-z plane

    # *note:
    #   argument 0 for Left Camera, and 1 for Right Camera

    U = (u[1] * np.tan(theta[0]) - u[0] * np.tan(theta[1])) / (np.tan(theta[0]) - np.tan(theta[1]))
    V = (v[0] + v[1]) / 2 + (u[1] - u[0]) * (np.tan(beta[1]) - np.tan(beta[0]))/(np.tan(theta[0]) - np.tan(theta[1])) / 2
    W = (u[1] - u[0]) / (np.tan(theta[0]) - np.tan(theta[1]))

    return U, V, W


# ------------------ CLI ------------------
parser = argparse.ArgumentParser(description='Stereoscopic PIV image processing')
parser.add_argument('--coeff', '-c', type=str, help='mapping coefficient json file path.')
parser.add_argument('--root', '-r', default='./images/demo', type=str, help='root directory for the series of input images')
parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')

parser.add_argument('--window-size', '-ws', default=[1.0, 1.0], type=float, nargs='+',
                    help="Window size in the real length")
parser.add_argument('--fps', default=60, type=int, help="camera frame rate (FPS)")

main_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(main_dir)


def read_coeff(path: str):
    """
    Utility function to read the mapping coefficient json file.
    Args:
        path    : (str) json file path
    Returns:
        list of stereo mapping coefficient, for both left and right camera view.
    """
    assert os.path.isfile(path)

    with open(path) as fp:
        coeffdict = json.load(fp)

    return coeffdict


if __name__ == "__main__":
    # -------------------- Debugging mode here --------------------
    debug_input = [
        'stereo.py',
        '--coeff', './outputs/30-5_0.json',
        '--root', '',
    ]
    sys.argv = debug_input  # Uncomment for debugging

    # -------------------- INPUT Init. --------------------
    args = parser.parse_args()
    coeffdict = read_coeff(args.coeff)
    assert os.path.isdir()

    # Init. variable
    time_frame = 1 / args.fps  # second

    # START here
    for key, coeff in coeffdict.items():

        # PIV processing

        # Real length calibration
        length_cal = np.array(args.window_size) / np.array(flow.shape)
        flow_cal = [flow[:, :, i] * len_cal for i, len_cal in enumerate(length_cal.tolist())]

        # Stereo calibration
        calibrate_flow = nl_trans(flow_cal[0], flow_cal[1], coeff)
