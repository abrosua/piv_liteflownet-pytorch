import numpy as np
from tqdm import tqdm

import os, sys
import argparse
from glob import glob
import json

from stereo.dewarp import nl_trans
from stereo.vel3d import willert
from inference import Inference
from src.utils_plot import read_flow, write_flow


# ------------------ CLI ------------------
parser = argparse.ArgumentParser(description='Stereoscopic PIV image processing')
parser.add_argument('--coeff', '-c', type=str, help='mapping coefficient json file path.')
parser.add_argument('--root', '-r', default='./images/demo', type=str, help='root directory for series of images')
parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')

parser.add_argument('--theta', default=[45.0, 45.0], type=float, nargs='+', help='object plane angle')
parser.add_argument('--alpha', default=[0.0, 0.0], type=float, nargs='+',
                    help='scheimpflug criterion, image plane angle')
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

def direct_process(args, net, device: str = 'cpu'):
    pass


def manual_process(args, net, device: str = 'cpu'):
    # Instatiate inferencing object
    infer = Inference(net, output_dir=args.save, device=device)
    coeffdict = read_coeff(args.coeff)
    beta = [a for a in args.alpha]

    # Performing PIV for left and right images
    stereo_inputs = [x[0] for x in os.walk(args.root)
                     if os.path.basename(x[0]) != os.path.basename(args.root)]
    for path in stereo_inputs:
        infer.dataloader_parsing(path, pair=False, write=True)

    # Iterate for calculating stereo results
    naming = ['left', 'right']
    left_flos = sorted(glob(os.path.join(args.save, os.path.join(f"*{naming[0]}*", "*.flo"))))
    right_dir = glob(os.path.join(args.save, f"*{naming[1]}*"))[0]

    for left_flo in tqdm(left_flos, ncols=100, leave=True, unit='flo', desc=f'Evaluating {args.save}'):
        # Init.
        flobase = os.path.basename(left_flo)
        right_flo = os.path.join(right_dir, flobase)

        # Generate the flow array
        flow = [read_flow(left_flo), read_flow(right_flo)]
        cal_flow = [np.dstack(nl_trans(flo[:, :, 0], flo[:, :, 1], coeffdict[naming[i].capitalize()]))
                    for i, flo in enumerate(flow)]
        stereo_flow = willert(cal_flow, args.theta, beta)
        write_flow(stereo_flow, os.path.join(args.save, "stereo", flobase))


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
