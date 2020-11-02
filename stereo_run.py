import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

import os, sys
import argparse
from glob import glob
from typing import List, Optional
import json

from stereo.dewarp import nl_trans
from stereo.vel3d import willert
from inference import Inference, estimate
from src.utils_plot import read_flow, write_flow
from src.models import piv_liteflownet
from src.datasets import InferenceRun


# ------------------ CLI ------------------
parser = argparse.ArgumentParser(description='Stereoscopic PIV image processing')
parser.add_argument('--coeff', '-c', type=str, help='mapping coefficient json file path.')
parser.add_argument('--root', '-r', default=None, type=str, help='root directory for series of images')
parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')

parser.add_argument('--theta', default=[45.0, 45.0], type=float, nargs='+', help='object plane angle')
parser.add_argument('--alpha', default=[0.0, 0.0], type=float, nargs='+',
                    help='scheimpflug criterion, image plane angle')
parser.add_argument('--window-size', '-ws', default=[1.0, 1.0], type=float, nargs='+',
                    help="Window size in the real length")
parser.add_argument('--fps', default=1, type=int, help="camera frame rate (FPS).")
parser.add_argument('--calib', default=None, type=float, help="real length calibration in meters (m).")

parser.add_argument('--model', default="./models/pretrain_torch/PIV-LiteFlowNet-en.paramOnly", type=str,
                    help="model weight parameters to use")
parser.add_argument('--model_version', default=1, type=int, choices=[1, 2],
                    help="choose which base model version to use, LiteFlowNet or LiteFlowNet2")
parser.add_argument('--inference_mode', default='manual', type=str, choices=['manual', 'direct'],
                    help="choose which inference method to use")

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
    # Init.
    coeffdict = read_coeff(args.coeff)
    beta = [a for a in args.alpha]

    # Check calibration point
    if "calib" in coeffdict.keys():
        calib = args.calib / coeffdict["calib"] if args.calib else None
    else:
        calib = None

    # Instantiate dataloader
    stereo_dataset = InferenceRun(root=args.root, pair=False, use_stereo=True)
    stereo_dataloader = DataLoader(stereo_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    # Processing dataloader
    for images, floname in tqdm(stereo_dataloader, ncols=100, leave=True, unit='pair', desc=f'Evaluating {args.root}'):
        images = [image.to(device) for image in images]  # Add to device

        flow_cal = [_stereo_cal(estimate(net, images[i], images[i+1], tensor=False),
                                coeffdict[naming.capitalize()],
                                args.fps,
                                calib)
                    for i, naming in enumerate(['left', 'right'])
                    ]
        stereo_flow = willert(flow_cal, args.theta, beta)

        flosave = str(floname[0]) + '_2d3c.flo'
        write_flow(stereo_flow, os.path.join(args.save, "stereo", flosave))


def manual_process(args, net, device: str = 'cpu'):
    # Instatiate inferencing object
    infer = Inference(net, output_dir=args.save, device=device)

    # Performing PIV for left and right images
    stereo_inputs = [x[0] for x in os.walk(args.root)
                     if os.path.basename(x[0]) != os.path.basename(args.root)]
    for path in stereo_inputs:
        infer.dataloader_parsing(path, pair=False, write=True)

    # Iterate for calculating stereo results
    _flo_process(args)


def _flo_process(args):
    # Init.
    coeffdict = read_coeff(args.coeff)
    beta = [a for a in args.alpha]
    naming = ['left', 'right']

    # Check calibration point
    if "calib" in coeffdict.keys():
        calib = args.calib / coeffdict["calib"] if args.calib else None
    else:
        calib = None

    left_flos = sorted(glob(os.path.join(args.save, naming[0], "*.flo")))
    right_dir = os.path.join(args.save, naming[1])

    for left_flo in tqdm(left_flos, ncols=100, leave=True, unit='flo', desc=f'Evaluating {args.save}'):
        # Init.
        flobase = os.path.basename(left_flo).rsplit("-", 1)[0]
        right_flo = os.path.join(right_dir, flobase + "-R_out.flo")

        # Checking the file(s) availability
        assert os.path.isfile(left_flo) and os.path.isfile(right_flo)

        # Generate the flow array
        flow_cal = [_stereo_cal(read_flow(floname),
                                coeffdict[naming[i].capitalize()],
                                args.fps,
                                calib)
                    for i, floname in enumerate([left_flo, right_flo])
                    ]
        stereo_flow = willert(flow_cal, args.theta, beta)

        flosave = flobase + '_2d3c.flo'
        write_flow(stereo_flow, os.path.join(args.save, "stereo", flosave))


def _stereo_cal(flow, A, fps: float, calibrate: Optional[List[float]] = None):
    # Stereo calibration
    flow_cal = nl_trans(flow[:, :, 0], flow[:, :, 1], A)
    flow_stereo = np.dstack(flow_cal)

    # Real flow calibration
    # TODO: CHECK the spatial calibration due to the Stereoscopic recons!
    if calibrate:
        flow_stereo = flow_stereo * calibrate * fps  # meters / second

    return flow_stereo


if __name__ == "__main__":
    # -------------------- Debugging mode here --------------------
    debug_input = [
        'stereo.py',
        '--coeff', './outputs/30-5_0.json',
        '--root', 'null',
        '--calib', 'null',
    ]
    sys.argv = debug_input  # Uncomment for debugging

    # -------------------- INPUT Init. --------------------
    args = parser.parse_args()

    # START here
    if args.root is None:
        _flo_process(args)
    else:
        # Init.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if os.path.isfile(args.model):
            params = torch.load(args.model)
        else:
            raise ValueError(f'Unknown model params input ({args.model})!')
        net = piv_liteflownet(params, args.model_version).to(device)

        assert os.path.isdir(args.root)
        manual_process(args, net, device) if args.inference_mode is "manual" else direct_process(args, net, device)
