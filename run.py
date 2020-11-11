import os
import re
import sys
import argparse
import colorama
import setproctitle
from glob import glob
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from inference import Inference, estimate
from src import utils
from src.models import piv_liteflownet, hui_liteflownet
from src.utils_plot import write_flow, flowname_modifier
from src.datasets import Run


# ------------------ CLI ------------------
parser = argparse.ArgumentParser(description='Inferencing script for LiteFlowNet')

parser.add_argument("--start", "-s", type=int, default=0, help="Input image starting index.")
parser.add_argument("--num_images", "-n", type=int, default=-1,
                    help="Number of image(s) to process from the directory.")
parser.add_argument("--is_pair", "-p", action="store_true", help="To check if the input image format is in pair.")

parser.add_argument("--model", "-m", type=str, choices=["hui", "piv"],
                    help="Select which model to solve the problem!")
parser.add_argument("--version", "-v", type=int, choices=[1, 2], default=1,
                    help="Select the LiteFlowNet model backbone version (i.e., LiteFlowNet or LiteFlowNet2)!")
parser.add_argument("--input", "-i", default=["./images/demo"], type=str, nargs="+",
                    help="Input images directory(ies).")
parser.add_argument("--output", "-o", default="./results", type=str, help="Main output directory.")
parser.add_argument("--no_cuda", action="store_true")

main_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(main_dir)


def getpair(inputdir: str, n_images: int = 2, start_at: int = 0,
            extensions: Tuple[str] = ('jpg', 'jpeg', 'png', 'bmp', 'tif', 'ppm')):
    """
    Subroutine to gather input image(s)
    :param inputdir: Input image file directory.
    :param n_images: Number of image(s) to process.
    :param start_at: Starting index of the input image.
    :param extensions: Image extension.
    :return:
    """
    assert n_images != 1
    assert os.path.isdir(inputdir)

    img_files = []
    for ext in extensions:
        img_files += sorted(glob(os.path.join(inputdir, f'*.{ext}')))

    if n_images < 0:
        return img_files[start_at:]
    else:
        return img_files[start_at : start_at+n_images]


def get_weights(modelpath):
    """
    Subroutine to obtain the weights from the param path.
    :param modelpath: Path to the weights file.
    :return:
    """
    if os.path.isfile(modelpath):
        weights = torch.load(modelpath)
        netname = os.path.splitext(os.path.basename(modelpath))[0]
    else:
        raise ValueError('Unknown params input!')

    return weights, netname


def main(net, inputdir: str, savedir: str, start_id: int = 0, num_images: int = -1, device: str = "cpu"):
    """
    Main function for motion estimator inference.
    :param net: The model object.
    :param inputdir: The directory to the input image(s).
    :param savedir: The directory target to save the flo file(s).
    :param start_id: Starting index of the processed image.
    :param num_images: Number of image(s) to process.
    :param device: Select the processing device(s).
    :return:
    """

    # Init.
    imnames = getpair(inputdir, n_images=num_images, start_at=start_id)  # Getting the image name(s)
    os.makedirs(savedir) if not os.path.isdir(savedir) else None  # Checking the save directory

    out_names = []
    prev_frame = None

    for curr_frame in tqdm(imnames, ncols=100, leave=True, unit='pair', desc=f'Evaluating {inputdir}'):
        if prev_frame is not None:
            out_flow = Inference.parser(net,
                                        Image.open(prev_frame).convert('RGB'),
                                        Image.open(curr_frame).convert('RGB'),
                                        device=device)
            # Post-processing here
            out_name = flowname_modifier(prev_frame, savedir, pair=False)
            write_flow(out_flow, out_name)
            out_names.append(out_name)

        prev_frame = curr_frame
    tqdm.write(f'Finish processing all images from {inputdir} path!')


def main_dl(net, inputdir: str, savedir: str, is_pair: bool = False, start_id: int = 0, num_images: int = -1,
            device: str = "cpu"):
    """
    Main function for motion estimator inference with PyTorch's DataLoader.
    :param net: The model object.
    :param inputdir: The directory to the input image(s).
    :param savedir: The directory target to save the flo file(s).
    :param start_id: Starting index of the processed image.
    :param num_images: Number of image(s) to process.
    :param device: Select the processing device(s).
    :return:
    """

    # Init.
    os.makedirs(savedir) if not os.path.isdir(savedir) else None  # Checking the save directory

    # Dataset preparation
    infer_dataset = Run(root=inputdir, is_pair=is_pair, n_images=num_images, start_at=start_id)
    infer_dataloader = DataLoader(infer_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    infer_size = len(infer_dataset)
    print(f"Processing {infer_size} pairs of images...")

    for images, imname in tqdm(infer_dataloader, ncols=100, leave=True, unit='pair', desc=f'Evaluating {inputdir}'):
        # Model inference
        images[0], images[1] = images[0].to(device), images[1].to(device)
        out_flow = estimate(net, images[0], images[1], tensor=False)

        # Writing the output files
        out_name = flowname_modifier(imname, savedir, pair=is_pair)
        write_flow(out_flow, out_name)

    tqdm.write(f'Finish processing all images from {inputdir} path!')


if __name__ == '__main__':
    # Debugging test
    debug_input = [
        "run.py",
        "--start", "0", "--num_images", "-1",
        "--model", "piv", "--version", "1",
        "--input", "./images/test",
        "--output", "./test-output",
    ]
    sys.argv = debug_input  # Uncomment for debugging

    # ------------------------------ PARSING THE INPUT ------------------------------
    # Parse the official arguments
    with utils.TimerBlock("Parsing Arguments") as block:
        log_args = {}
        args = parser.parse_args()

        # Get argument defaults (hastag #thisisahack)
        parser.add_argument('--IGNORE', action='store_true')
        defaults = vars(parser.parse_args(['--IGNORE']))

        # Print all arguments, color the non-defaults. Also prepare for the parameters logger
        for argument, value in sorted(vars(args).items()):
            reset = colorama.Style.RESET_ALL
            color = reset if value == defaults[argument] else colorama.Fore.MAGENTA
            block.log('{}{}: {}{}'.format(color, argument, value, reset))

            if not bool(re.search('logger', argument)):
                log_args[argument] = value

    # Setting up the model
    with utils.TimerBlock(f"Building '{args.model}' model with backbone version = {args.version}") as block:
        # Setting up the processing device(s)
        if torch.cuda.is_available() and not args.no_cuda:
            block.log('Initializing CUDA...')
            device = 'cuda'
        else:
            block.log('CUDA is NOT being used!')
            device = 'cpu'

        # Choosing the model to use
        backbone = "LiteFlowNet" if args.version == 1 else f"LiteFlowNet{args.version}"

        if args.model == "hui":
            block.log(f"Generating Hui-{backbone} model...")
            args_model = "models/pretrain_torch/Hui-LiteFlowNet.paramOnly"
            args.params = os.path.join(main_dir, args_model)
            weights, netname = get_weights(modelpath=args.params)
            net = hui_liteflownet(weights, version=args.version).to(device)
        elif args.model == "piv":
            block.log(f"Generating PIV-{backbone} model...")
            args_model = "models/pretrain_torch/PIV-LiteFlowNet-en.paramOnly"
            args.params = os.path.join(main_dir, args_model)
            weights, netname = get_weights(modelpath=args.params)
            net = piv_liteflownet(weights, version=args.version).to(device)
        else:
            raise ValueError(f"Unknown model parameter at '{args.model}'! Choose between 'hui' and 'piv' only!")

    # Multiple input directory processing
    imdirs = args.input
    for i, imdir in enumerate(imdirs):
        print(f"---------- Processing images from directory #{str(i).zfill(2)}: '{imdir}'")
        args.input = imdir

        # Setting up output directory
        with utils.TimerBlock(f"Setting up output directory #{str(i).zfill(2)}") as block:
            # Name checking
            is_all_flow = (args.start == 0) and (args.num_images < 0)
            num_images = "end" if args.num_images < 0 else args.num_images

            # Check basename
            checkname = os.path.basename(args.input)
            if checkname.lower() in ["left", "right"]:  # For stereoscopic images
                extradir = checkname.lower()
                bname = os.path.basename(os.path.dirname(args.input))
            else:
                extradir = None
                bname = checkname

            outsubdir = f"{bname}-{args.start}_{num_images}" if not is_all_flow else bname
            args.save = os.path.join(args.output, netname, outsubdir)
            flodir = os.path.join(args.save, "flow") if extradir is None else os.path.join(args.save, "flow", extradir)
            args.saveflo = flodir

            # Setting up the logger
            block.log(f"Initializing save directory #{str(i).zfill(2)}: {args.save}")
            os.makedirs(args.save) if not os.path.exists(args.save) else None  # Create the save directory

            # Setting up the metadata filename
            argsname = "args.txt" if extradir is None else f"args_{extradir}.txt"
            log_file = os.path.join(args.save, argsname)

        # Saving the log file
        for argument, value in sorted(vars(args).items()):
            block.log2file(log_file, '{}: {}'.format(argument, value))

        # Main script
        main(net=net, start_id=args.start, num_images=args.num_images, inputdir=imdir, savedir=flodir, device=device)
        # main_dl(net=net, is_pair=args.is_pair, start_id=args.start, num_images=args.num_images, inputdir=imdir, savedir=flodir, device=device)
