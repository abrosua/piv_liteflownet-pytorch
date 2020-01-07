#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import math
import cv2
from imutils.video import FileVideoStream, WebcamVideoStream
import numpy as np
import PIL.Image

import sys, os
import getopt
import time
from tqdm import tqdm

from src.models import piv_liteflownet, hui_liteflownet
from src.utils_data import image_files_from_folder
from src.utils_plot import quiver_plot, write_flow, flowname_modifier
from src.datasets import InferenceRun

# INPUT
args_model = './models/torch/Hui-LiteFlowNet.paramOnly'
args_img1 = './images/first.png'
args_img2 = './images/second.png'
args_output = './out.flo'

##########################################################

def estimate(net: torch.nn.Module, img1: torch.Tensor, img2: torch.Tensor, tensor: bool = False):
	# Ensure that both the first and second images have the same dimension!
	assert (img1.size(2) == img2.size(2))
	assert (img1.size(3) == img2.size(3))

	input_width = img1.size(3)
	input_height = img1.size(2)

	# Adaptive width and height
	adaptive_width = int(math.floor(math.ceil(input_width / 32.0) * 32.0))
	adaptive_height = int(math.floor(math.ceil(input_height / 32.0) * 32.0))

	# Scale factor
	scale_width = float(input_width) / float(adaptive_width)
	scale_height = float(input_height) / float(adaptive_height)

	tensor_im1 = torch.nn.functional.interpolate(input=img1, size=(adaptive_height, adaptive_width),
													mode='bilinear', align_corners=False)
	tensor_im2 = torch.nn.functional.interpolate(input=img2, size=(adaptive_height, adaptive_width),
													mode='bilinear', align_corners=False)

	# make sure to not compute gradients for computational performance
	with torch.set_grad_enabled(False):
		net.eval()
		tensor_raw_output = net(tensor_im1, tensor_im2)

	# Interpolate the flow result back to the desired input size
	tensor_flow = torch.nn.functional.interpolate(input=tensor_raw_output, size=(input_height, input_width),
													mode='bilinear', align_corners=False)

	tensor_flow[:, 0, :, :] *= scale_width
	tensor_flow[:, 1, :, :] *= scale_height

	if tensor:
		return tensor_flow.detach()
	else:
		output_flow = torch.squeeze(tensor_flow).permute(1, 2, 0).detach().cpu().numpy()
		return output_flow


class Inference:
	def __init__(self, net, netname=None, output_dir='./outputs', device='cpu'):
		if netname is None:
			self.netname = 'test'
		else:
			self.netname = os.path.splitext(os.path.basename(netname))[0]

		self.default = os.path.join(output_dir, self.netname)
		self.device = device if torch.cuda.is_available() else 'cpu'
		self.net = net

	def video_parsing(self, vidfile=0, write: bool = True) -> None:
		# Define the video type (offline or direct stream)
		if isinstance(vidfile, str):
			if not os.path.isfile(vidfile):
				raise ValueError(f'Input video file is NOT found! At {vidfile}')

			window_name = os.path.splitext(os.path.basename(vidfile))[0]
			cap = FileVideoStream(vidfile).start()
		else:
			window_name = 'piv_stream'
			cap = WebcamVideoStream(vidfile).start()

		time.sleep(2.0)  # warming up the input

		# create output directory
		outdir = os.path.join(self.default, f'vid_{window_name}')
		if not os.path.isdir(outdir):
			os.makedirs(outdir)

		count = 0
		prev_frame = None

		while True:
			curr_frame = cap.read()
			if curr_frame is None:  # End of the video!
				break

			if prev_frame is not None:
				count += 1
				out_flow = self.parser(self.net, prev_frame, curr_frame, device=self.device)

				# Post-processing here!
				out_name = window_name + '_%06d_out.flo' % count
				out_name = os.path.join(outdir, out_name)
				if write:
					write_flow(out_flow, out_name)
					tqdm.write(f'Writing {out_name}')

			prev_frame = curr_frame
			# displaying the input video
			cv2.imshow(window_name, curr_frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		cv2.destroyAllWindows()
		cap.stop()
		print(f'Finish processing all images from {window_name} video stream!')
		print(f'Total {count} frames are processed.')

	def images_parsing(self, imgdir: str, pair: bool = True, write: bool = True) -> None:
		if isinstance(imgdir, str):  # single directory
			if not os.path.isdir(imgdir):
				raise ValueError(f'Input directory is NOT found! At {imgdir}')

			else:
				# create output directory
				basedir = os.path.basename(imgdir) + '_parse'
				outdir = os.path.join(self.default, basedir)
				if not os.path.isdir(outdir):
					os.makedirs(outdir)

				im_files = image_files_from_folder(imgdir, pair=pair)
				if pair:  # filename as a paired images
					for file1 in tqdm(im_files, ncols=100, leave=True, unit='pair', desc=f'Evaluating {imgdir}'):
						fbase, fext = os.path.splitext(file1)
						file2 = fbase.rsplit('_', 1)[0] + '_img2' + fext

						if os.path.isfile(file2):
							out_flow = self.parser(self.net,
													PIL.Image.open(file1).convert('RGB'),
													PIL.Image.open(file2).convert('RGB'),
													device=self.device)
							# Post-processing here
							out_name = flowname_modifier(file1, outdir, pair=pair)
							if write:
								write_flow(out_flow, out_name)

				else:  # filename as a sequential frame
					prev_frame = None
					for curr_frame in tqdm(im_files, ncols=100, leave=True, unit='pair', desc=f'Evaluating {imgdir}'):
						if prev_frame is not None:
							out_flow = self.parser(self.net,
													PIL.Image.open(prev_frame).convert('RGB'),
													PIL.Image.open(curr_frame).convert('RGB'),
													device=self.device)
							# Post-processing here
							out_name = flowname_modifier(prev_frame, outdir, pair=pair)
							if write:
								write_flow(out_flow, out_name)

						prev_frame = curr_frame

				tqdm.write(f'Finish processing all images from {imgdir} path!')
		else:
			raise ValueError('Unknown input! Input must be a directory path')

	def dataloader_parsing(self, dir: str, pair: bool = True, write: bool = True) -> None:
		if not os.path.isdir(dir):
			raise ValueError(f'Input directory is NOT found! At {dir}')

		basedir = os.path.basename(dir) + '_loader'
		outdir = os.path.join(self.default, basedir)
		if not os.path.isdir(outdir):
			os.makedirs(outdir)

		test_dataset = InferenceRun(root=dir, pair=pair)
		test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

		for images, img_name in tqdm(test_dataloader, ncols=100, leave=True, unit='pair', desc=f'Evaluating {dir}'):

			# Add to device
			images[0], images[1] = images[0].to(self.device), images[1].to(self.device)
			out_flow = estimate(net, images[0], images[1], tensor=False)

			# Post-processing here
			out_name = flowname_modifier(img_name[0], outdir, pair=pair)
			if write:
				write_flow(out_flow, out_name)

		tqdm.write(f'Done processing {len(test_dataloader)} pairs')

	@staticmethod
	def parser(net, im1, im2, device='cpu'):
		assert im1.size == im2.size

		tensor_im1 = transforms.ToTensor()(im1).to(device)
		tensor_im2 = transforms.ToTensor()(im2).to(device)

		C, H, W = tensor_im1.size()
		tensor_im1 = tensor_im1.view(1, C, H, W)
		tensor_im2 = tensor_im2.view(1, C, H, W)
		out_flow = estimate(net, tensor_im1, tensor_im2)
		return out_flow


##########################################################
if __name__ == '__main__':
	tic = time.time()
	root_model = './models/pretrain_torch'
	root_input = './images'

	# INPUTS
	args_model = os.path.join(root_model, 'PIV-LiteFlowNet-en.paramOnly')
	args_vid = os.path.join(root_input, 'stepen_exp_rot32.gif')
	args_imdir_pair = os.path.join(root_input, 'pair_cai_SQG')
	args_imdir_seq = os.path.join(root_input, 'seq_TA_sbr')
	# args_imdir_seq = os.path.join(root_input, 'seq_hiroki_imf108g30-1-25')  # 1024 x 1024 takes too much memory!
	# args_img1 = os.path.join(root_input, 'DNS_turbulence_img1.tif')
	# args_img2 = os.path.join(root_input, 'DNS_turbulence_img2.tif')

	# Set the model
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	if os.path.isfile(args_model):
		weights = torch.load(args_model)
	else:
		raise ValueError('Unknown params input!')

	# net = hui_liteflownet(args_model, device=device)
	net = piv_liteflownet(weights).to(device)
	infer = Inference(net, netname=args_model, device=device)

	# infer.images_parsing(args_imdir_seq, pair=False, write=True)
	# infer.images_parsing(args_imdir_pair, pair=True, write=True)
	# infer.dataloader_parsing(args_imdir_seq, pair=False, write=True)
	infer.dataloader_parsing(args_imdir_pair, pair=True, write=True)
	# infer.video_parsing(vidfile=0, write=False)
	dur = time.time() - tic
	tqdm.write(f'Finish processing in {float("{0:.2f}".format(dur))} s!')

	# Displaying the results (for manual parser)
	out_name = os.path.join(os.path.dirname(args_img1), 'test_piv.flo')
	out_name_q = os.path.join(os.path.dirname(args_img1), 'test_piv.png')
	out_flow = Inference.parser(net,
								PIL.Image.open(args_img1).convert('RGB'),
								PIL.Image.open(args_img2).convert('RGB'),
								device=device)

	write_flow(out_flow, out_name)
	u, v = quiver_plot(out_flow, filename=out_name_q)

	# Object output (Sniklaus, PyTorch)
	object_output = open(args_output, 'wb')

	np.array([80, 73, 69, 72], np.uint8).tofile(object_output)
	np.array([out_flow.size(2), out_flow.size(1)], np.int32).tofile(object_output)
	np.array(out_flow.numpy().transpose(1, 2, 0), np.float32).tofile(object_output)

	object_output.close()
