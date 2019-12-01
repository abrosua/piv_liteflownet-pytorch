import numpy as np
import torch
import torch.nn as nn

from typing import Dict, Optional, List, Tuple, Union, Iterable, Any


__all__ = [
    'hui_loss', 'piv_loss'
]

def EPE(input_flow, target_flow, mean=True):
	epe_map = torch.norm(target_flow - input_flow, p=2, dim=1)
	batch_size = epe_map.size(0)

	if mean:
		epe_loss = epe_map.mean()
	else:
		epe_loss = epe_map.sum() / batch_size

	return epe_loss


class L1(nn.Module):
	def __init__(self, mean=True):
		super(L1, self).__init__()
		self.mean = mean

	def forward(self, output, target):
		loss_map = torch.abs(output - target)
		batch_size = loss_map.size(0)

		if self.mean:
			lossvalue = loss_map.mean()
		else:
			lossvalue = loss_map.sum() / batch_size

		return lossvalue


class L2(nn.Module):
	def __init__(self, mean=True):
		super(L2, self).__init__()
		self.mean = mean

	def forward(self, output, target):
		loss_map = torch.norm(output - target, p=2, dim=1)
		batch_size = loss_map.size(0)

		if self.mean:
			lossvalue = loss_map.mean()
		else:
			lossvalue = loss_map.sum() / batch_size

		return lossvalue


class L1Loss(nn.Module):
	def __init__(self, mul_scale=1):
		super(L1Loss, self).__init__()
		self.mul_flow = float(mul_scale)

		self.loss = L1()
		self.loss_labels = ['L1', 'EPE']

	def forward(self, output, target):
		lossvalue = self.mul_flow * self.loss(output, target)
		epevalue = self.mul_flow * EPE(output, target)
		return [lossvalue, epevalue]


class L2Loss(nn.Module):
	def __init__(self, mul_scale=1):
		super(L2Loss, self).__init__()
		self.mul_flow = float(mul_scale)

		self.loss = L2()
		self.loss_labels = ['L2', 'EPE']

	def forward(self, output, target):
		lossvalue = self.mul_flow * self.loss(output, target)
		epevalue = self.mul_flow * EPE(output, target)
		return [lossvalue, epevalue]


class MultiScale(nn.Module):
	"""
	Multi stage loss calculation to calculate total loss.
	Pyramid level is sorted BACKWARD (e.g., 6, 5, 4, 3, 2, 1)
	"""
	def __init__(self, div_scale=0.05, startScale=2, l_weight=None, norm='L1'):
		super(MultiScale, self).__init__()

		if l_weight is None:  # Default value, taken from Hui, 2018
			l_weight = (0.32, 0.08, 0.02, 0.01, 0.005)

		self.startScale = startScale
		self.loss_weights = l_weight
		self.numScales = len(l_weight)

		self.div_flow = div_scale
		self.multiScales = [nn.AvgPool2d(self.startScale * (2 ** scale), self.startScale * (2 ** scale))
							for scale in reversed(range(self.numScales))]

		self.loss = L1() if norm == 'L1' else L2()
		self.loss_labels = ['MultiScale-' + norm, 'EPE'],

	def forward(self, output, target):
		lossvalue, epevalue = 0, 0

		if type(output) in [tuple, list]:  # For TRAINING mode/error
			assert self.numScales == len(output)  # Check the number of pyramid level used
			target = self.div_flow * target

			for i, output_ in enumerate(output):
				target_ = self.multiScales[i](target)

				if type(output_) not in [tuple, list]:
					epevalue += self.loss_weights[i] * EPE(output_, target_)
					lossvalue += self.loss_weights[i] * self.loss(output_, target_)

				else:
					for out_ in output_:
						epevalue += self.loss_weights[i] * EPE(out_, target_)
						lossvalue += self.loss_weights[i] * self.loss(out_, target_)

			return [lossvalue, epevalue]

		else:  # For TESTING mode/error
			epevalue += EPE(output, target)
			lossvalue += self.loss(output, target)
			return [lossvalue, epevalue]


class LevelLoss(nn.Module):
	"""
	Multi stage loss calculation to calculate loss at each stage.
	Pyramid level is sorted BACKWARD (e.g., 6, 5, 4, 3, 2, 1)
	"""
	def __init__(self, div_scale=0.05, startScale=2, n_level=5, norm='L1'):
		super(LevelLoss, self).__init__()

		self.startScale = startScale
		self.numScales = n_level

		self.div_flow = div_scale
		self.multiScales = [nn.AvgPool2d(self.startScale * (2 ** scale), self.startScale * (2 ** scale))
							for scale in reversed(range(self.numScales))]

		self.loss = L1() if norm == 'L1' else L2()
		self.loss_labels = ['MultiScale-' + norm, 'EPE'],

	def forward(self, output, target):
		lossvalue, epevalue = [], []

		if type(output) in [tuple, list]:
			target = self.div_flow * target
			assert self.numScales == len(output)  # Check the number of pyramid level used

			for i, output_ in enumerate(output):
				target_ = self.multiScales[i](target)

				if type(output_) not in [tuple, list]:
					epevalue.append(EPE(output_, target_))
					lossvalue.append(self.loss(output_, target_))

				else:  # Take the last result of each level (Hence the output of each level's final calculation)
					epevalue.append(EPE(output_[-1], target_))
					lossvalue.append(self.loss(output_[-1], target_))

			return [lossvalue, epevalue]

		else:
			raise ValueError(f'The "output" type must be a list/tuple to perform per level evaluation!')


# ----------------- calling function -----------------
def hui_loss(level_eval=False, mul_scale=20, norm='L1'):
	if level_eval:  # Evaluate the error on every level!
		return LevelLoss(div_scale=1/mul_scale, norm=norm)

	else:
		return MultiScale(div_scale=1/mul_scale, norm=norm)


def piv_loss(level_eval=False, mul_scale=5, norm='L1'):
	if level_eval:  # Evaluate the error on every level!
		return LevelLoss(div_scale=1 / mul_scale, startScale=1, n_level=6, norm=norm)

	else:
		# loss_weight = (0.32, 0.08, 0.02, 0.01, 0.005, 0.01)
		loss_weight = (0.001, 0.001, 0.001, 0.001, 0.001, 0.01)  # Value taken from Cai, 2019
		return MultiScale(div_scale=1 / mul_scale, startScale=1, l_weight=loss_weight, norm=norm)
