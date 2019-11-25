#   creates a test image showing the color encoding scheme

#   According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein  
#   Contact: dqsun@cs.brown.edu
#   Contact: schar@middlebury.edu

#   Author: Johannes Oswald, Technical University Munich
#   Contact: johannes.oswald@tum.de
#   Date: 26/04/2017

#	For more information, check http://vision.middlebury.edu/flow/ 

import numpy as np
import cv2
import math, os

from src.johswald import computeColor
from src.johswald import readFlowFile
from src.utils_plot import read_flow


if __name__ == '__main__':
	outdir = '../../images/'
	flow = read_flow(os.path.join(outdir, 'test.flo'))
	n_bands, height, width = flow.shape

	# CONST.
	truerange = 1
	range_f = truerange * 1.04
	s2 = int(round(height / 2))

	u = flow[: , : , 0]
	v = flow[: , : , 1]

	img = computeColor.computeColor(u/truerange, v/truerange)

	# img[s2, :, :] = 0
	# img[:, s2, :] = 0

	cv2.imshow('saved and reloaded test color pattern',img)
	cv2.waitKey()

	# color encoding scheme for optical flow
	img = computeColor.computeColor(u/range_f/math.sqrt(2), v/range_f/math.sqrt(2))

	cv2.imshow('optical flow color encoding scheme',img)
	cv2.imwrite('test.png', img)
	cv2.waitKey()
