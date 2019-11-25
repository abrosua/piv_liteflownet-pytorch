"""
This code was adapted from the C++ code here: http://vision.middlebury.edu/flow/code/flow-code.zip
by https://github.com/marximus/flowviz.git
"""
import sys
import numpy as np

# --------------------- INIT ---------------------
ncols = 0
MAXCOLS = 60
colorwheel = np.empty((MAXCOLS, 3))


# --------------------- COLOR CODE ---------------------
def _setcols(r, g, b, k):
	global colorwheel

	colorwheel[k, 0] = r
	colorwheel[k, 1] = g
	colorwheel[k, 2] = b


def _makecolorwheel():
	global ncols

	RY = 15
	YG = 6
	GC = 4
	CB = 11
	BM = 13
	MR = 6
	ncols = RY + YG + GC + CB + BM + MR
	if ncols > MAXCOLS:
		sys.exit(1)

	k = 0
	for i in range(0, RY):
		_setcols(255, 255 * i / RY, 0, k)
		k += 1
	for i in range(0, YG):
		_setcols(255 - 255 * i / YG, 255, 0, k)
		k += 1
	for i in range(0, GC):
		_setcols(0, 255, 255 * i / GC, k)
		k += 1
	for i in range(0, CB):
		_setcols(0, 255 - 255 * i / CB, 255, k)
		k += 1
	for i in range(0, BM):
		_setcols(255 * i / BM, 0, 255, k)
		k += 1
	for i in range(0, MR):
		_setcols(255, 0, 255 - 255 * i / MR, k)
		k += 1


def compute_color(fx, fy, colim, original_color: bool = False):
	"""
    Parameters
    ----------
    fx : ndarray, dtype float, shape (height, width)
        U vector components.
    fy : ndarray, dtype float, shape (height, width)
        V vector components.
    colim : ndarray, dtype uint8, shape(height, width, 3)
        Colored image.
    Returns
    -------
    None
    """
	if ncols == 0:
		_makecolorwheel()

	rad = np.sqrt(fx * fx + fy * fy)
	a = np.arctan2(-fy, -fx) / np.pi
	fk = (a + 1.0) / 2.0 * (ncols - 1)
	# k0 = int(fk)
	k0 = fk.astype(np.int)
	k1 = (k0 + 1) % ncols

	if original_color:
		f = 0  # original color wheel
	else:
		f = fk - k0  # marximus' version of color wheel

	for b in range(0, 3):
		col0 = colorwheel[k0, b] / 255.0
		col1 = colorwheel[k1, b] / 255.0
		col = (1 - f) * col0 + f * col1
		idx = rad <= 1
		col[idx] = 1 - rad[idx] * (1 - col[idx])
		col[rad > 1] = col[rad > 1] * 0.75
		colim[..., 2 - b] = (255.0 * col).astype(np.int)
