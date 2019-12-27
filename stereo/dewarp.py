import numpy as np
import scipy.optimize as so
import cv2


class Guess:
    def __init__(self, old_pts, center_dist, pt1):
        # Variable init.
        self.old_x, self.old_y = old_pts[:, 0], old_pts[:, 1]
        self.center_dist_x, self.center_dist_y = center_dist
        self.pt1 = pt1
        self.CMAX = 100  # specify the maximum number points in each column to analyze.

        # Results init.
        self.new_x = np.zeros(np.shape(self.old_x))
        self.new_y = np.zeros(np.shape(self.old_y))

    def __call__(self, *args, **kwargs):
        """
        Perform guessing the new points coordinate.
        """

        # Guessing along the X-axis.
        self._guess_x(mult=1)
        self._guess_x(mult=-1)

        # finding the X-coordinate where all the middle column's cross points form a straight vertical line(as above).
        xnew_tmp = np.divide((self.new_x - self.new_x[self.pt1]), self.center_dist_x)
        xnew_tmp = np.round(xnew_tmp)

        self.new_y[self.pt1] = self.old_y[self.pt1]
        i0 = np.where(xnew_tmp == 0)
        self._guess_y(i0, mult=1)
        self._guess_y(i0, mult=-1)

        # when all the cross points in the middle column has each new coordinates,
        # lets move to the right column. Also with some added code to return all
        # the values of variables back to the default setting, the rest is the
        # same(again).
        self._guess_rest(xnew_tmp, mult=1)
        self._guess_rest(xnew_tmp, mult=-1)

        new_pts = np.array([self.new_x, self.new_y]).T
        return new_pts

    def _guess_x(self, mult: int = 1):
        """
        Guessing along X-axis.
        """
        x_dist = np.divide((self.old_x - self.old_x[self.pt1]), self.center_dist_x)
        x_dist_round = np.round(x_dist)
        t = np.where(x_dist_round == 0)
        self.new_x[t[0]] = self.old_x[self.pt1]
        x_centroid = np.mean(self.old_x[t[0]])

        # guessing to the right of reference vertical line,
        # until the last point at the right bottom corner of image.
        n = 1
        for a in range(1, self.CMAX + 1):  # the process analyse the columns from top to bottom.
            x_st = x_centroid + mult * n * self.center_dist_x  # SCALAR
            x_dist = np.divide((self.old_x - x_st), self.center_dist_x)
            x_dist_round = np.round(x_dist)
            t = np.where(x_dist_round == 0)

            if np.size(t[0]) != 0:  # if the process already hit the bottom of each analysed column of the image
                self.new_x[t[0]] = self.old_x[self.pt1] + mult * self.center_dist_x * a
                x_centroid = np.mean(self.old_x[t[0]])
                n = 1
            else:
                n = n + 1  # continue to the next column on the right

    def _guess_y(self, nearest_y, mult: int = 1):
        """
        Perform guessing of the respective y coord., in the same middle column,
        by adding integer A (A=1,2,3,...) * center_dist_y until all the cross points above the
        central cross point are the multiplication of the integer A.
        """
        alfa_y = self.old_y[nearest_y[0]]
        old_init_y = self.old_y[self.pt1]

        n = 1
        for A in range(1, self.CMAX + 1):
            old_init_y = old_init_y if np.size(old_init_y) == 1 else old_init_y[0]

            y_dist = np.divide((alfa_y - old_init_y), self.center_dist_y)
            y_dist = np.round(y_dist)
            y_id = np.where(y_dist == (mult * n))

            if np.size(y_id[0]) != 0:
                self.new_y[nearest_y[0][y_id[0]]] = self.old_y[self.pt1] + mult * A * self.center_dist_y
                old_init_y = self.old_y[nearest_y[0][y_id[0]]]
                n = 1
            else:
                n = n + 1

    def _guess_rest(self, xnew_tmp, mult: int = 1):
        d = 1
        for n in range(1, self.CMAX + 1):
            i1 = np.where(xnew_tmp == (mult * (n - d)))
            i2 = np.where(xnew_tmp == (mult * n))

            if np.size(i2[0]) != 0:
                alfa_y1 = self.old_y[i1[0]]
                alfa_y2 = self.old_y[i2[0]]
                yleng2 = np.ceil(len(alfa_y2) * 0.5)
                ystart = alfa_y2[np.int(yleng2)]

                ydiff = np.divide((ystart - alfa_y1), self.center_dist_y)
                ydmin = np.amin(np.abs(ydiff))
                y_id = np.argmin(np.abs(ydiff))
                ydmin = np.round(ydiff[y_id])
                yfix = self.new_y[i1[0][y_id]] + ydmin * self.center_dist_y
                yzero = np.where(np.round(np.divide((ystart - alfa_y2), self.center_dist_y)) == 0)
                self.new_y[i2[0][yzero[0]]] = yfix

                old_init_y = ystart
                u = 1
                for k in range(1, self.CMAX + 1):
                    old_init_y = old_init_y if np.size(old_init_y) == 1 else old_init_y[0]
                    y_dist = np.divide((alfa_y2 - old_init_y), self.center_dist_y)
                    y_dist = np.round(y_dist)
                    y_id = np.where(y_dist == u)

                    if np.size(y_id[0]) != 0:
                        self.new_y[i2[0][y_id[0]]] = yfix + k * self.center_dist_y
                        old_init_y = self.old_y[i2[0][y_id[0]]]
                        u = 1
                    else:
                        u = u + 1

                old_init_y = ystart
                b = 1
                for k in range(1, self.CMAX + 1):
                    old_init_y = old_init_y if np.size(old_init_y) == 1 else old_init_y[0]
                    y_dist = np.divide((alfa_y2 - old_init_y), self.center_dist_y)
                    y_dist = np.round(y_dist)
                    y_id = np.where(y_dist == -b)

                    if np.size(y_id[0]) != 0:
                        self.new_y[i2[0][y_id[0]]] = yfix - k * self.center_dist_y
                        old_init_y = self.old_y[i2[0][y_id[0]]]
                        b = 1
                    else:
                        b = b + 1
                d = 1

            else:
                d = d + 1


def map_coeff(old_coord: np.array, new_coord: np.array, pt1):
    """
    Calculating 24 mapping coefficient by solving Levenberg-Marquardt equation.
    Args:
        old_coord   : (np.array) Old calibration coordinate points.
        new_coord   : (np.array) New/dewarped calibration coordinate points.
        pt1         : (np.array) First selected reference point.
    Returns:
        The 24 warping coefficients.
    """
    # Init.
    new_rel = new_coord - new_coord[pt1]
    old_rel = old_coord - old_coord[pt1]
    p, q = new_rel[:, 0], new_rel[:, 1]
    k1, k2 = old_rel[:, 0], old_rel[:, 1]

    # ----- 1st mapping equation -----
    estimate_1 = [0, 0.0001, 0.0001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5000, 0]
    jrk = lambda a: np.sum(
        np.power((k1 - np.divide((a[0]*p + a[1]*q + a[2]), (a[3]*p + a[4]*q + a[5]))), 2) +
        np.power((k2 - np.divide((a[6]*p + a[7]*q + a[8]), (a[9]*p + a[10]*q + a[11]))), 2)
    )
    a = so.minimize(jrk, x0=np.array([1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]), method='Nelder-Mead')
    a = a['x']
    estimate_1 = jrk(a)

    # ----- 2nd mapping equation -----
    estimate_2 = [0, 0.0001, 0.0001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20000, 0]
    Jrk = lambda A: np.sum(
        np.power((k1 - np.divide(
            (A[0]*p + A[1]*q + A[2] + A[3]*np.power(p, 2) + A[4]*np.power(q, 2) + A[5]*np.multiply(p, q)),
            (A[6]*p + A[7]*q + A[8] + A[9]*np.power(p, 2) + A[10]*np.power(q, 2) + A[11]*np.multiply(p, q)))), 2) +
        np.power((k2 - np.divide(
            (A[12]*p + A[13]*q + A[14] + A[15]*np.power(p, 2) + A[16]*np.power(q, 2) + A[17]*np.multiply(p, q)),
            (A[18]*p + A[19]*q + A[20] + A[21]*np.power(p, 2) + A[22]*np.power(q, 2) + A[23]*np.multiply(p, q)))), 2)
    )
    A = so.minimize(Jrk, x0=np.array([a[0], a[1], a[2], 0, 0, 0, a[3], a[4], a[5], 0, 0, 0,
                                      a[6], a[7], a[8], 0, 0, 0, a[9], a[10], a[11], 0, 0, 0]), method='Nelder-Mead'
                    )
    A = A['x']
    estimate_2 = Jrk(A)

    return A


def warp(gray_img, old_pts, pt1, A):
    """

    Args:
        gray_img: (W, H) The grayscaled input image.
        old_pts : (pts, 2) Old/original point coordinates.
        pt1     : (2,) The first reference point (tl - top left).
        A       : (24,) The 24 mapping coefficients.
    Returns:
        Dewarped image
    """
    # Asserting input
    if gray_img.max() <= 1.0:
        gray_img = gray_img * 255
    if not np.issubclass_(gray_img.dtype.type, np.uint8):
        gray_img = gray_img.astype(np.uint8)

    # Init.
    width, height = gray_img.shape
    old_x, old_y = old_pts[:, 0], old_pts[:, 1]

    # preparation for the place of dewarping values (1 Megapixel = 1000x1000 pixels).
    w_grid, h_grid = np.meshgrid(np.arange(0, width), np.arange(0, height))

    # calculating the new X and Y coordinates of the dewarped image.
    x = w_grid - old_x[pt1]
    y = h_grid - old_y[pt1]

    new_x, new_y = nl_trans(x, y, A)

    new_x = np.round(new_x + old_x[pt1])
    new_y = np.round(new_y + old_y[pt1])

    # recover the image area that are lost during the mapping with black color.
    mask_x = np.multiply((new_x >= 0), (new_x <= (width - 1)))
    mask_y = np.multiply((new_y >= 0), (new_y <= (height - 1)))
    fill_x = np.logical_or((new_x > (width - 1)), (new_x < 0))
    fill_y = np.logical_or((new_y > (height - 1)), (new_y < 0))

    new_x = np.multiply(mask_x, new_x) + np.multiply(fill_x, (width - 1))
    new_y = np.multiply(mask_y, new_y) + np.multiply(fill_y, (height - 1))

    # dewarping the raw images into the new symetrical image.
    gray_img = np.transpose(gray_img)
    dewarp = np.reshape(gray_img, width * height)  # Vector
    i = new_y + (new_x * height)
    i = np.transpose(i)
    i = np.int64(np.reshape(i, width * height))

    i_min, i_max = np.amin(i), np.amax(i)
    min_pos = np.where(i == i_min)

    dewarp_new = dewarp[i]
    dewarp_new = np.reshape(dewarp_new, [height, width])  # Array
    dewarp_new = np.transpose(dewarp_new)

    return dewarp_new


def nl_trans(x, y, A):
    """

    :param x:
    :param y:
    :param A:
    :return:
    """
    new_x = np.divide((A[0] * x + A[1] * y + A[2] + A[3] * np.power(x, 2) + A[4] * np.power(y, 2) + A[5] * x * y),
                      (A[6] * x + A[7] * y + A[8] + A[9] * np.power(x, 2) + A[10] * np.power(y, 2) + A[11] * x * y)
                      )
    new_y = np.divide((A[12] * x + A[13] * y + A[14] + A[15] * np.power(x, 2) + A[16] * np.power(y, 2) + A[17] * x * y),
                      (A[18] * x + A[19] * y + A[20] + A[21] * np.power(x, 2) + A[22] * np.power(y, 2) + A[23] * x * y)
                      )

    return new_x, new_y


def dewarping(old_coord, points_ref, center_point):  # UNFINISHED !!!
    """
    Guessing the new/dewarped coordinate, by obtaining the perspective transform matrix
    Args:
        old_coord:
        points_ref:
        center_point:
    Returns:
    """
    (tl, tr, br, bl) = points_ref

    # calculate maximum width and height
    width1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width1), int(width2))

    height1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height1), int(height2))

    init = np.array([center_point[0], center_point[1]], dtype="float32")
    # init = np.array([tl[0], tl[1]], dtype="float32")
    init_dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32"
    )
    dst = init + init_dst - np.array([max_width - 1 / 2, max_height - 1 / 2], dtype="float32")

    # Get perspective transform
    points_ref = np.float32(points_ref)  # change type into float32 for perspective transform
    mat_transform = cv2.getPerspectiveTransform(points_ref, dst)
    # mat_translate = mat_transform[:2, 2]

    # Performing dewarping
    old_coord_ones = np.hstack([old_coord, np.ones([len(old_coord), 1])])
    new_coord_ones = np.dot(mat_transform, old_coord_ones.T)
    new_coord = (new_coord_ones[:2, :] / new_coord_ones[2, :]).T  #- mat_translate

    return new_coord
