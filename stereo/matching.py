import cv2
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt


def gen_template(TC: int = 5, HC: int = 25, LC: int = 25) -> np.array:
    """
    Generate template image.
    Args:
        TC  : (int) cross thickness
        HC  : (int) cross global height
        LC  : (int) cross global length/width
    Returns:
        temp;ate    : (np.array) template image, in greyscale uint8 format.
    """
    LC2, HC2, TC2 = np.ceil(LC / 2), np.ceil(HC / 2), np.floor(TC / 2)
    template = np.zeros([HC, LC])

    if TC2 != np.ceil(TC / 2):  # if TC is odd.
        template[np.int(HC2 - TC2 - 1):np.int(HC2 + TC2), 0:np.int(LC)] = np.ones([TC, LC])  # horizontal cross.
        template[0:np.int(HC), np.int(LC2 - TC2 - 1):np.int(LC2 + TC2)] = np.ones([HC, TC])  # vertical cross.
    else:  # if TC is even.
        template[HC2 - TC2 - 1:HC2 + TC2 - 1, 0:LC] = np.ones([TC, LC])  # horizontal cross.
        template[0:HC, LC2 - TC2 - 1:LC2 + TC2] = np.ones([HC, TC])  # vertical cross.

    template = np.array(template * 255, dtype=np.uint8)

    return template


def template_matching(gray_img: np.array, template: np.array, threshold: float = 0.0) -> np.array:
    """
    Perform template matching between the main image (gray_img) and the template image.
    Args:
        gray_img    : (np.array) bitwise main image.
        template    : (np.array) bitwise template image.
        threshold   : (float) threshold value for the correlation map.
    Returns:
        Processed correlation map.
    """
    # Padding init.
    pad = [int((template.shape[0] - 1) / 2), int((template.shape[1] - 1) / 2)]
    pad_gray = np.zeros([gray_img.shape[0] + 2 * pad[0], gray_img.shape[1] + 2 * pad[1]], dtype=np.uint8)

    pad_gray[pad[0]:-pad[0], pad[1]:-pad[1]] = gray_img
    res = cv2.matchTemplate(pad_gray, template, cv2.TM_CCOEFF_NORMED)

    # Threshold: 0.0 (without threshold); 0.7 (default config.)
    thres_mask = res > threshold
    res_thres = res * thres_mask

    # Add gaussian blur
    result = cv2.blur(res_thres, (2, 2))

    return result


def findLocalMax(image: np.array):
    """
    Finding local maxima from dots pattern.
    Args:
        image   : (np.array) W, H post-processed correlation map image.
    Returns:
        coords  : (np.array) N, 2 array with (x, y) config.
    """
    lbl = ndimage.label(image)
    points = ndimage.measurements.center_of_mass(
        image, lbl[0], [i + 1 for i in range(lbl[1])]
    )

    # Change into (x, y) format!
    coord = np.fliplr(np.array(points))

    return coord


def select_ref(coords):
    """
    Choosing the 4 nearest reference points using clockwise, Left - Right - Down - Left (L-R-D-L), direction.
    Args:
        select_ref  : (N, 2) The original coordinate points.
    """
    n_point_ref = 4
    points_selected = []

    for i in range(n_point_ref):
        point_ref = plt.ginput(1, timeout=-1, show_clicks=True)[0]
        print(f'\t{i+1}. Clicked at {point_ref}')

        # in (x, y) coordinate format
        distance = np.linalg.norm(coords - np.array(point_ref), axis=1)
        min_distance_arg = np.argmin(distance)
        plt.plot(coords[min_distance_arg, 0], coords[min_distance_arg, 1], 'yo')  # Plotting each selected point

        points_selected.append(min_distance_arg)

    # Plotting the line
    points_ref = np.zeros([4, 2])
    for i in range(n_point_ref):
        j = i + 1 if i < n_point_ref - 1 else 0
        i_pt, j_pt = points_selected[i], points_selected[j]

        plt.plot([coords[i_pt, 0], coords[j_pt, 0]],
                 [coords[i_pt, 1], coords[j_pt, 1]], 'r-')

        points_ref[i, :] = coords[i_pt, :]

    # Calculating the center point
    c_x = (np.abs(points_ref[1, 0] - points_ref[0, 0]) + np.abs(points_ref[3, 0] - points_ref[2, 0])) * 0.5
    c_y = (np.abs(points_ref[3, 1] - points_ref[0, 1]) + np.abs(points_ref[2, 1] - points_ref[1, 1])) * 0.5

    c_point = [c_x, c_y]
    # c_point = points_ref.mean(axis=0)
    return points_ref, points_selected, c_point
