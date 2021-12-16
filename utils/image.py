
import numpy as np
import cv2
from matplotlib import pyplot as plt

def delta_to_corner(dbbox):
    xmin, ymin, dx, dy = dbbox
    return xmin, ymin, xmin + dx, ymin + dy

def corner_to_delta(cbbox):
    xmin, ymin, xmax, ymax = cbbox
    return xmin, ymin, xmax - xmin, ymax - ymin

def area_dbbox(dbbox):
    _, _, dx, dy = dbbox
    return dx * dy

def cyx_to_yxc(img):
    return np.moveaxis(img, 0, -1)

def yxc_to_cyx(img):
    return np.moveaxis(img, 2, 0)

def rgb_swap_bgr(img):
    return img[:, :, ::-1]

def f1_to_f255(img):
    return (img * 255).astype('uint8')

def pil_to_cv2(img):
    return f1_to_f255(cyx_to_yxc(img))

def draw_cbbox(img, cbbox, color = (255, 0, 0), thickness = 2):
    min = cbbox[:2]
    max = cbbox[2:]
    return cv2.rectangle(img.copy(), min, max, color, thickness)

def cbbox_mask(img, cbbox):
    mask = np.zeros(img.shape[:2])
    xmin, ymin, xmax, ymax = cbbox
    mask[ymin:ymax, xmin:xmax] = 1

    return mask.astype('uint8')

def bitwise_and(img, mask):
    img = img.copy()
    return cv2.bitwise_and(img, img, mask = mask)

def show(img, mode = 'color'):
    if mode == 'color':
        plt.imshow(img.astype('uint8'))
        plt.show()
    if mode == 'gray' or mode == 'grey':
        plt.imshow(img.astype('uint8'), cmap='gray', vmin=0, vmax=255)
        plt.show()