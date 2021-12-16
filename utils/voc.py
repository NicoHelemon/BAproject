import numpy as np
import scipy.ndimage.morphology as ndimage

def true_mask(sgm, i, dilation = 2):
    true_mask = np.where(sgm == i, 1, 0)
    if dilation == 0:
        return true_mask
    else:
        return ndimage.binary_dilation(true_mask, iterations = dilation)

def accommodate_undef(mask, sgm):
    undef = np.where(sgm == 255, 1, 0)
    return mask & ~ undef

def IoU(mask1, mask2):
    return (mask1 & mask2).sum() / (mask1 | mask2).sum()

def IoU_acc_undef(true, pred, sgm):
    pred = accommodate_undef(pred, sgm)
    return IoU(true, pred)

def time_str(t, shift = 3):
    return str(int(t // 60)).rjust(shift) + "m" + str(int(t % 60)).zfill(2) + "s"