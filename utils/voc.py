import numpy as np
import scipy.ndimage.morphology as ndimage

#def true_mask(sgm, i, dilation = 2):
#    true_mask = np.where(sgm == i, 1, 0)
#    if dilation == 0:
#        return true_mask
#    else:
#        return ndimage.binary_dilation(true_mask, iterations = dilation)

def true_mask(sgm, i):
    return np.where(sgm == i, 1, 0)

# mask1 - mask2
def diff(mask1, mask2):
    return mask1 & ~ mask2

def xnor(mask1, mask2):
    return ~ (mask1 ^ mask2)

# TP / (TP + FP + FN)
def IoU(true, pred, sgm):
    undef = np.where(sgm == 255, 1, 0).astype(bool)

    TP       = diff(true & pred, undef).sum()
    TP_FP_FN = diff(true | pred, undef).sum()

    return TP / TP_FP_FN

# TP + TN / (TP + TN + FP + FN)
def accuracy(true, pred, sgm):
    undef = np.where(sgm == 255, 1, 0).astype(bool)

    TP_TN       = diff(xnor(true, pred), undef).sum()
    TP_TN_FP_FN = diff(true | ~true, undef).sum()

    return TP_TN / TP_TN_FP_FN

def time_str(t, shift = 3):
    return str(int(t // 60)).rjust(shift) + "m" + str(int(t % 60)).zfill(2) + "s"