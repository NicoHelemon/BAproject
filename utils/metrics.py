# Contains metrics definitions.

import numpy as np

def true_mask(sgm, i):
    return np.where(sgm == i, 1, 0).astype(bool)

def TP_FN_FP_TN(true, pred, undef):
    """ Given the ground truth segmentation, the prediction and the undefined region
        returns the True Positive, False Negative, False Positive and True Negative

    Parameters
    ----------
    true :
        Ground trush segmentation
    pred :
        Prediction segmentation
    undef :
        Undefined region

    Returns
    -------
    TP : int
        Nb of True Positive pixels
    FN : int
        Nb of False Negative pixels
    FP : int
        Nb of False Positive pixels
    TN : int
        Nb of True Negative pixels
    """

    pred = pred & ~ undef

    TP = true & pred
    FN = true & ~ pred # true - pred
    FP = pred & ~ true # pred - true
    TN = (~ (true | pred)) & ~ undef

    return TP.sum(), FN.sum(), FP.sum(), TN.sum()

def time_str(t, shift = 3):
    return str(int(t // 60)).rjust(shift) + "m" + str(int(t % 60)).zfill(2) + "s"