import numpy as np

def true_mask(sgm, i):
    return np.where(sgm == i, 1, 0).astype(bool)

def TP_FN_FP_TN(true, pred, undef):
    true = true & ~ undef

    TP = true & pred
    FN = true & ~ pred # true - pred
    FP = pred & ~ true # pred - true
    TN = (~ (true | pred)) & ~ undef

    return TP.sum(), FN.sum(), FP.sum(), TN.sum()

def time_str(t, shift = 3):
    return str(int(t // 60)).rjust(shift) + "m" + str(int(t % 60)).zfill(2) + "s"