import cv2
import numpy as np

def grabcut(img, mask, iterCount = 1, mode = 'RECT'):
    cv2.setRNGSeed(0)
    
    if mode == 'RECT':
        mask, _, _ = cv2.grabCut(
        img, 
        np.zeros(img.shape[:2], dtype="uint8"), 
        mask, 
        np.zeros((1, 65), dtype="float"),
        np.zeros((1, 65), dtype="float"), 
        iterCount = iterCount, 
        mode = cv2.GC_INIT_WITH_RECT)
     
    elif mode == 'MASK':
        mask, _, _ = cv2.grabCut(
        img, 
        mask.copy(), 
        None, 
        np.zeros((1, 65), dtype="float"),
        np.zeros((1, 65), dtype="float"), 
        iterCount = iterCount,
        mode = cv2.GC_INIT_WITH_MASK)
    
    return mask.astype('uint8')


def gcmask_to_grayimg(mask):
    mask = mask.astype('uint8')

    mask[mask == 0] = 0    # definite background
    mask[mask == 1] = 255  # definite foreground
    mask[mask == 2] = 85   # probable background
    mask[mask == 3] = 170  # probable foreground

    return mask