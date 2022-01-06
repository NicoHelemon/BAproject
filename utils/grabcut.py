import cv2
import numpy as np
import utils.image as im

def grabcut(img_pil, mask, iterCount = 1, mode = 'RECT'):
    cv2.setRNGSeed(0)
    img_cv2 = im.pil_to_cv2(img_pil)
    
    if mode == 'RECT':
        mask, _, _ = cv2.grabCut(
        img_cv2, 
        np.zeros(img_cv2.shape[:2], dtype="uint8"), 
        mask, 
        np.zeros((1, 65), dtype="float"),
        np.zeros((1, 65), dtype="float"), 
        iterCount = iterCount, 
        mode = cv2.GC_INIT_WITH_RECT)
     
    elif mode == 'MASK':
        mask, _, _ = cv2.grabCut(
        img_cv2, 
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