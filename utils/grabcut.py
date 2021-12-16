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
        mask, 
        None, 
        np.zeros((1, 65), dtype="float"),
        np.zeros((1, 65), dtype="float"), 
        iterCount = iterCount,
        mode = cv2.GC_INIT_WITH_MASK)

    quaternary_mask3 = mask                                 # [0, 1, 2, 3]

    binary_mask255 = gcmask_to_grayimg(mask, mode = '2ary') # [0, 255]

    binary_mask1 = mask.astype('uint8') % 2                 # [0, 1]
    
    return quaternary_mask3, binary_mask255, binary_mask1


def gcmask_to_grayimg(mask, mode = '4ary'):
    mask = mask.astype('uint8')

    if mode == '2ary':
        mask = mask % 2 # merge probable and definite values together

    mask[mask == 0] = 0    # definite background
    mask[mask == 1] = 255  # definite foreground
    mask[mask == 2] = 85   # probable background
    mask[mask == 3] = 170  # probable foreground

    return mask