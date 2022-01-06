import utils.grabcut as gcut
import utils.image as im
import utils.cam as cam
import numpy as np

def IU(mask1, mask2, c1 = (255, 127, 127), c2 = (127, 255, 127)):
    mask1, mask2 = mask1.astype(bool), mask2.astype(bool)
    x, y = mask1.shape
    visu = np.zeros((x, y, 3))
    visu[mask1 & ~ mask2] = np.array(c1)
    visu[mask1 & mask2]   = np.minimum(c1, c2)
    visu[mask2 & ~ mask1] = np.array(c2)
    return visu

def visu_grabcut(img_pil, true, cbbox):
    img_cv2 = im.pil_to_cv2(img_pil)

    input_mask_4 = im.cbbox_mask(img_cv2.shape[:2], cbbox)
    input_mask_4[input_mask_4 == 1] = 170

    pred_4 = gcut.grabcut(img_pil, im.corner_to_delta(cbbox), mode = 'RECT')
    pred_2 = pred_4 % 2

    output_mask_4 = gcut.gcmask_to_grayimg(pred_4)
    output_mask_2 = gcut.gcmask_to_grayimg(pred_4)

    crop = im.bitwise_and(img_cv2, pred_2)

    iu = IU(pred_2, true)

    visus = [input_mask_4, output_mask_4, output_mask_2, crop, iu]
    names = ['gc_in4', 'gc_out4', 'gc_out2', 'gc_crop', 'gc_iu']

    return visus, names