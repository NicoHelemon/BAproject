import utils.grabcut as gcut
import utils.image as im
import utils.cam as cam

def sgm_grabcut(img_cv2, cbbox):
    pred = gcut.grabcut(img_cv2, im.corner_to_delta(cbbox), mode = 'RECT') % 2

    return pred.astype(bool)

def sgm_grabcut_cam(img_cv2, img_cam, t, mode = 'PF_PB', cbbox = None):
    delta = 0.01

    if mode   == 'PF_PB':
        t0, t1, t2 = 0.0, t, 1.0
    elif mode == 'F_PF':
        t0, t1, t2 = 0.0, 0.0 + delta, t
    elif mode == 'F_PB':
        t0, t1, t2 = 0.0, t, t + delta
 
    gcmask_cam = cam.cam_to_gcmask(img_cam, t0, t1, t2) 
    pred = gcut.grabcut(img_cv2, gcmask_cam, mode = 'MASK') % 2

    if cbbox is not None:
        pred = im.bitwise_and(pred, im.cbbox_mask(pred.shape[:2], cbbox))

    return pred.astype(bool)

def sgm_cam(img_cam, t):
    pred  = cam.cam_to_gcmask(img_cam, t) % 2

    return pred.astype(bool)


