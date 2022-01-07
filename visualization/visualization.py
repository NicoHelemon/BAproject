from torchvision import transforms
import numpy as np
from PIL import Image
import random
import cv2
from IPython.display import clear_output

import sys
sys.path.append('../')
import utils.grabcut as gcut
import utils.image as im
import utils.cam as cam
import utils.json as json
import utils.path as path
import utils.metrics as m

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
    output_mask_2 = gcut.gcmask_to_grayimg(pred_2)

    crop = im.bitwise_and(img_cv2, pred_2)

    iu = IU(pred_2, true)

    visus = [input_mask_4, output_mask_4, output_mask_2, crop, iu]
    names = ['in4', 'out4', 'out2', 'crop', 'iu']
    names = ['gc_' + n for n in names]

    return visus, names

def visu_grabcut_cam(img_pil, true, img_cam, t, mode = 'PF_PB', cbbox = None):
    img_cv2 = im.pil_to_cv2(img_pil)

    delta = 0.01
    if mode   == 'PF_PB':
        t0, t1, t2 = 0.0, t, 1.0
    elif mode == 'F_PF':
        t0, t1, t2 = 0.0, 0.0 + delta, t
    elif mode == 'F_PB':
        t0, t1, t2 = 0.0, t, t + delta
 
    input_mask_4 = cam.cam_to_gcmask(img_cam, t0, t1, t2)

    pred_4 = gcut.grabcut(img_pil, input_mask_4, mode = 'MASK')
    if cbbox is not None:
        pred_4 = im.bitwise_and(pred_4, im.cbbox_mask(pred_4.shape[:2], cbbox))

    pred_2 = pred_4 % 2

    input_mask_4  = gcut.gcmask_to_grayimg(input_mask_4)
    output_mask_4 = gcut.gcmask_to_grayimg(pred_4)
    output_mask_2 = gcut.gcmask_to_grayimg(pred_2)

    crop = im.bitwise_and(img_cv2, pred_2)

    iu = IU(pred_2, true)

    visus = [input_mask_4, output_mask_4, output_mask_2, crop, iu]
    names = ['in4', 'out4', 'out2', 'crop', 'iu']
    names = [f'gccam_0{int(t*10)}_{mode}_' + n for n in names]

    return visus, names

def visu_cam(img_pil, true, img_cam, t):
    img_cv2 = im.pil_to_cv2(img_pil)

    pred_4 = cam.cam_to_gcmask(img_cam, 0.0, t, 1.0)
    pred_2 = pred_4 % 2

    output_mask_4 = gcut.gcmask_to_grayimg(pred_4)
    output_mask_2 = gcut.gcmask_to_grayimg(pred_2)

    crop = im.bitwise_and(img_cv2, pred_2)

    iu = IU(pred_2, true)

    visus = [output_mask_4, output_mask_2, crop, iu]
    names = ['out4', 'out2', 'crop', 'iu']
    names = [f'cam_0{int(t*10)}_' + n for n in names]

    return visus, names

# ====================================================

root_path   = path.goback_from_current_dir(1)
json_path   = root_path + 'json\\'
devkit_path = root_path + 'VOCdevkit\VOC2012\\'

def load(img_name, mode):
    preprocess = transforms.Compose([transforms.ToTensor()])
    img       = preprocess(Image.open(devkit_path + 'JPEGImages\\' + img_name + '.jpg'))
    if mode == 'Object':
        true_sgms = preprocess(Image.open(devkit_path + 'SegmentationObject\\' + img_name + '.png'))
    elif mode == 'Class':
        true_sgms = preprocess(Image.open(devkit_path + 'SegmentationClass\\' + img_name + '.png'))
    
    return im.process(img, true_sgms)

def browse(i, limit, browsed):
    print(f'|| {browsed} browsing ||'.upper())
    ipt = input(f"Type:\n - \"n\" to go to next {browsed} \n - \"p\" to go to previous {browsed} \n" +
                f" - \"up\" to exit the {browsed} browsing and go one level up \n" + 
                f" - \"quit\" to quit the visualization \n" +
                f" - Anything else to remain on this current {browsed}\n")
    print("")
    if   ipt == "n":
        if i+1 > limit:
            return -1
        return i+1
    elif ipt == "p":
        return max(0, i-1)
    elif ipt == "up":
        return -1
    elif ipt == 'quit':
        return -2
    return i

def display_save(img_name, imgs, techs_names = [""], output_img_path = None):
    for img, name in zip(imgs, techs_names):
        print(name)
        im.show(img)

    save = False
    if output_img_path is not None:
        save_ipt = input(f"Type:\n - \"save\" if you want to save the last images\n" +
                      " - Anything else to not save\n")
        save = save_ipt == "save"

    if save:
        for img, name in zip(imgs, techs_names):
            if name != "":
                name = "_" + name
            if img.ndim == 3:
                img = im.rgb_swap_bgr(img)
            cv2.imwrite(output_img_path + f'{img_name}{name}.jpg', img)

def get_threshold():
    def is_float(x):
        try:
            float(x)
            return True
        except ValueError:
            return False

    t = input("Enter a default_t in range [0.1 - 0.9]: ")
    while (not is_float(t)) or (not (0.1 <= float(t) <= 0.9)):
        t = input("Error ; enter a default_t in range [0.1 - 0.9]: ")

    print("")
    return float(t)

# ====================================================


class V():
    def __init__(self):
        self.camnet             = cam.Cam()
        self.annotations_object = json.open_json(json_path + "voc-object-annotations-clean")
        self.annotations_class  = json.open_json(json_path + "voc-class-annotations")
        self.voc_classes        = json.open_json(json_path + "voc-classes")
        self.images             = list(self.annotations_object)

    def visualize(self,
                    images  = 'Default',
                    task    = 'Object', 
                    output_img_path = None, 
                    classes = 'Default', 
                    shuffle = False, 
                    default_t = None):

        if images == 'Default':
            images = self.images
        if classes == 'Default':
            classes = self.voc_classes

        view = [i for i in images if set(self.annotations_class[i]) & set(classes)]

        if shuffle:
            random.shuffle(view)

        if task == 'Object':
            self.visu_object(view, classes, output_img_path, default_t)
        elif task == 'Class':
            self.visu_class(view, classes, output_img_path, default_t)


    def visu_object(self, view, classes, output_img_path, default_t):
        N = len(view) - 1

        i = 0
        while True:
            img_name = view[i]
            img, true_sgms, _ = load(img_name, mode = 'Object')
            img_cv2 = im.pil_to_cv2(img)
            
            print("=============================================================================")    
            display_save(img_name, [img_cv2], output_img_path = output_img_path)
            
            j = 0
            while True:
                if j >= len(self.annotations_object[img_name]):
                    break
                c, _, cbbox = self.annotations_object[img_name][j]

                # skip unwanted classes
                if c not in classes:
                    j += 1
                    continue

                true_sgm    = m.true_mask(true_sgms, j + 1)
                cbbox_mask  = im.cbbox_mask(true_sgm.shape[:2], cbbox)
                
                pred_class, prob, _               = self.camnet.get_top1(img)
                pred_class_voc, prob_voc, img_cam = self.camnet.get_top_voc_to_imagenet(img, c)
                img_cam = im.bitwise_and(img_cam, cbbox_mask)
                
                print(f'Voc class: {c}\n')
                print(f'Top1 imagenet class:\n   {pred_class} with probability {prob:.4f}\n')
                print(f'Top imagenet class being associated to the given voc class \"{c}\":\n' +
                      f'   {pred_class_voc} with probability {prob_voc[1]:.4f} (top {prob_voc[0]})\n')
                
                display_save(f'{img_name}_{j+1}_crop', 
                            [im.bitwise_and(img_cv2, cbbox_mask)],
                            output_img_path = output_img_path)
                display_save(f'{img_name}_{j+1}{c}_cam',
                            [cam.heat_map(img_cv2, img_cam)],
                            output_img_path = output_img_path)
                display_save(f'{img_name}_{j+1}_true',
                            [im.f1_to_f255(true_sgm)],
                            output_img_path = output_img_path)
                
                while True:
                    print("")
                    print(f'|| technique browsing ||'.upper())
                    k = input("Type:\n - \"0\" to run grabCut \n - \"1\" to run grabCut-cam in PF_PB mode\n" +  
                            " - \"2\" to run grabCut-cam in F_PF mode \n - \"3\" to run grabCut-cam in F_PB mode\n" +
                            " - \"quit\" to quit the visualization \n" +
                            " - Anything else to move to another object or image \n" )
                    
                    if k == 'quit':
                        return

                    if k not in [str(i) for i in range(4)]:
                        break
                        
                    k = int(k)    
                    
                    if k == 0:
                        display_save(f'{img_name}_{j+1}{c}',
                                    *visu_grabcut(img, true_sgm, cbbox),
                                    output_img_path = output_img_path)
                    else:
                        t = default_t if default_t is not None else get_threshold()
                        mode = {1: 'PF_PB', 2: 'F_PF', 3: 'F_PB'}
                        display_save(f'{img_name}_{j+1}{c}',
                                    *visu_grabcut_cam(img, true_sgm, img_cam, t, mode = mode[k], cbbox = cbbox),
                                    output_img_path = output_img_path)
                        
                j = browse(j, len(self.annotations_object[img_name])-1, 'object')
                if j == -1:
                    break
                if j == -2:
                    return

            if j >= len(self.annotations_object[img_name]):
                print("No more objects\n")

            i = browse(i, N, 'image')
            if i == -1:
                break
            if j == -2:
                return

            clear = input("Type:\n - \"clear\" to clear the cell outputs \n - Anything else to keep them \n")
            if clear == "clear":
                clear_output(wait=True)  
            print("")

    def visu_class(self, view, classes, output_img_path, default_t):
        N = len(view) - 1

        i = 0
        while True:
            img_name = view[i]
            img, true_sgms, _ = load(img_name, mode = 'Class')
            img_cv2 = im.pil_to_cv2(img)
            
            print("=============================================================================")    
            display_save(img_name, [img_cv2], output_img_path = output_img_path)
            
            j = 0
            while True:
                if j >= len(self.annotations_class[img_name]):
                    break
                c = self.annotations_class[img_name][j]

                # skip unwanted classes
                if c not in classes:
                    j += 1
                    continue

                true_sgm    = m.true_mask(true_sgms, self.voc_classes.index(c) + 1)
                
                pred_class, prob, _               = self.camnet.get_top1(img)
                pred_class_voc, prob_voc, img_cam = self.camnet.get_top_voc_to_imagenet(img, c)
                
                print(f'Voc class: {c}\n')
                print(f'Top1 imagenet class:\n   {pred_class} with probability {prob:.4f}\n')
                print(f'Top imagenet class being associated to the given voc class \"{c}\":\n' +
                      f'   {pred_class_voc} with probability {prob_voc[1]:.4f} (top {prob_voc[0]})\n')
                
                display_save(f'{img_name}_{c}_true',
                            [cam.heat_map(img_cv2, img_cam)],
                            output_img_path = output_img_path)
                display_save(f'{img_name}_{c}_true',
                            [im.f1_to_f255(true_sgm)],
                            output_img_path = output_img_path)
                
                while True:
                    print("")
                    print(f'|| technique browsing ||'.upper())
                    k = input("Type:\n - \"0\" to run cam \n - \"1\" to run grabCut-cam in PF_PB mode\n" +  
                            " - \"2\" to run grabCut-cam in F_PF mode \n - \"3\" to run grabCut-cam in F_PB mode\n" +
                            " - \"quit\" to quit the visualization \n" +
                            " - Anything else to move to another class or image \n" )

                    if k == 'quit':
                        return
                    
                    if k not in [str(i) for i in range(4)]:
                        break
                        
                    k = int(k)

                    t = default_t if default_t is not None else get_threshold()
                    
                    if k == 0:
                        display_save(f'{img_name}_{c}',
                                    *visu_cam(img, true_sgm, img_cam, t),
                                    output_img_path = output_img_path)
                    else:
                        
                        mode = {1: 'PF_PB', 2: 'F_PF', 3: 'F_PB'}
                        display_save(f'{img_name}_{c}_true',
                                    *visu_grabcut_cam(img, true_sgm, img_cam, t, mode = mode[k]),
                                    output_img_path = output_img_path)
                        
                j = browse(j, len(self.annotations_class[img_name])-1, 'class')
                if j == -1:
                    break
                if j == -2:
                    return
                
            if j >= len(self.annotations_class[img_name]):
                print("No more classes\n")

            i = browse(i, N, 'image')
            if i == -1:
                break
            if j == -2:
                return

            clear = input("Type:\n - \"clear\" to clear the cell outputs \n - Anything else to keep them \n")
            if clear == "clear":
                clear_output(wait=True)
            print("") 
        