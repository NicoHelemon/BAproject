import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import json
from PIL import Image
import xml.etree.ElementTree as ET


def img_names(input_dir):
    return [name[:-4] for name in os.listdir(input_dir)]

def img_annotations(input_dir, img_names):
    img_annotations = {}
    for name in img_names:
        tree = ET.parse(input_dir + name + '.xml')
        path = './/object/bndbox//'
        xmin = tree.findall(path + 'xmin')
        ymin = tree.findall(path + 'ymin')
        xmax = tree.findall(path + 'xmax')
        ymax = tree.findall(path + 'ymax')
        
        obj_class = [c.text for c in tree.findall('.//object/name')]
        obj_difficult = [int(d.text) for d in tree.findall('.//object/difficult')]
        
        def gc_bbox(bbox):
            bbox = [coord - 1 for coord in bbox]
            return list(corner_to_delta(bbox))
        
        obj_bbox = [gc_bbox([int(coord.text) for coord in bbox]) 
                            for bbox in zip(xmin, ymin, xmax, ymax)]
        
        img_annotations[name] = [(c, d, b) for (c, d, b) in zip(obj_class, obj_difficult, obj_bbox)]
        
    return img_annotations

def save_json(name, array):
    file = open(name + '.json', "w")
    json.dump(array, file, indent=4)
    file.close()
    

    
def draw_bbox(image, bbox, color = (255, 0, 0), thickness = 2):
    bbox = delta_to_corner(bbox)
    top_left = bbox[:2]
    bot_right = bbox[2:]
    return cv2.rectangle(image.copy(), top_left, bot_right, color, thickness)
                        
def plt_show(image, mode = 'color'):
    if mode == 'color':
        plt.imshow(cv2.cvtColor(image.astype('uint8'), cv2.COLOR_BGR2RGB))
        plt.show()
    if mode == 'gray' or mode == 'grey':
        plt.imshow(image.astype('uint8'), cmap='gray', vmin=0, vmax=255)
        plt.show()
        
def idx_to_color(n):
    nb2 = [int(b) for b in list(format(n,'06b'))]
    return [nb2[i] * 64 + nb2[i+3] * 128 for i in range(3)]

def color_to_idx(c):
    def f(c, i):
        p = 1 + i // 3
        return ((c[i%3] % (128 *p)) // (64 *p) ) * (2 ** (5-i))
    
    return sum([f(c, i) + f(c, i+3) for i in range(3)])

MAPPING = [idx_to_color(i) for i in range(1, 64)]

def grabcut(image, mask, iterCount = 4, mode = 'RECT'):
    cv2.setRNGSeed(0)
    
    if mode == 'RECT':
        mask, _, _ = cv2.grabCut(
        image, 
        np.zeros(image.shape[:2], dtype="uint8"), 
        mask, 
        np.zeros((1, 65), dtype="float"),
        np.zeros((1, 65), dtype="float"), 
        iterCount=iterCount, 
        mode=cv2.GC_INIT_WITH_RECT)
     
    elif mode == 'MASK':
        mask = mask.copy()
        mask, _, _ = cv2.grabCut(
        image, 
        mask, 
        None, 
        np.zeros((1, 65), dtype="float"),
        np.zeros((1, 65), dtype="float"), 
        iterCount=iterCount, 
        mode=cv2.GC_INIT_WITH_MASK)
    
    binary_mask255 = mask_to_gray_img(mask, nuance = False)
    
    binary_mask01 = mask.astype('uint8') % 2
    
    return mask, binary_mask255, binary_mask01

def mask_to_gray_img(mask, nuance = True):
    mask = mask.astype('uint8')
    if not nuance:
        mask = mask % 2 # merge probable and definite values together
    mask[mask == 0] = 0    # definite background
    mask[mask == 1] = 255  # definite foreground
    mask[mask == 2] = 85   # probable background
    mask[mask == 3] = 170  # probable foreground
    return mask

black = np.array([0, 0, 0])
beige = np.array([192, 224, 224])

def img_3d_to_1d_c(img):
    return img.reshape(-1, img.shape[2])

def drop_background_and_undef(img_1d_c):
    img_1d_c = img_1d_c[~ np.equal(img_1d_c, black).all(1)]
    img_1d_c = img_1d_c[~ np.equal(img_1d_c, beige).all(1)]
    return img_1d_c
    
def pil_to_cv2(img):
    cv2_image = np.moveaxis(img, 0, -1)
    return (cv2_image[:, :, ::-1] * 255).astype('uint8')

def extract_colored_mask(img, color):
    img[np.all(img == color, axis=-1)] = [1, 1, 1]
    img[~np.all(img == [1, 1, 1], axis=-1)] = [0, 0, 0]
    return img[:, :, 1]

def IoU(mask1, mask2):
    return (mask1 & mask2).sum() / (mask1 | mask2).sum()

