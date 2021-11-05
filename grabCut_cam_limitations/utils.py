import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import json
from PIL import Image

# Let's denote the horizontal axis by X and the vertical axis by Y.
# There are two image formats, namely (X, Y) or (Y, X).
# - (X, Y) means that the first axis of the 2D array / tuple represents the horizontal axis
# - (Y, X) means that the first axis of the 2D array / tuple represents the vertical axis
# 
# In CV2, we have the (X, Y) format while in numpy we have the (Y, X) format

# Rectangles are specified by 4 values ; there are two formats:
# - The double-corner format (CC) i.e. top left corner coordinates and bottom right
#   coordinates: (topL_x, topL_y, botR_x, botR_y)
# - The corner-delta format (CΔ) i.e. top left corner coordinates and horizontal and
#   vertical deltas : (topL_x, topL_y, delta_x, delta_y)
# We have the relation (botR_x, botR_y) = (topL_x, topL_y) + (delta_x, delta_y)

def resize_target_height(image, target_height = 500):
    height, width = image.shape[:2]
    
    dim = (int(width * target_height/height), target_height)
    
    return cv2.resize(image, dim)

def resize_target_area(image, target_area = 250000):
    height, width = image.shape[:2]
    area = height * width
    
    factor = np.sqrt(target_area / area)
    
    dim = (int(width * factor), int(height * factor))
    
    return cv2.resize(image, dim)

# IN    deltaRec: (X, Y) ; CΔ
# OUT   cornerRec: (X, Y) ; CC
def deltaRecToCornerRec(deltaRec):
    topL_x, topL_y, dx, dy = deltaRec
    botR_x, botR_y = topL_x + dx, topL_y + dy
    return topL_x, topL_y, botR_x, botR_y

# IN    image: (Y, X), bbox: (X, Y) ; CΔ
# OUT   cropped image: (Y, X)
def crop(image, bbox):
    bbox = deltaRecToCornerRec(bbox)
    return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

# IN    image: (Y, X)
# OUT   bbox: (X, Y) ; CΔ
def boundingBox(image):
    r = cv2.selectROI(image)
    cv2.destroyAllWindows()
    return r

# IN    image: (Y, X)
# OUT   bbox: (X, Y) ; CΔ
def fullBoundingBox(image):
    return 0, 0, image.shape[1] - 1, image.shape[0] - 1

# IN    image: (Y, X)
# OUT   image with highlighted bbox: (X, Y)
def drawBbox(image, bbox, color = (255, 0, 0), thickness = 3):
    bbox = deltaRecToCornerRec(bbox)
    topL = bbox[:2]
    botR = bbox[2:]
    return cv2.rectangle(image.copy(), topL, botR, color, thickness)

def bboxToMask(image, bbox, background_val):
    mask = np.ones(image.shape[:2], dtype="uint8") * background_val
    topL_x, topL_y, dx, dy = bbox
    mask[topL_y:topL_y+dy, topL_x:topL_x+dx] = cv2.GC_PR_FGD
    return mask

# Applies cv2.grabCut algorithm to an image and a bounding box
# It returns "mask", i.e. a 2d-array of shape image (w/o the channel) whose pixels take
# value in {0,1,2,3} = {background, foreground, probable background, probable foreground} 
# It also returns the final true output mask, a 2d-array again of shape image whose pixels
# take value in {0,1} = {background, foreground}, which can be used with a bitwise operator.
def grabCut(image, mask, mode = 'RECT'):
    cv2.setRNGSeed(0) #https://stackoverflow.com/questions/60954921/python-opencv-doesnt-give-same-output-at-the-same-image
    
    # https://github.com/opencv/opencv/blob/master/modules/imgproc/src/grabcut.cpp
    
    if mode == 'RECT':
        mask, _, _ = cv2.grabCut(
        image, 
        np.zeros(image.shape[:2], dtype="uint8"), 
        mask, 
        np.zeros((1, 65), dtype="float"),
        np.zeros((1, 65), dtype="float"), 
        iterCount=10, 
        mode=cv2.GC_INIT_WITH_RECT)
     
    elif mode == 'MASK':
        mask = mask.copy()
        mask, _, _ = cv2.grabCut(
        image, 
        mask, 
        None, 
        np.zeros((1, 65), dtype="float"),
        np.zeros((1, 65), dtype="float"), 
        iterCount=10, 
        mode=cv2.GC_INIT_WITH_MASK)
    
    binaryMask = maskToGrayImage(mask, nuance = False)
    
    return mask, binaryMask

def maskToGrayImage(mask, nuance = True):
    mask = mask.astype('uint8')
    if not nuance:
        mask = mask % 2 # merge probable and definite values together
    mask[mask == 0] = 0    # definite background
    mask[mask == 1] = 255  # definite foreground
    mask[mask == 2] = 85   # probable background
    mask[mask == 3] = 170  # probable foreground
    return mask

def cv2Show(image, title = 'title'):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def pltShow(image, mode = 'color'):
    if mode == 'color':
        plt.imshow(cv2.cvtColor(image.astype('uint8'), cv2.COLOR_BGR2RGB))
        plt.show()
    if mode == 'gray' or mode == 'grey':
        plt.imshow(image.astype('uint8'), cmap='gray', vmin=0, vmax=255)
        plt.show()
        
def load(input_directory, images_name = None, with_bboxes = True):
    if images_name == None:
        images_name = []
        for name in os.listdir(input_directory):
            if name.endswith(".jpg"):
                images_name.append(name)

    images_cv2 = {}
    for name in images_name:
        images_cv2[name] = cv2.imread(input_directory + name)
        
    images_pil = {}
    for name in images_name:
        images_pil[name] = Image.open(input_directory + name)

    bboxes = None
    if with_bboxes:
        bboxes = json.loads(open(input_directory + "images_bboxes.json", "r").read())
    
    return images_name, images_cv2, images_pil, bboxes