# Contains Class Activation Map related code and CamExtension and Cam classes definitions.


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models, transforms

import numpy as np
import cv2

import utils.json as json
import utils.path as path
import utils.image as im

root_path = path.goback_from_current_dir(1)
json_path = root_path + 'json\\'

# https://sebastianraschka.com/faq/docs/fc-to-conv.html
def conv1x1_from_lin(weight, bias):
    weight, bias = weight.data, bias.data
    
    num_classes, nc = weight.shape
    conv1x1 = nn.Conv2d(in_channels = nc, 
                        out_channels = num_classes, kernel_size = (1, 1))
    conv1x1.weight.data = weight.view(num_classes, nc, 1, 1)
    conv1x1.bias.data = bias
    return conv1x1

class CamExtension(nn.Module):
    def __init__(self, model):
        super(CamExtension, self).__init__()
        self.model = model
        
        if model   == 'squeezenet11':
            self.net = models.squeezenet1_1(pretrained = True)
            self.features = self.net.features
            self.conv1x1 = self.net.classifier[:2]
            self.avgpool = self.net.classifier[2:]
            
        elif model == 'resnet18':
            self.net = models.resnet18(pretrained = True)
            self.features = nn.Sequential(*list(self.net.children())[:-2])
            weight_softmax, bias_softmax = list(self.net.parameters())[-2:]
            self.conv1x1 = conv1x1_from_lin(weight_softmax, bias_softmax)
            self.avgpool = self.net.avgpool
            
        elif model == 'densenet161':
            self.net = models.densenet161(pretrained = True)
            self.features = self.net.features
            weight_softmax, bias_softmax = list(self.net.parameters())[-2:]
            self.conv1x1 = conv1x1_from_lin(weight_softmax, bias_softmax)
            self.classifier = self.net.classifier
        
    def forward(self, x):
        x = self.features(x)
        
        if self.model   == 'squeezenet11':
            cams = x = self.conv1x1(x)
            x = self.avgpool(x)
            
        elif self.model == 'resnet18':
            cams = x = self.conv1x1(x)
            x = self.avgpool(x)
            
        elif self.model == 'densenet161':
            x = F.relu(x, inplace = True)
            cams = x = self.conv1x1(x)
            x = F.adaptive_avg_pool2d(x, (1, 1))
        
        return torch.flatten(x, 1), cams

PREPROCESS = transforms.Compose([
   transforms.Resize((224, 224)),
   transforms.Normalize(
   mean = [0.485, 0.456, 0.406],
   std =  [0.229, 0.224, 0.225])])

NET_CLASSES        = json.open_json(json_path + 'imagenet-classes')
NET_SIMPLE_CLASSES = json.open_json(json_path + 'imagenet-simple-classes')
VOC_CLASSES        = json.open_json(json_path + 'voc-classes')
VOC_TO_NET         = json.open_json(json_path + 'voc-to-imagenet-classes')

class Cam:
    def __init__(self, model = 'squeezenet11'):
        self.net = CamExtension(model = model)
        self.net.eval()

    def feed_img(self, img):
        logit, cams = self.net(Variable(PREPROCESS(img).unsqueeze(0)))
        return logit, cams

    # Returns the top1 class and associated probability and CAM
    def get_top1(self, img):
        logit, cams = self.feed_img(img)

        h_x = F.softmax(logit, dim = 1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        idx = idx.numpy()

        simple_class = NET_SIMPLE_CLASSES[idx[0]]
        proba        = probs.numpy()[0]
        cam          = cam_process(img, cams[0][idx[0]].detach().numpy())

        return simple_class, proba, cam


    # Given a VOC class, search for the best associated 
    # (according to the VOC-Imagenet mapping) Imagenet class.
    # Returns the Imagenet class and associated, ranking (top n class), probability and CAM
    def get_top_voc_to_imagenet(self, img, voc_class):
        associated_imagenet_classes = VOC_TO_NET[voc_class]

        logit, cams = self.feed_img(img)

        h_x = F.softmax(logit, dim = 1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        idx = idx.numpy()

        i = 0
        c = NET_CLASSES[idx[i]]
        while c not in associated_imagenet_classes:
            i += 1
            c = NET_CLASSES[idx[i]]

        simple_class = NET_SIMPLE_CLASSES[NET_CLASSES.index(c)]
        rank_proba   = ((i+1), probs.numpy()[i])
        cam          = cam_process(img, cams[0][idx[i]].detach().numpy())

        return simple_class, rank_proba, cam

def cam_process(img, cam, size_upsample = (256, 256)):
    cam = cam - np.min(cam)
    cam = cam / np.max(cam) 
    cam = cv2.resize(cam, size_upsample)
    _, height, width = img.shape
    cam = cv2.resize(cam, (width, height))

    return cam

def heat_map(img_cv2, cam, heat_f = 0.3, img_f = 0.5):
    height, width, _ = img_cv2.shape
    heatmap = cv2.applyColorMap(cv2.resize(np.uint8(255 * cam), (width, height)), cv2.COLORMAP_JET)

    return im.rgb_swap_bgr(heatmap * heat_f + img_cv2 * img_f)

def cam_to_gcmask(cam, t0, t1, t2):
    # BGD, 0  ||t0||  PR_BGD, 2  ||t1||  PR_FGD, 3  ||t2||  FGD, 1
    max_coord = np.unravel_index(cam.argmax(), cam.shape)
    min_coord = np.unravel_index(cam.argmin(), cam.shape)

    # The point is to be efficient, thus the weird formula at line 2
    # (A transformation with basic operations is better than manual case mapping)
    mask = np.digitize(cam, np.array([t0, t1, t2]), right = True)
    mask = (2*mask - mask//2) % 4 # maps [0, 1, 2, 3] to [0, 2, 3, 1].

    if mask[max_coord] % 2 == 0:
        mask[max_coord] = 1       # We should ensure that there at least on FGD pixel for anchoring
    if mask[min_coord] % 2 == 1:
        mask[min_coord] = 0       # We should ensure that there at least on BGD pixel for anchoring

    return mask.astype('uint8')
