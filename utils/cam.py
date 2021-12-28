import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models, transforms

import numpy as np
import cv2
import os

import utils.json as json
import utils.path as path

cam_dir = os.path.dirname(os.path.realpath(__file__))
json_path = path.goback_from_current_dir(1, cam_dir) + 'json\\'

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

class Cam:
    def __init__(self, model = 'squeezenet11'):
        self.classes = json.open_json(json_path + 'imagenet-classes')
        self.voc_to_imagenet = json.open_json(json_path + 'voc-to-imagenet-classes')
        self.net = CamExtension(model = model)
        self.net.eval()

    def feed_img(self, img):
        logit, cams = self.net(Variable(PREPROCESS(img).unsqueeze(0)))
        return logit, cams

    #def get(self, img, img_class):
    #    _, cams = self.feed_img(img)

    #    return cams[0][self.classes.index(img_class)].detach().numpy()

    #def get_super(self, img, img_super_classes):
    #    cams = [self.get(img, c) for c in img_super_classes]
    #    return np.mean(cams, axis = 0)

    def get_top(self, img, voc_class):
        imagenet_classes = self.voc_to_imagenet[voc_class]

        logit, cams = self.feed_img(img)

        h_x = F.softmax(logit, dim = 1).data.squeeze()
        _, idx = h_x.sort(0, True)
        idx = idx.numpy()

        i = 0
        c = self.classes[idx[i]]
        while c not in imagenet_classes:
            i += 1
            c = self.classes[idx[i]]

        return cams[0][idx[i]].detach().numpy()

def cam_process(img, cam, size_upsample = (256, 256)):
    cam = cam - np.min(cam)
    cam = cam / np.max(cam) 
    cam = cv2.resize(cam, size_upsample)
    _, height, width = img.shape
    cam = cv2.resize(cam, (width, height))

    return cam

def heat_map(img, cam):
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(np.uint8(255 * cam), (width, height)), cv2.COLORMAP_JET)
    return heatmap * 0.3 + img * 0.5

def anchoring_fgd(mask, coord):
    if mask[coord] % 2 == 0:
        mask[coord] = 1
    return mask

def cam_to_gcmask(cam, threshold, bin = False):
    # BGD, 0 |0| PR_BGD, 2 |threshold| PR_FGD, 3 |1| FGD, 1
    max_coord = np.unravel_index(cam.argmax(), cam.shape)

    mask = np.digitize(cam, np.array([0.0, threshold, 1.0]), right = True)
    mask = (2*mask - mask//2) % 4 # maps [0, 1, 2, 3] to [0, 2, 3, 1]
    mask = anchoring_fgd(mask, max_coord).astype('uint8')
    if bin:
        return (mask % 2).astype(bool)
    else:
        return mask
