import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data
from torchvision import models, transforms, datasets
import numpy as np



# use the ImageNet transformation
transform = transforms.Compose([transforms.Resize((224, 224)), 
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# define a 1 image dataset
dataset = datasets.ImageFolder(root='./data/Elephant/', transform=transform)

# define the dataloader to load that single image
dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)

class resnet18_cam(nn.Module):
    def __init__(self):
        super(resnet18_cam, self).__init__()

        self.net = models.resnet18(pretrained=True)
        
        self.features_conv = self.net.layer4[-1]
        
        self.fc = self.net.fc
         
        self.avgpool = self.net.avgpool
        
    def forward(self, x):
        x = self.features_conv(x)

        x = self.fc(x)

        x = self.avgpool(x)
        return x
    
    def get_cams(self, x):
        return self.fc(x)