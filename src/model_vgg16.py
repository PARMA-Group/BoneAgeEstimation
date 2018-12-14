import os
import torch
import torch.nn as nn
import torchvision.models as models

def vgg16(state_dict):
    pretrain = True if not os.path.isfile(state_dict) else False
    model = models.vgg16(pretrained = pretrain)
    
    model.features[0] = nn.Conv2d(1, 64, kernel_size = 3)
    model.classifier[0] = nn.Linear(18432, 4096) 
    model.classifier[6] = nn.Linear(4096, 1)

    if not pretrain:   
        model.load_state_dict(torch.load(state_dict))      
    return model
