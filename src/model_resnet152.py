import os
import torch
import torch.nn as nn
import torchvision.models as models

def resnet152(state_dict):
    pretrain = True if not os.path.isfile(state_dict) else False
    model = models.resnet152(pretrained = pretrain)

    model.conv1 = nn.Conv2d(1, 64, kernel_size = (7,7), stride = (2,2), padding = (3,3), bias = False)
    model.fc = nn.Linear(in_features=2048, out_features=1, bias=True)
    
    if not pretrain:   
        model.load_state_dict(torch.load(state_dict))
    return model