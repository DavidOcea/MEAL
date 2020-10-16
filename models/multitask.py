import torch.nn as nn
from . import backbones
import numpy as np
import torch
from .label_smoothing import LSR

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class MultiTaskWithLoss(nn.Module):
    def __init__(self, backbone, num_classes, feature_dim, criterion=None):
        super(MultiTaskWithLoss, self).__init__()
        self.basemodel = backbones.__dict__[backbone]()
        self.criterion = LSR()
        self.num_tasks = len(num_classes)
        self.fcs = nn.ModuleList([nn.Linear(feature_dim, num_classes[k]) for k in range(self.num_tasks)])
            
    def forward(self,input,target=None,slice_idx=None,eval_mode=False,mixup_criterion=None,mixup_data=None, mix_alpha=None):
        feature_maps = []
        feature = self.basemodel(input)
        out = [self.fcs[k](feature) for k in range(self.num_tasks)]
        
        if slice_idx is not None:
            x = [self.fcs[k](feature[slice_idx[k]:slice_idx[k+1], ...]) for k in range(self.num_tasks)] 
            target_slice = [target[slice_idx[k]:slice_idx[k+1]] for k in range(self.num_tasks)]
            los = [self.criterion(xx, tg) for xx, tg in zip(x, target_slice)]
            return out,los
        else:
            return out

        # return out,los