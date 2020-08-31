import torch.nn as nn
from . import backbones
import numpy as np
import torch

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
        self.criterion = nn.CrossEntropyLoss()
        self.num_tasks = len(num_classes)
        #普通
        self.fcs = nn.ModuleList([nn.Linear(feature_dim, num_classes[k]) for k in range(self.num_tasks)])
        
        # 第一任务加空间注意力  --  要注意backbone里要去掉avg_pooling及之后的层，都搬到外面来
        # self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        # self.ca = ChannelAttention(feature_dim)
        # self.sa = SpatialAttention()
        # self.fcs = nn.ModuleList()
        # for k in range(self.num_tasks):
        #     # if k == 1:
        #     # self.fcs.append(nn.Sequential(self.sa,nn.Linear(feature_dim, num_classes[k]))) #sa
        #     self.fcs.append(nn.Sequential(self.ca,self.sa,nn.Linear(feature_dim, num_classes[k])))
            # else:
                # self.fcs.append(nn.Linear(feature_dim, num_classes[k]))
        self.layers = nn.AdaptiveMaxPool1d(10)
            
    def forward(self,input,target=None,slice_idx=None,eval_mode=False,mixup_criterion=None,mixup_data=None, mix_alpha=None):

        # if mix_alpha is not None:
        #     inputs = [input[slice_idx[k]:slice_idx[k+1]] for k in range(self.num_tasks)]
        #     target_slice = [target[slice_idx[k]:slice_idx[k+1]] for k in range(self.num_tasks)]
        #     mix_data = [mixup_data(ins,tag,mix_alpha, True) for ins,tag in zip(inputs,target_slice)]
        #     m_feture = [self.basemodel(ins[0]) for ins in mix_data]
        #     #普通
        #     y_p = [self.fcs[k](m_feture[k]) for k in range(self.num_tasks)]
            
        #     # 第一任务加空间注意力
        #     # y_p = []
        #     # for k in range(self.num_tasks):
        #     #     if k == 0:
        #     #         # y_p.append(self.fcs[k][1](torch.flatten(self.global_avg_pooling(self.fcs[k][0](m_feture[k])*m_feture[k]), start_dim=1, end_dim=-1))) #sa
        #     #         y_p.append(self.fcs[k][2](torch.flatten(self.global_avg_pooling(self.fcs[k][1](self.fcs[k][0](m_feture[k])*m_feture[k])*(self.fcs[k][0](m_feture[k])*m_feture[k])), start_dim=1, end_dim=-1)))
        #     #     else:
        #     #         y_p.append(self.fcs[k](torch.flatten(self.global_avg_pooling(m_feture[k]), start_dim=1, end_dim=-1)))

        #     return [mixup_criterion(self.criterion,xx,md[1],md[2],md[3]) for xx, md in zip(y_p, mix_data)]

        # else:
        feature_maps = []
        feature = self.basemodel(input)
        out = [self.fcs[k](feature) for k in range(self.num_tasks)]

        x = [self.fcs[k](feature[slice_idx[k]:slice_idx[k+1], ...]) for k in range(self.num_tasks)] 
        target_slice = [target[slice_idx[k]:slice_idx[k+1]] for k in range(self.num_tasks)]
        los = [self.criterion(xx, tg) for xx, tg in zip(x, target_slice)]

        return out,los
            # if eval_mode:
            #     #普通
            #     x = [self.fcs[k](feature) for k in range(self.num_tasks)]

            #     # 第一任务加空间注意力
            #     # x = []
            #     # for k in range(self.num_tasks):
            #     #     # if k == 1:
            #     #     # x.append(self.fcs[k][1](torch.flatten(self.global_avg_pooling(self.fcs[k][0](feature)*feature), start_dim=1, end_dim=-1))) #sa
            #     #     x.append(self.fcs[k][2](torch.flatten(self.global_avg_pooling(self.fcs[k][1](self.fcs[k][0](feature)*feature)*(self.fcs[k][0](feature)*feature)), start_dim=1, end_dim=-1)))
            #     #     else:
            #     #         x.append(self.fcs[k](torch.flatten(self.global_avg_pooling(feature), start_dim=1, end_dim=-1)))

            #     return x
            # else:
            #     #普通
            #     x = [self.fcs[k](feature[slice_idx[k]:slice_idx[k+1], ...]) for k in range(self.num_tasks)] 
                
            #     # 第一任务加空间注意力
            #     # x = []
            #     # for k in range(self.num_tasks):
            #     #     # if k == 1:
            #     #     # x.append(self.fcs[k][1](torch.flatten(self.global_avg_pooling(self.fcs[k][0](feature[slice_idx[k]:slice_idx[k+1], ...])*feature[slice_idx[k]:slice_idx[k+1], ...]), start_dim=1, end_dim=-1))) #sa
            #     #     x.append(self.fcs[k][2](torch.flatten(self.global_avg_pooling(self.fcs[k][1](self.fcs[k][0](feature[slice_idx[k]:slice_idx[k+1], ...])*feature[slice_idx[k]:slice_idx[k+1], ...])*(self.fcs[k][0](feature[slice_idx[k]:slice_idx[k+1], ...])*feature[slice_idx[k]:slice_idx[k+1], ...])), start_dim=1, end_dim=-1)))
            #     #     else:
            #     #         x.append(self.fcs[k](torch.flatten(self.global_avg_pooling(feature[slice_idx[k]:slice_idx[k+1], ...]), start_dim=1, end_dim=-1)))
    
            #     target_slice = [target[slice_idx[k]:slice_idx[k+1]] for k in range(self.num_tasks)]
            #     return [self.criterion(xx, tg) for xx, tg in zip(x, target_slice)]
        # out = [out[k].view(out[k].size(0), -1) for k in range(self.num_tasks)]
        # return [feature_maps.append(self.layers(out[k].view(out[k].size(0), -1))) for k in range(self.num_tasks)]
