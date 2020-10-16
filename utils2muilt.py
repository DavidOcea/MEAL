'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

from models import *

import torch.nn as nn
import torch.nn.init as init
import torch.backends.cudnn as cudnn


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        if isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        if isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


# _, term_width = os.popen('stty size', 'r').read().split()
term_width = 171
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def get_model(args, config, device):
    teachers = []

    model_map = {"vgg19": VGG, "vgg19_BN": VGG, "resnet18": ResNet18, 'preactresnet18': PreActResNet18,
                 "googlenet": GoogLeNet, "densenet121": DenseNet121,"densenet_cifar": densenet_cifar,
                 "resnext": ResNeXt29_2x64d, "mobilenet": MobileNet, "dpn92": DPN92, "mul_mult_prun8_gpu":'mult_prun8_gpu',
                 "mul_mult_prun8_gpu_prun":'mult_prun8_gpu_prun','mul_multnas5_gpu':'multnas5_gpu',
                 'mul_multnas5_gpu_prun':'multnas5_gpu_prun','mul_multnas5_gpu_2_18':'multnas5_gpu_2_18',
                 'mul_se_resnext101_32x4d':'se_resnext101_32x4d','mul_se_resnet50_18':'se_resnet50_18'}

    # Add teachers models into teacher model list
    for t in args.teachers:
        if t in model_map:
            if "mul" not in t:
                net = model_map[t](args)
                net.__name__ = t
            else:
                net = MultiTaskWithLoss(backbone=t[4:],
                                    num_classes=args.num_classes, 
                                    feature_dim=args.feature_dim)
                net.__name__ = t
            teachers.append(net)

    assert len(teachers) > 0, "teachers must be in %s" % " ".join(model_map.keys)

    # Initialize student model

    assert args.student in model_map, "students must be in %s" % " ".join(model_map.keys)
    if "mul" not in args.student:
        student = model_map[args.student](args)
    else:
        
        student = MultiTaskWithLoss(backbone=args.student[4:],
                                num_classes=args.num_classes, 
                                feature_dim=args.feature_dim)

    # Model setup

    if device == "cuda":
        cudnn.benchmark = True

    for i, teacher in enumerate(teachers):
        for p in teacher.parameters():
            p.requires_grad = False #False
        teacher = teacher.to(device)
        if device == "cuda":
            teachers[i] = torch.nn.DataParallel(teacher)
            teachers[i].__name__ = teacher.__name__

    # Load parameters in teacher models
    for teacher in teachers:
        if teacher.__name__ != "shake_shake":
            if teacher.__name__ == 'mul_mult_prun8_gpu':
                checkpoint = torch.load('./pretrain_models/0713ckpt_epoch_47.pth.tar')
            elif teacher.__name__ == 'mul_multnas5_gpu':
                checkpoint = torch.load('./pretrain_models/0710ckpt_epoch_38.pth.tar')
            elif teacher.__name__ == 'mul_multnas5_gpu_2_18':
                checkpoint = torch.load('./pretrain_models/multnas5_gpu_18_1009_28.pth.tar')
            elif teacher.__name__ == 'mul_se_resnext101_32x4d':
                checkpoint = torch.load('./pretrain_models/0805_se_res101_51.pth.tar')
            elif teacher.__name__ == 'mul_se_resnet50_18':
                checkpoint = torch.load('./pretrain_models/se_resnet1010_11.pth.tar')
            #原生的
            # model_dict = teacher.state_dict()
            # pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            # model_dict.update(pretrained_dict)
            
            #修改后的load
            state_dict = checkpoint['state_dict']
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # head = k[:7]
                if 'last_linear' in k:
                    continue  # remove `module.`
                else:
                    name = k
		        # name = 'module.{}'.format(k)
                new_state_dict[name] = v
            state_dict = new_state_dict
        
            teacher.load_state_dict(state_dict) #model_dict
            # print("teacher %s acc: ", (teacher.__name__, checkpoint['acc']))
        
        #--原来的--
        # if teacher.__name__ != "shake_shake":
        #     checkpoint = torch.load('/workspace/mnt/storage/yangdecheng/yangdecheng/models/checkpoint/%s/ckpt.t7' % teacher.__name__)
        #     model_dict = teacher.state_dict()
        #     pretrained_dict = {k: v for k, v in checkpoint['net'].items() if k in model_dict}
        #     model_dict.update(pretrained_dict)
        #     teacher.load_state_dict(model_dict)
        #     print("teacher %s acc: ", (teacher.__name__, checkpoint['acc']))

    student = student.to(device)
    if device == "cuda":
    #     out_dims = student.out_dims
        student = torch.nn.DataParallel(student)
    #     student.out_dims = out_dims

    if args.teacher_eval:
        for teacher in teachers:
            teacher.eval()

    return teachers, student
