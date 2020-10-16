'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from collections import OrderedDict
import random

import os
import argparse
import numpy as np

from models import *
from models import discriminator
from utils2muilt import progress_bar, get_model
from loss import *

from torch.utils.data import DataLoader
from data.transforms import RandomResizedCrop, Compose, Resize, CenterCrop, ToTensor, \
    Normalize, RandomHorizontalFlip, ColorJitter, Lighting
from data.datasets import GivenSizeSampler, FileListLabeledDataset, FileListDataset
from torch.autograd import Variable
from optimizer import adjust_learning_rate
import logging

# ================= Arugments ================ #

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--d_lr', default=0.1, type=float, help='discriminator learning rate')
parser.add_argument('--teachers', default='[\'shufflenetg2\']', type=str, help='teacher networks type')
parser.add_argument('--student', default='shufflenetg2', type=str, help='student network type')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--gpu_id', default='3', type=str, help='gpu id')
parser.add_argument('--gamma', default='[1,1,1,1,1]', type=str, help='')
parser.add_argument('--eta', default='[1,1,1,1,1]', type=str, help='')
parser.add_argument('--fc_out', default=1, type=int, help='if immediate output from fc-layer')
parser.add_argument('--loss', default="ce", type=str, help='loss selection')
parser.add_argument('--adv', default=1, type=int, help='add discriminator or not')
parser.add_argument('--name', default=None, type=str, help='the name of this experiment')
parser.add_argument('--pool_out', default="max", type=str, help='the type of pooling layer of output')
parser.add_argument('--out_layer', default="[-1]", type=str, help='the type of pooling layer of output')
parser.add_argument('--out_dims', default="[5000,1000,500,200,10]", type=str, help='the dims of output pooling layers')
parser.add_argument('--teacher_eval', default=0, type=int, help='use teacher.eval() or not')

# model config
parser.add_argument('--depth', type=int, default=26)
parser.add_argument('--base_channels', type=int, default=96)
parser.add_argument('--grl', type=bool, default=False, help="gradient reverse layer")

# run config
parser.add_argument('--outdir', type=str, default="results")
parser.add_argument('--seed', type=int, default=17)
parser.add_argument('--num_workers', type=int, default=7)

# optim config
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--base_lr', type=float, default=0.2)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--nesterov', type=bool, default=True)
parser.add_argument('--lr_min', type=float, default=0)
parser.add_argument('--meal_type', type=str, default="0")


args = parser.parse_args()

# ================= Config Collection ================ #

model_config = OrderedDict([
    ('depth', args.depth),
    ('base_channels', args.base_channels),
    ('input_shape', (1, 3, 32, 32)),
    ('n_classes', 10),
    ('out_dims', args.out_dims),
    ('fc_out', args.fc_out),
    ('pool_out', args.pool_out)
])

optim_config = OrderedDict([
    ('epochs', args.epochs),
    ('batch_size', args.batch_size),
    ('base_lr', args.base_lr),
    ('weight_decay', args.weight_decay),
    ('momentum', args.momentum),
    ('nesterov', args.nesterov),
    ('lr_min', args.lr_min),
])

data_config = OrderedDict([
    ('dataset', 'CIFAR10'),
])

run_config = OrderedDict([
    ('seed', args.seed),
    ('outdir', args.outdir),
    ('num_workers', args.num_workers),
])

config = OrderedDict([
    ('model_config', model_config),
    ('optim_config', optim_config),
    ('data_config', data_config),
    ('run_config', run_config),
])

print(args)

# ================= Initialization ================ #

os.environ['CUDA_VISIBLE_DEVICES']=args.gpu_id
device = 'cuda'
# device = 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# ================= Data Loader ================ #

print('==> Preparing data..')

args.data_root = ['/workspace/mnt/storage/yangdecheng/yangdecheng-data1/TR-NMA-09/CX_20200918',\
        '/workspace/mnt/storage/yangdecheng/yangdecheng-data1/TR-NMA-09/TK_20200918',\
        '/workspace/mnt/storage/yangdecheng/yangdecheng-data1/TR-NMA-09/ZR_20200918',\
        '/workspace/mnt/storage/yangdecheng/yangdecheng-data1/TR-NMA-09/TX_20200918',\
        '/workspace/mnt/storage/yangdecheng/yangdecheng-data1/TR-NMA-09/WM_20200918']  #TK_20200820/WM_20200820

args.data_root_val = ['/workspace/mnt/storage/yangdecheng/yangdecheng-data1/TR-NMA-09/CX_20200918',\
        '/workspace/mnt/storage/yangdecheng/yangdecheng-data1/TR-NMA-09/TK_20200918',\
        '/workspace/mnt/storage/yangdecheng/yangdecheng-data1/TR-NMA-09/ZR_20200918',\
        '/workspace/mnt/storage/yangdecheng/yangdecheng-data1/TR-NMA-09/TX_20200918',\
        '/workspace/mnt/storage/yangdecheng/yangdecheng-data1/TR-NMA-09/WM_20200918']

args.train_data_list = ['/workspace/mnt/storage/yangdecheng/yangdecheng-data1/TR-NMA-09/CX_20200918/txt/cx_train.txt',\
        '/workspace/mnt/storage/yangdecheng/yangdecheng-data1/TR-NMA-09/TK_20200918/txt/tk_train.txt',\
        '/workspace/mnt/storage/yangdecheng/yangdecheng-data1/TR-NMA-09/ZR_20200918/txt/zr_train.txt',\
        '/workspace/mnt/storage/yangdecheng/yangdecheng-data1/TR-NMA-09/TX_20200918/txt/tx_train.txt',\
        '/workspace/mnt/storage/yangdecheng/yangdecheng-data1/TR-NMA-09/WM_20200918/txt/wm_train.txt']

args.val_data_list = ['/workspace/mnt/storage/yangdecheng/yangdecheng-data1/TR-NMA-09/CX_20200918/txt/cx_val.txt',\
        '/workspace/mnt/storage/yangdecheng/yangdecheng-data1/TR-NMA-09/TK_20200918/txt/tk_val.txt',\
        '/workspace/mnt/storage/yangdecheng/yangdecheng-data1/TR-NMA-09/ZR_20200918/txt/zr_val.txt',\
        '/workspace/mnt/storage/yangdecheng/yangdecheng-data1/TR-NMA-09/TX_20200918/txt/tx_val.txt',\
        '/workspace/mnt/storage/yangdecheng/yangdecheng-data1/TR-NMA-09/WM_20200918/txt/wm_val.txt']

args.backends = 'mult_prun8_gpu'
args.feature_dim = 18
args.batchSize = [args.batch_size,args.batch_size,args.batch_size,args.batch_size,args.batch_size]
args.ngpu = len(args.gpu_id.split(','))

num_tasks = len(args.data_root)

trainset = []
for i in range(num_tasks): 
    if i == 1:
        trainset.append(FileListLabeledDataset(
    args.train_data_list[i], args.data_root[i],
    Compose([
    RandomResizedCrop(112,scale=(0.94, 1.), ratio=(1. / 4., 4. / 1.)), #scale=(0.7, 1.2), ratio=(1. / 1., 4. / 1.)
    RandomHorizontalFlip(),
    ColorJitter(brightness=[0.5,1.5], contrast=[0.5,1.5], saturation=[0.5,1.5], hue= 0),
    ToTensor(),
    Lighting(1, [0.2175, 0.0188, 0.0045], [[-0.5675,  0.7192,  0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948,  0.4203]]),
    Normalize([0.406, 0.456, 0.485], [0.225, 0.224, 0.229]),])))
    else:
        trainset.append(FileListLabeledDataset(
    args.train_data_list[i], args.data_root[i],
    Compose([
    RandomResizedCrop(112,scale=(0.7, 1.2), ratio=(1. / 1., 4. / 1.)),
    RandomHorizontalFlip(),
    ColorJitter(brightness=[0.5,1.5], contrast=[0.5,1.5], saturation=[0.5,1.5], hue= 0),
    ToTensor(),
    Lighting(1, [0.2175, 0.0188, 0.0045], [[-0.5675,  0.7192,  0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948,  0.4203]]),
    Normalize([0.406, 0.456, 0.485], [0.225, 0.224, 0.229]),])))

args.num_classes = [td.num_class for td in trainset]
train_longest_size = max([int(np.ceil(len(td) / float(bs))) for td, bs in zip(trainset, args.batchSize)])
train_sampler = [GivenSizeSampler(td, total_size=train_longest_size * bs, rand_seed=0) for td, bs in zip(trainset, args.batchSize)]
trainloader = [DataLoader(
                            trainset[k], 
                            batch_size=args.batchSize[k], 
                            shuffle=False,
                            num_workers=8, 
                            pin_memory=False, sampler=train_sampler[k]) for k in range(num_tasks)]

testset = [FileListLabeledDataset(
                args.val_data_list[i], args.data_root_val[i],
                Compose([
                 Resize((112,112)),
                ToTensor(),
                Normalize([0.406, 0.456, 0.485], [0.225, 0.224, 0.229]),]),) for i in range(num_tasks)]

test_longest_size = max([int(np.ceil(len(td) / float(bs))) for td, bs in zip(testset, args.batchSize)])
test_sampler = [GivenSizeSampler(td, total_size=test_longest_size * bs, rand_seed=0) for td, bs in zip(testset, args.batchSize)]
testloader = [DataLoader(
                    testset[k], 
                    batch_size=args.batchSize[k], 
                    shuffle=False,
                    num_workers=8, 
                    pin_memory=False,sampler=test_sampler[k]) for k in range(num_tasks)]

optim_config['steps_per_epoch'] = max([len(trainloader[k]) for k in range(num_tasks)])

# ================= Model Setup ================ #

args.teachers = eval(args.teachers)

print('==> Training', args.student if args.name is None else args.name)
print('==> Building model..')

# get models as teachers and students
teachers, student = get_model(args, config, device="cuda")

print("==> Teacher(s): ", " ".join([teacher.__name__ for teacher in teachers]))
print("==> Student: ", args.student)

# dims = [student.out_dims[i] for i in eval(args.out_layer)]
dims = [5,3,2,2,7]
print("dims:", dims)

update_parameters = [{'params': student.parameters()}]

if args.adv:
    discriminators = discriminator.Discriminators(dims, grl=args.grl)
    for d in discriminators.discriminators:
        d = d.to(device)
        if device == "cuda":
            d = torch.nn.DataParallel(d)
        update_parameters.append({'params': d.parameters(), "lr": args.d_lr})

print(args)

args.resume = 1
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if args.student == 'mul_mult_prun8_gpu_prun':
        checkpoint = torch.load('./pretrain_models/0820_e1-5_epoch_10.pth.tar')
    elif args.student == 'mul_multnas5_gpu_prun':
        checkpoint = torch.load('./pretrain_models/purn_20200717_5T_t2_20e.pth.tar')
    state_dict = checkpoint['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if 'gate' in k:
            continue
        # if head == 'module.':
        #     name = k[7:]  # remove `module.`
        else:
            name = k
		# name = 'module.{}'.format(k)
        new_state_dict[name] = v
    student.load_state_dict(new_state_dict) #,strict=False
    # start_epoch = checkpoint['epoch']

# ================= Loss Function for Generator ================ #

if args.loss == "l1":
    loss = F.l1_loss
elif args.loss == "l2":
    loss = F.mse_loss
elif args.loss == "l1_soft":
    loss = L1_soft
elif args.loss == "l2_soft":
    loss = L2_soft
elif args.loss == "ce":
    loss = CrossEntropy      # CrossEntropy for multiple classification

criterion = betweenLoss(eval(args.gamma), loss=loss)

g_loss = KLLoss()

# ================= Loss Function for Discriminator ================ #

if args.adv:
    discriminators_criterion = discriminatorLoss(discriminators, eval(args.eta))
else:
    discriminators_criterion = discriminatorFakeLoss()

# ================= Optimizer Setup ================ #

if args.student == "densenet_cifar":
    optimizer = optim.SGD(update_parameters, lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150 * min(2, len(teachers)), 250 * min(2, (len(teachers)))],gamma=0.1)
    print("nesterov = True")
elif args.student == "mobilenet":
    optimizer = optim.SGD(update_parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)  # nesterov = True, weight_decay = 1e-4，stage = 3, batch_size = 64
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150 * min(1, len(teachers)), 250 * min(1, (len(teachers)))],gamma=0.1)
else:
    optimizer = optim.SGD(update_parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)  # nesterov = True, weight_decay = 1e-4，stage = 3, batch_size = 64
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25,35,50],gamma=0.1) #学习率变化

# ================= Training and Testing ================ #

def teacher_selector(teachers):
    idx = np.random.randint(len(teachers))
    return teachers[idx]

def output_selector(outputs, answers, idx):
    return [outputs[i] for i in idx], [answers[i] for i in idx]

def train(epoch):
    print('\nEpoch: %d' % epoch)
    scheduler.step()
    student.train()
    train_loss = 0
    correct = 0
    total = 0
    discriminator_loss = 0
    
    for batch_idx, all_in in enumerate(zip(*tuple(trainloader))):
        # adjust_learning_rate(optimizer, epoch, args.epochs, args.lr)
        input, target = zip(*[all_in[k] for k in range(num_tasks)])
        slice_pt = 0
        slice_idx = [0]
        for l in [p.size(0) for p in input]:
            slice_pt += l // args.ngpu
            slice_idx.append(slice_pt)
        organized_input = []
        organized_target = []
        for ng in range(args.ngpu):
            for t in range(len(input)):
                bs = args.batchSize[t] // args.ngpu
                organized_input.append(input[t][ng * bs : (ng + 1) * bs, ...])
                organized_target.append(target[t][ng * bs : (ng + 1) * bs, ...])

        input = torch.cat(organized_input, dim=0)
        target = torch.cat(organized_target, dim=0)

        # measure data loading time
        inputs = Variable(input.cuda(),requires_grad=False)
        targets = Variable(target.cuda(),requires_grad=False)
    
    # for batch_idx, (inputs, targets) in enumerate(trainloader):
        total += targets.size(0)
        # inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
#----
        soft_labels_ = [[torch.unsqueeze(teachers[idx](inputs)[k], dim=2) for idx in range(len(teachers))] for k in range(num_tasks)]
        soft_labels_softmax = [[F.softmax(i, dim=1) for i in soft_labels_[k]] for k in range(num_tasks)]
        soft_labels_ = [torch.cat(soft_labels_[k], dim=2).mean(dim=2) for k in range(num_tasks)]
        soft_labels = [torch.cat(soft_labels_softmax[k], dim=2).mean(dim=2) for k in range(num_tasks)]
        
        s_output = student(inputs)
        
        #Calculate loss
        g_loss_output = [g_loss((s_output[k], soft_labels[k]), targets) for k in range(num_tasks)]
        # d_loss_value = [discriminator_loss([s_output[k]], [soft_labels_[k]]) for k in range(num_tasks)]
        d_loss = discriminators_criterion(s_output, soft_labels_)

        g_loss_k = 0
        for k in range(num_tasks):
            if isinstance(g_loss_output[k], tuple): 
                g_loss_value_, outputs_ = g_loss_output[k]
                g_loss_k = g_loss_k +  g_loss_value_
            else:
                g_loss_value_ = g_loss_output[k]
                g_loss_k = g_loss_k +  g_loss_value_

        total_loss = g_loss_k + d_loss
#----
        total_loss.backward()
        optimizer.step()

        train_loss += g_loss_k.item()
        discriminator_loss += d_loss.item()
        for k in range(num_tasks):
            _, predicted = s_output[k].max(1)
            sl1 = k*args.batch_size
            sl2 = (k+1)*args.batch_size
            correct += predicted[sl1:sl2].eq(targets[sl1:sl2]).sum().item()
        if batch_idx % 20 == 0:
            print(batch_idx, len(trainloader), 'Teacher: %s | Lr: %.4e | K_Loss: %.3f | D_Loss: %.3f | '
                % ('teacher', scheduler.get_lr()[0], train_loss / (batch_idx + 1), discriminator_loss / (batch_idx + 1)))
            # logging.info(
            #     'Epoch: [{epoch}][{batch}/{epoch_size}] | Lr: {lr:.4e} | K_Loss: {k_loss:.3f} | D_Loss: {d_loss:.3f} | '.format(
            #         epoch=epoch, batch=batch_idx + 1, epoch_size=len(trainloader), lr=scheduler.get_lr()[0], k_loss=train_loss / (batch_idx + 1), 
            #         d_loss=discriminator_loss / (batch_idx + 1)
            #     )
            # )


def test(epoch):
    global best_acc
    student.eval()
    test_loss = 0
    correct = 0
    total = 0
    discriminator_loss = 0
    with torch.no_grad():
        
        for batch_idx, all_in in enumerate(zip(*tuple(testloader))):
            input, target = zip(*[all_in[k] for k in range(num_tasks)])
            slice_pt = 0
            slice_idx = [0]
            for l in [p.size(0) for p in input]:
                slice_pt += l // 1 #args.ngpu
                slice_idx.append(slice_pt)
            organized_input = []
            organized_target = []
            for ng in range(1): #args.ngpu
                for t in range(len(input)):
                    bs = args.batchSize[t] // 1 #args.ngpu
                    organized_input.append(input[t][ng * bs : (ng + 1) * bs, ...])
                    organized_target.append(target[t][ng * bs : (ng + 1) * bs, ...])

            input = torch.cat(organized_input, dim=0)
            target = torch.cat(organized_target, dim=0)

            # measure data loading time
            inputs = Variable(input.cuda(),requires_grad=False)
            targets = Variable(target.cuda(),requires_grad=False)

        
        # for batch_idx, (inputs, targets) in enumerate(testloader):
            total += targets.size(0)
            # inputs, targets = inputs.to(device), targets.to(device)

            # Get output from student model
            outputs = student(inputs)
            # Get teacher model
            teacher = teacher_selector(teachers)
            # Get output from teacher model
            answers = teacher(inputs)
            # Select output from student and teacher
            outputs, answers = output_selector(outputs, answers, eval(args.out_layer))
            # Calculate loss between student and teacher
            loss = criterion(outputs, answers)
            # Calculate loss for discriminators
            d_loss = discriminators_criterion(outputs, answers)

            test_loss += loss.item()
            discriminator_loss += d_loss.item()
            # _, predicted = outputs[-1].max(1)
            # correct += predicted.eq(targets).sum().item()
            for k in range(num_tasks):
                _, predicted = outputs[k].max(1)
                sl1 = k*args.batch_size
                sl2 = (k+1)*args.batch_size
                correct += predicted[sl1:sl2].eq(targets[sl1:sl2]).sum().item()

            print(batch_idx, len(testloader), 'Lr: %.4e | G_Loss: %.3f | D_Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (scheduler.get_lr()[0], test_loss / (batch_idx + 1), discriminator_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        best_acc = max(100. * correct / total, best_acc)

    # Save checkpoint (the best accuracy).
    if epoch % 10 == 0 and best_acc == (100. * correct / total):
        print('Saving..')
        state = {
            'state_dict': student.state_dict(),
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        FILE_PATH = './checkpoint' + '/' + "_".join(args.teachers) + '_meal_v2-generator'
        if os.path.isdir(FILE_PATH):
            # print 'dir exists'generator
            pass
        else:
            # print 'dir not exists'
            os.mkdir(FILE_PATH)
        save_name = './checkpoint' + '/' + "_".join(args.teachers) + '_meal_v2-generator/ckpt.t7'
        torch.save(state, save_name)
    if epoch % 1 == 0:
        states = {
                'epoch': epoch,
                'state_dict': student.state_dict(),
            }
        paths = './checkpoint' + '/' + "_".join(args.teachers) + '_meal_v2-generator/'
        torch.save(states, '{}/{}.pth.tar'.format(paths, epoch))

for epoch in range(start_epoch, start_epoch+args.epochs*(len(teachers))):
    train(epoch)
    test(epoch)
