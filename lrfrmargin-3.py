#!/usr/bin/env python
# coding: utf-8

# In[57]:


import os
import random
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T

from tqdm import tqdm
import os
data_folder = "/home/arina/src/archive/casia-112x112/casia-112x112"
max_len_dict = {}
for folder in tqdm(os.listdir(data_folder)): 
    cur_path = os.path.join(data_folder, folder)
    if os.path.isdir(cur_path):
        max_len_dict[folder] = len(os.listdir(cur_path))

import pandas as pd 
keys_train = list(pd.Series(max_len_dict).sort_values().keys())
#min_len_key = pd.Series(max_len_dict).sort_values()
print(len(keys_train))

f = open("/home/arina/src/train.txt", "a")
for elem in keys_train:
    for file in os.listdir(os.path.join(data_folder, elem)):
        f.write(f"{os.path.join(elem, file)} {elem} \n")
f.close()

def transform(image):
    """ Transform a image by cv2.
        @author: Hang Du, Jun Wang
    """
    img_size = image.shape[0]
    # random crop
    if random.random() > 0.5:
        crop_size = 9
        x1_offset = np.random.randint(0, crop_size, size=1)[0]
        y1_offset = np.random.randint(0, crop_size, size=1)[0]
        x2_offset = np.random.randint(img_size-crop_size, img_size, size=1)[0]
        y2_offset = np.random.randint(img_size-crop_size, img_size, size=1)[0]
        image = image[x1_offset:x2_offset,y1_offset:y2_offset]
        image = cv2.resize(image,(img_size,img_size))
    # horizontal flipping
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
    # grayscale conversion
    if random.random() > 0.8:
        image= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # rotation
    if random.random() > 0.5:
        theta = (random.randint(-10,10)) * np.pi / 180
        M_rotate = np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0]], dtype=np.float32)
        image = cv2.warpAffine(image, M_rotate, (img_size, img_size))
    # normalizing
    if image.ndim == 2:
        image = (image - 127.5) * 0.0078125
        new_image = np.zeros([3,img_size,img_size], np.float32)
        new_image[0,:,:] = image
        image = torch.from_numpy(new_image.astype(np.float32))
    else:
        image = (image.transpose((2, 0, 1)) - 127.5) * 0.0078125
        image = torch.from_numpy(image.astype(np.float32))
    return image

class ImageDataset_Triplet(Dataset):
    def __init__(self, data_root, train_file):
        #self.length = length
        self.data_root = data_root
        train_file_buf = open(train_file)
        line = train_file_buf.readline().strip()
        self.all_possible_data = {}
        while line:
            image_path, label = line.split(' ')
            label = int(label)
            line = train_file_buf.readline().strip()
            if label not in self.all_possible_data:
                self.all_possible_data[label] = []
            self.all_possible_data[label].append(image_path)
        print(f"There are {len(self.all_possible_data.keys())} classes")

    def __len__(self):
        return len(self.all_possible_data.keys()) * 3

    def __getitem__(self, index):
        anchor_id = random.choice(list(self.all_possible_data.keys()))
        anchor_img = random.choice(self.all_possible_data[anchor_id])
        img1 = random.choice([elem for elem in self.all_possible_data[anchor_id] if elem != anchor_img])
        img_names = [anchor_img, img1] 
        labels = [anchor_id, anchor_id]
        for idx in range(4):
            label = random.choice([elem for elem in list(self.all_possible_data.keys()) if elem not in labels])
            labels.append(label)
            img_names.append(random.choice(self.all_possible_data[label]))
        images = []
        for img_name in img_names:
            img_path = os.path.join(self.data_root, img_name)
            image = cv2.imread(img_path)
            images.append(transform(image))
        assert len(images) == len(labels) == 6
        return images[0], images[1], images[2], images[3], images[4], images[5], labels[0], labels[1], labels[2], labels[3], labels[4], labels[5]

class ImageDataset(Dataset):
    def __init__(self, data_root, train_file, crop_eye=False):
        self.data_root = data_root
        self.train_list = []
#         [x[0] for x in os.walk("/kaggle/input/casia-webface/casia-webface")][1:5]
        train_file_buf = open(train_file)
        line = train_file_buf.readline().strip()
        un_ids = [] 
        labels = {elem: idx for idx, elem in enumerate(keys_train)}
        while line:
            image_path, image_label = line.split(' ')
            #if image_label in keys_train: 
                #un_ids.append(image_label)
            self.train_list.append((image_path, labels[image_label]))
            line = train_file_buf.readline().strip()
        #print(f"There are {len(self.train_list)} images and {set(un_ids)} ids")
        self.crop_eye = crop_eye

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, index):
        image_path0, image_label = self.train_list[index]
        image_path = os.path.join(self.data_root, image_path0)
        #print(image_path)
        image = cv2.imread(image_path)
        if self.crop_eye:
            image = image[:60, :]
        #image = cv2.resize(image, (128, 128)) #128 * 128
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        if index % 2 == 1:
            image = cv2.resize(cv2.resize(image, (16, 16)), (112, 112))
        #m = self.get_sharpness(image)
        m = 1
        image = (image.transpose((2, 0, 1)) - 127.5) * 0.0078125 
        image = torch.from_numpy(image.astype(np.float32))
        return image, image_label, m 

    def get_sharpness(self, img):
        laplacian = cv2.Laplacian(img,cv2.CV_64F)
        gnorm = np.sqrt(laplacian**2)
        return 1 / np.average(gnorm) 



import math
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
from collections import namedtuple
import torch


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0 ,bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0 ,bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride ,bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))

    def forward(self, x):
            shortcut = self.shortcut_layer(x)
            res = self.res_layer(x)
            return res + shortcut

class bottleneck_IR_SE(Module):
        def __init__(self, in_channel, depth, stride):
            super(bottleneck_IR_SE, self).__init__()
            if in_channel == depth:
                self.shortcut_layer = MaxPool2d(1, stride)
            else:
                self.shortcut_layer = Sequential(
                    Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                    BatchNorm2d(depth))
            self.res_layer = Sequential(
                BatchNorm2d(in_channel),
                Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
                PReLU(depth),
                Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
                BatchNorm2d(depth),
                SEModule(depth, 16)
            )

        def forward(self, x):
            shortcut = self.shortcut_layer(x)
            res = self.res_layer(x)
            return res + shortcut

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''
def get_block(in_channel, depth, num_units, stride=2):
        return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]

def get_blocks(num_layers):
        if num_layers == 50:
            return [
                get_block(in_channel=64, depth=64, num_units=3),
                get_block(in_channel=64, depth=128, num_units=4),
                get_block(in_channel=128, depth=256, num_units=14),
                get_block(in_channel=256, depth=512, num_units=3)]


class Resnet(Module):
    def __init__(self, num_layers=50, drop_ratio=0.4, mode='ir', feat_dim=512, out_h=7, out_w=7):
        super(Resnet, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_layer = Sequential(BatchNorm2d(512),
                                       Dropout(drop_ratio),
                                       Flatten(),
                                       Linear(512 * out_h * out_w, feat_dim),  # for eye
                                       BatchNorm1d(feat_dim))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        #print(x.shape)
        x = self.body(x)
        #print(x.shape)
        x = self.output_layer(x)
        #print("out", x.shape)
        return x

import torch.nn as nn
class ArcFace(torch.nn.Module):
    """ reference: <Additive Angular Margin Loss for Deep Face Recognition>
    """
    def __init__(self, feat_dim=512, num_class=10575, scale=64., margin_arc=0.35):
        super(ArcFace, self).__init__()
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.s = scale
        self.m = margin_arc
        self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        nn.init.xavier_normal_(self.w)

    def forward(self, x, y, m0):
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)
        cos_theta = F.normalize(x, dim=1).mm(self.w)
        with torch.no_grad():
            theta_m = torch.acos(cos_theta.clamp(-1+1e-5, 1-1e-5))
            y_ = y.view(-1, 1)
            m = m0.view(-1, 1).float()
            #m = torch.ones(y_.shape[0]).cuda().view(-1, 1) * self.m
            theta_m.scatter_(1, y_, self.m, reduce='add')
            theta_m.clamp_(1e-5, 3.14159)
            d_theta = torch.cos(theta_m) - cos_theta

        logits = self.s * (cos_theta + d_theta)
        return logits 

class ArcFace0(Module):
    """Implementation for "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    """
    def __init__(self, feat_dim=512, num_class=10575, margin_arc=0.35, margin_am=0.0, scale=32):
        super(ArcFace, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_class))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin_arc = margin_arc
        self.margin_am = margin_am
        self.scale = scale
        self.cos_margin = math.cos(margin_arc)
        self.sin_margin = math.sin(margin_arc)
        self.min_cos_theta = math.cos(math.pi - margin_arc)

    def forward(self, feats, labels):
        kernel_norm = F.normalize(self.weight, dim=0)
        feats = F.normalize(feats)
        cos_theta = torch.mm(feats, kernel_norm)
        cos_theta = cos_theta.clamp(-0.9999999, 0.9999999)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * self.cos_margin - sin_theta * self.sin_margin
        # 0 <= theta + m <= pi, ==> -m <= theta <= pi-m
        # because 0<=theta<=pi, so, we just have to keep theta <= pi-m, that is cos_theta >= cos(pi-m)
        cos_theta_m = torch.where(cos_theta > self.min_cos_theta, cos_theta_m, cos_theta-self.margin_am)
        index = torch.zeros_like(cos_theta)
        index.scatter_(1, labels.data.view(-1, 1), 1)
        index = index.byte().bool()
        output = cos_theta * 1.0
        output[index] = cos_theta_m[index]
        output *= self.scale
        return output
    
class CenterLoss(torch.nn.Module):
    """Center loss
    https://github.com/KaiyangZhou/pytorch-center-loss

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, feat_dim, num_classes):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = torch.nn.Parameter(torch.Tensor(num_classes, feat_dim)).to(torch.device("cuda:0"))
        self.centers.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, feats, labels):
        batch_size = feats.size(0)
        distmat = torch.pow(feats, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) +                   torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, feats, self.centers.t())

        classes = torch.arange(self.num_classes).long().to(torch.device("cuda:0"))
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat * mask.float() 
        
        return dist.sum() + F.log_softmax(feats)
    
class SphereFace(Module):
    def __init__(self, feat_dim, num_class, margin=0.35):
        super(SphereFace, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_class))
        self.num_class = num_class
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin = margin

    def forward(self, feats, labels):
        kernel_norm = F.normalize(self.weight, dim=0) 
        feats = F.normalize(feats)  
        cos_theta = torch.mm(feats, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        
        cos_theta_m = torch.cos(torch.acos(cos_theta) * self.margin)
        #cos_theta_m = cos_theta * self.margin
        onehot = F.one_hot(labels, self.num_class)
        logits = torch.linalg.norm(feats) * torch.where(onehot == 1, cos_theta_m, cos_theta) 
        return logits

class CosFace(Module):
    def __init__(self, feat_dim, num_class, margin=0.35, scale=30):
        super(CosFace, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_class))
        self.num_class = num_class
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin = margin
        self.scale = scale

    def forward(self, feats, labels):
        kernel_norm = F.normalize(self.weight, dim=0) 
        feats = F.normalize(feats)  
        cos_theta = torch.mm(feats, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        
        cos_theta_m = cos_theta - self.margin
        onehot = F.one_hot(labels, self.num_class)
        logits = self.scale * torch.where(onehot == 1, cos_theta_m, cos_theta) 
        return logits
"""
class ArcFace(Module):
    def __init__(self, feat_dim, num_class, margin=0.35):
        super(ArcFace, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_class))
        self.num_class = num_class
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin = margin

    def forward(self, feats, labels):
        kernel_norm = F.normalize(self.weight, dim=0) 
        feats = F.normalize(feats)  
        cos_theta = torch.mm(feats, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        
        cos_theta_m = torch.cos(torch.acos(cos_theta) + self.margin)
        onehot = F.one_hot(labels, self.num_class)
        logits = self.scale * torch.where(onehot == 1, cos_theta_m, cos_theta) 
        return logits"""


from scipy.special import binom

"""https://github.com/amirhfarzaneh/lsoftmax-pytorch"""
class LSoftmaxLinear(Module):
    def __init__(self, feat_dim, num_class, margin=2):
        super(LSoftmaxLinear, self).__init__()
        self.feat_dim = feat_dim
        self.num_class = num_class 
        self.margin = margin 
        self.beta = 100
        self.beta_min = 0
        self.scale = 0.99
        

        # Initialize L-Softmax parameters
        self.weights =  Parameter(torch.Tensor(feat_dim, num_class))
        self.divisor = math.pi / self.margin  # pi/m
        self.C_m_2n = torch.Tensor(binom(margin, range(0, margin + 1, 2))).long().cuda()  # C_m{2n}
        self.cos_powers = torch.Tensor(range(self.margin, -1, -2)).long().cuda()  # m - 2n
        self.sin2_powers = torch.Tensor(range(len(self.cos_powers))).long().cuda()  # n
        self.signs = torch.ones(margin // 2 + 1)  # 1, -1, 1, -1, ...
        self.signs[1::2] = -1

    def calculate_cos_m_theta(self, cos_theta):
        sin2_theta = 1 - cos_theta**2
        cos_terms = cos_theta.unsqueeze(1) ** self.cos_powers.unsqueeze(0)  # cos^{m - 2n}
        sin2_terms = (sin2_theta.unsqueeze(1)  # sin2^{n}
                      ** self.sin2_powers.unsqueeze(0))

        cos_m_theta = (self.signs.unsqueeze(0) *  # -1^{n} * C_m{2n} * cos^{m - 2n} * sin2^{n}
                       self.C_m_2n.unsqueeze(0) *
                       cos_terms *
                       sin2_terms).sum(1)  # summation of all terms

        return cos_m_theta

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight.data.t())

    def find_k(self, cos):
        # to account for acos numerical errors
        eps = 1e-7
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        acos = cos.acos()
        k = (acos / self.divisor).floor().detach()
        return k

    def forward(self, feats, labels):
        beta = max(self.beta, self.beta_min)
        logits = feats.mm(self.weights)
        logit_target = logits[:, labels]
        w_target_norm = self.weights[:, labels].norm(p=2, dim=0)
        x_norm = feats.norm(p=2, dim=1)
        cos_theta_target = logit_target / (w_target_norm * x_norm + 1e-10)
        cos_m_theta_target = self.calculate_cos_m_theta(cos_theta_target)
        k = self.find_k(cos_theta_target)
        logit_target_updated = (w_target_norm *
                                x_norm *
                                (((-1) ** k * cos_m_theta_target) - 2 * k))
        logit_target_updated_beta = (logit_target_updated + beta * logits[:, labels]) / (1 + beta)

        logits[:, labels] = logit_target_updated_beta
        self.beta *= self.scale
        return logits
    
import torch
import torch.nn.functional as F
class LGMLoss(torch.nn.Module):
    """
    Refer to paper:
    Weitao Wan, Yuanyi Zhong,Tianpeng Li, Jiansheng Chen
    Rethinking Feature Distribution for Loss Functions in Image Classification. CVPR 2018
    re-implement by yirong mao
    2018 07/02
    """
    def __init__(self, feat_dim, num_class, alpha=3):
        super(LGMLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_class
        self.alpha = alpha

        self.centers = torch.nn.Parameter(torch.Tensor(num_class, feat_dim))
        self.log_covs = torch.nn.Parameter(torch.Tensor(num_class, feat_dim))
        self.centers.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.log_covs.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, feats, labels):
        batch_size = feats.shape[0]
        log_covs = torch.unsqueeze(self.log_covs, dim=0)


        covs = torch.exp(log_covs) # 1*c*d
        tcovs = covs.repeat(batch_size, 1, 1) # n*c*d
        diff = torch.unsqueeze(feats, dim=1) - torch.unsqueeze(self.centers, dim=0)
        wdiff = torch.div(diff, tcovs)
        diff = torch.mul(diff, wdiff)
        dist = torch.sum(diff, dim=-1) #eq.(18)

        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = torch.autograd.Variable(y_onehot)
        y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.alpha)
        y_onehot = y_onehot + 1.0
        onehot = F.one_hot(labels, self.num_classes)
        margin_dist = torch.mul(dist, y_onehot)

        slog_covs = torch.sum(log_covs, dim=-1) #1*c
        tslog_covs = slog_covs.repeat(batch_size, 1)
        margin_logits = -0.5*(tslog_covs + margin_dist) #eq.(17)
        logits = -0.5 * (tslog_covs + dist)

        cdiff = feats - torch.index_select(self.centers, dim=0, index=labels.long())
        cdist = cdiff.pow(2).sum(1).sum(0) / 2.0

        slog_covs = torch.squeeze(slog_covs)
        reg = 0.5*torch.sum(torch.index_select(slog_covs, dim=0, index=labels.long()))
        likelihood = (1.0/batch_size) * (cdist + reg)

        return logits #, margin_logits, likelihood

class COCOLoss(torch.nn.Module):
    """
        Refer to paper:
        Yu Liu, Hongyang Li, Xiaogang Wang
        Rethinking Feature Discrimination and Polymerization for Large scale recognition. NIPS workshop 2017
        re-implement by yirong mao
        2018 07/02
        """

    def __init__(self, feat_dim, num_class, alpha=6.25):
        super(COCOLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_class
        self.alpha = alpha
        self.centers = torch.nn.Parameter(torch.Tensor(num_class, feat_dim))
        self.centers.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, feats, labels):
        norms = torch.norm(feats, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feats, norms)
        snfeat = self.alpha * nfeat

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)

        logits = torch.matmul(snfeat, torch.transpose(ncenters, 0, 1))
        return logits

class FullModel(torch.nn.Module):
    def __init__(self, path=None, head_type="ArcFace", feat_dim=512, num_class=10575):
        super(FullModel, self).__init__()
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.backbone = self.get_backbone(path, True)
        if head_type == "ArcFace":
            self.head = ArcFace(feat_dim, num_class)
        elif head_type == "CosFace": 
            self.head = CosFace(feat_dim, num_class)
        elif head_type == "CenterLoss": 
            self.head = CenterLoss(feat_dim, num_class)
        elif head_type == "SphereFace": 
            self.head = SphereFace(feat_dim, num_class)
        elif head_type == "LSoftmaxLinear": 
            self.head = LSoftmaxLinear(feat_dim, num_class)
        elif head_type == "COCOLoss":
            self.head = COCOLoss(feat_dim, num_class)
        elif head_type =="LGMLoss":
            self.head = LGMLoss(feat_dim, num_class)
        else: 
            raise ValueError(f"Wrong Head Type: {head_type}")

    def get_backbone(self, path=None, pretrained=False):
        backbone = Resnet(feat_dim=self.feat_dim)
        if pretrained:
            assert path is not None
            # model_path = "./FaceX-Zoo/training_mode/conventional_training/out_dir/Epoch_4.pt"
            model_dict = backbone.state_dict()
            pretrained_dict = torch.load(path)['state_dict']
            new_pretrained_dict = {}
            #print(pretrained_dict)
            for k in model_dict:
                new_pretrained_dict[k] = pretrained_dict['module.backbone.' + k]  # tradition training
            model_dict.update(new_pretrained_dict)
            backbone.load_state_dict(model_dict)
        return backbone

    def forward(self, data, label, m):
        feat = self.backbone.forward(data)
        pred = self.head.forward(feat, label, m)
        return pred, feat


# In[58]:


import os
import sys

import torch
from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import torch.nn.functional as F
import neptune.new as neptune
import numpy as np
import math
torch.autograd.set_detect_anomaly(True)

sys.path.append('../../')
os.environ['CUDA_LAUNCH_BLOCKING'] = "True"

#from dataset import ImageDataset_Triplet, ImageDataset
#from model import FullModel


class Training(object):
    def __init__(self):
        super(Training, self).__init__()
        self.batch_size = 4
        self.epoch = 16
        self.lr = 0.1
        self.mode = "scratch"  # "scratch" if we train from scratch / "resume" if we continue arcface training
        self.data_root = '/home/arina/src/archive/casia-112x112/casia-112x112'
        self.train_file = '/home/arina/src/train.txt'
        self.out_dir = '.'
        self.pretrain_model = "./LowArcFaceEpoch_1.pt"
        self.save_freq = 3000
        self.margin = 0.35
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.milestones = [5, 10, 15]
        self.gamma = 0.5
        self.ori_epoch = 0
        self.lambda_loss = 1e-3
        self.data_loader = DataLoader(ImageDataset(self.data_root, self.train_file),
                                 self.batch_size, shuffle=True)
        self.device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')
        self.embeds = {"id": [], "info": []}
        self.criterion_cross_entropy = torch.nn.CrossEntropyLoss().to(self.device)
        self.head_type = "ArcFace"
        self.model = FullModel(head_type=self.head_type, path=self.pretrain_model) if self.mode != "finetune" else FullModel().get_backbone(pretrained=True, path=self.pretrain_model)
        if self.mode == "resume":
            self.ori_epoch = torch.load(self.pretrain_model)['epoch'] + 1
            state_dict = torch.load(self.pretrain_model)['state_dict']
            self.model.load_state_dict(state_dict)
        self.model = torch.nn.DataParallel(self.model.to(self.device)) 
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.SGD(parameters, lr=self.lr,
                              momentum=self.momentum, weight_decay=self.weight_decay)
        self.lr_schedule = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.milestones, gamma=self.gamma)
        #self.run = neptune.init_run(
        #    project="arinalozhkina/face-recognition",
        #    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3ZDU2N2RmNy0zMDljLTRhYTctYjI3Ni0xNjRlZWE0YjU4NDUifQ==",
        #)
        #self.run["parameters"] = {'Head type': self.head_type, 'batch_size': self.batch_size, 'lr': self.lr, 'margin': self.margin}


    def train_epoch(self, cur_epoch):
        for batch_idx, (img, label, m) in enumerate(self.data_loader):
                img, label, m = img.to(self.device), label.to(self.device), m.to(self.device)
                '''if cur_epoch >= self.epoch - 2:
                if cur_epoch == self.epoch -1:
                    img = T.Resize(size=(112, 112))(T.Resize(size=(16, 16))(img))
                embeddings = self.model.backbone.forward(img)
                for idx in range(len(label)):
                    self.embeds["id"].append(int(label[idx].cpu())) 
                    self.embeds["info"].append(embeddings[idx].detach().cpu().numpy())
                else:'''
                em, embeddings = self.model.forward(img, label, m)
                loss = self.criterion_cross_entropy(em, label) 
                #self.run["cross_entropy_loss"].log(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if (batch_idx + 1) % self.save_freq == 0 or (batch_idx + 1) == len(self.data_loader):
                        print(f"cur_epoch:{cur_epoch} loss =", loss.item())
                        saved_name = f'Low{self.head_type}Epoch_{cur_epoch}.pt'
                        state = {
                            'state_dict': self.model.state_dict(),
                            'epoch': cur_epoch,
                            'batch_id': batch_idx
                        }
                        torch.save(state, os.path.join(self.out_dir, saved_name))
    
        self.lr_schedule.step()

    def train(self):
        self.model.train()

        for epoch in tqdm(range(self.ori_epoch, self.epoch)):
            self.train_epoch(epoch)
        
        #pd.DataFrame(self.embeds).to_csv("./res_adap.csv")
        #self.run["embeddings"].upload("./res_adap.csv")
        #self.run.stop()


if __name__ == '__main__':
    Training().train()




