
# -*- coding: utf-8 -*-
import math
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
from collections import namedtuple
import torch
from scipy.io import loadmat 
import random
import pandas as pd 
from sklearn import metrics 
import scipy 
device = torch.device("cpu")
#device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

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
        x = self.body(x)
        x = self.output_layer(x)
        return x



class ArcFace(Module):
    """Implementation for "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    """
    def __init__(self, feat_dim=512, num_class=72778, margin_arc=0.35, margin_am=0.0, scale=32):
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
        cos_theta = cos_theta.clamp(-1, 1)
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

class FullModel(torch.nn.Module):
    def __init__(self, path=None):
        super(FullModel, self).__init__()
        self.backbone = self.get_backbone(path)
        self.head = ArcFace()

    def get_backbone(self, path=None, pretrained=False):
        backbone = Resnet()
        if pretrained:
            assert path is not None
            model_dict = backbone.state_dict()
            pretrained_dict = torch.load(path, map_location='cpu')['state_dict']
            new_pretrained_dict = {}
            #print(pretrained_dict)
            for k in model_dict:
                 new_pretrained_dict[k] = pretrained_dict['backbone.' + k] 
            model_dict.update(new_pretrained_dict)
            backbone.load_state_dict(model_dict)
        return backbone

    def forward(self, data, label):
        feat = self.backbone.forward(data)
        pred = self.head.forward(feat, label)
        return pred

def get_list_of_pairs(line_strs):
        if len(line_strs) == 3:
                person_name = line_strs[0]
                image_index1 = line_strs[1]
                image_index2 = line_strs[2]
                image_name1 = person_name + '/' + person_name + '_' + image_index1.zfill(4) + '.jpg'
                image_name2 = person_name + '/' + person_name + '_' + image_index2.zfill(4) + '.jpg'
                label = 1
        elif len(line_strs) == 4:
                person_name1 = line_strs[0]
                image_index1 = line_strs[1]
                person_name2 = line_strs[2]
                image_index2 = line_strs[3]
                image_name1 = person_name1 + '/' + person_name1 + '_' + image_index1.zfill(4) + '.jpg'
                image_name2 = person_name2 + '/' + person_name2 + '_' + image_index2.zfill(4) + '.jpg'
                label = 0
        else:
                raise Exception('Line error: %s.' % line_strs)
        return (image_name1, image_name2, label)

def pairs_survface():
    mode_pairs = []
    for mode in ["positive", "negative"]:
        pairs = loadmat(f"./QMUL-SurvFace/Face_Verification_Test_Set/{mode}_pairs_names.mat", squeeze_me=True)
        pairs = pairs[f'{mode}_pairs_names']
        nb_pairs = pairs.shape[0]
        mode_pairs.append(np.stack([pairs[:, 0].flatten(), pairs[:, 1].flatten(), np.ones(nb_pairs) * int(mode == "positive")]).T)
    return np.vstack(mode_pairs)
    

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

def square_crop(im, S=112):
    if im.shape[0] > im.shape[1]:
        height = S
        width = int(float(im.shape[1]) / im.shape[0] * S)
        scale = float(S) / im.shape[0]
    else:
        width = S
        height = int(float(im.shape[0]) / im.shape[1] * S)
        scale = float(S) / im.shape[1]
    resized_im = cv2.resize(im, (width, height))
    det_im = np.zeros((S, S, 3), dtype=np.uint8)
    det_im[:resized_im.shape[0], :resized_im.shape[1], :] = resized_im
    return det_im

class CommonTestDataset1(Dataset):
    def __init__(self, image_root, image_list, target_size=(112, 112)):
        self.image_root = image_root
        self.image_list = image_list
        self.mean = 127.5
        self.std = 128.0
        self.target_size = target_size

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        short_image_path = self.image_list[index]
        image_path = os.path.join(self.image_root, short_image_path)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        image = (image[:, :, ::-1] / 255.  - self.mean) / self.std
        image = image.transpose(2, 0, 1)
        shape = sorted(image.shape)[1:]
        border = (shape[1] - shape[0]) // 2
        if border > 1:
            if image.shape[0] == 3:
                image = np.moveaxis(image, 0, -1)
            if image.shape[1] < image.shape[2]:
                image = np.moveaxis(image, 1, -1)
            image = cv2.copyMakeBorder(image, 0, 0, border, border, cv2.BORDER_CONSTANT)
            if image.shape[-1] == 3:
                image = np.moveaxis(image, 2, 0)
        image = torch.from_numpy(image.astype(np.float32))
        image = T.Resize(size=(112, 112))(T.Resize(size=self.target_size)(image))
        return image, short_image_path


class CommonTestDataset(Dataset):
    """ Data processor for model evaluation.

    Attributes:
        image_root(str): root directory of test set.
        image_list(list): the image list file.
    """
    def __init__(self, image_root, image_list, target_size=(112, 112)):
        self.image_root = image_root
        self.image_list = image_list
        self.mean = 127.5
        self.std = 128.0
        self.target_size = target_size

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        short_image_path = self.image_list[index]
        image_path = os.path.join(self.image_root, short_image_path)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        image = square_crop(image)
        if self.target_size != (112, 112):
            image = cv2.resize(cv2.resize(image, self.target_size), (112, 112))
        image = (image.transpose((2, 0, 1)) - self.mean) / self.std
        image = torch.from_numpy(image.astype(np.float32))
        return image, short_image_path

import torch.nn.functional as F
from tqdm import tqdm
def extract_features(model, data_loader):
        """Extract and return features.

        Args:
            model(object): initialized model.
            data_loader(object): load data to be extracted.

        Returns:
            image_name2feature(dict): key is the name of image, value is feature of image.
        """
        model.eval()
        image_name2feature = {}
        with torch.no_grad():
            for batch_idx, (images, filenames) in tqdm(enumerate(data_loader)):
                images = images.to(device)
                features = model(images)
                features = F.normalize(features)
                features = features.cpu().numpy()
                for filename, feature in zip(filenames, features):
                    image_name2feature[filename] = feature
        return image_name2feature


def getThreshold(score_list, label_list, num_thresholds=1000):
        """Get the best threshold by train_score_list and train_label_list.
        Args:
            score_list(ndarray): the score list of all pairs.
            label_list(ndarray): the label list of all pairs.
            num_thresholds(int): the number of threshold that used to compute roc.
        Returns:
            best_thres(float): the best threshold that computed by train set.
        """
        pos_score_list = score_list[label_list == 1]
        neg_score_list = score_list[label_list == 0]
        pos_pair_nums = pos_score_list.size
        neg_pair_nums = neg_score_list.size
        score_max = np.max(score_list)
        score_min = np.min(score_list)
        score_span = score_max - score_min
        step = score_span / num_thresholds
        threshold_list = score_min +  step * np.array(range(1, num_thresholds + 1))
        fpr_list = []
        tpr_list = []
        for threshold in threshold_list:
            fpr = np.sum(neg_score_list > threshold) / neg_pair_nums
            tpr = np.sum(pos_score_list > threshold) /pos_pair_nums
            fpr_list.append(fpr)
            tpr_list.append(tpr)
        fpr = np.array(fpr_list)
        tpr = np.array(tpr_list)
        best_index = np.argmax(tpr-fpr)
        best_thres = threshold_list[best_index]
        return  best_thres


def test_one_model(test_pair_list, image_name2feature, is_normalize = True):
        """Get the accuracy of a model.

        Args:
            test_pair_list(list): the pair list.
            image_name2feature(dict): the map of image name and it's feature.
            is_normalize(bool): wether the feature is normalized.

        Returns:
            mean: estimated mean accuracy.
            std: standard error of the mean.
        """
        len_subset = len(test_pair_list) // 10
        subsets_score_list = np.zeros((10, len_subset), dtype = np.float32)
        subsets_label_list = np.zeros((10, len_subset), dtype = np.int8)
        for index, cur_pair in tqdm(enumerate(test_pair_list)):
            cur_subset = index // len_subset
            cur_id = index % len_subset
            image_name1 = cur_pair[0]
            image_name2 = cur_pair[1]
            label = cur_pair[2]
            subsets_label_list[cur_subset][cur_id] = label
            feat1 = image_name2feature[image_name1]
            feat2 = image_name2feature[image_name2]
            if not is_normalize:
                feat1 = feat1 / np.linalg.norm(feat1)
                feat2 = feat2 / np.linalg.norm(feat2)
            cur_score = np.dot(feat1, feat2)
            subsets_score_list[cur_subset][cur_id] = cur_score

        subset_train = np.array([True] * 10)
        accu_list, fmr_list, fnmr_list = [], [], []
        for subset_idx in tqdm(range(10)):
            test_score_list = subsets_score_list[subset_idx]
            test_label_list = subsets_label_list[subset_idx]
            subset_train[subset_idx] = False
            train_score_list = subsets_score_list[subset_train].flatten()
            train_label_list = subsets_label_list[subset_train].flatten()
            subset_train[subset_idx] = True
            best_thres = getThreshold(train_score_list, train_label_list)
            positive_score_list = test_score_list[test_label_list == 1]
            negtive_score_list = test_score_list[test_label_list == 0]
            true_pos_pairs = np.sum(positive_score_list > best_thres)
            true_neg_pairs = np.sum(negtive_score_list < best_thres)
            fmr_list.append(np.sum(negtive_score_list >= best_thres) / len_subset)
            fnmr_list.append(np.sum(positive_score_list < best_thres) / len_subset)
            accu_list.append((true_pos_pairs + true_neg_pairs) / len_subset)
        mean = np.mean(accu_list)
        std = np.std(accu_list, ddof=1) / np.sqrt(10) #ddof=1, division 9.
        print("FMR", np.mean(fmr_list), "FNMR",  np.mean(fnmr_list))
        return mean, std

import numpy as np

class Tester(object):
    def __init__(self, ep=0):
        #path = "/home/arina/src/AdapMarginArcFaceEpoch_032.pt" # Approach 2
        #path = "/home/arina/src/Epoch_0_batch_504.pt"
        #path = f"/home/arina/src/WEIGHTS.pt" # Approach 1 
        ep = 3
        #path = "/home/arina/src/DataArcFaceEpoch_6.pt"       
        path = "/home/arina/src/Epoch_17coco.pt"
        #print(path)
        #path = f"/home/arina/src/weights_adap/ArcFaceEpoch_{ep}.pt"
        #path = "/home/arina/pretrained_Res50_Arcface_webface/Epoch_17.pt"
        self.model = FullModel().get_backbone(path=path, pretrained=True).to(device)
        self.model.eval()
        #self.test_dataset = ["survface_112"]
        self.test_dataset = ["lfw_7"] #,"lfw_56", 'lfw_14', 'survface_112', 'lfw_112', 'lfw7'] #, "lfw_56", "lfw_28", "lfw_14", "lfw_7",  "survface_112"] 
    def get_data(self, mode):
        batch_size = 16
        mode, target_size = mode.split("_")
        if mode == "lfw":
            with open('/home/arina/src/pairs.txt') as f:
                pairs_lines = list(map(lambda k: k.rstrip().split("\t"), f.readlines()[1:]))

            cropped_face_folder = "/home/arina/src/lfw_crop"
            all_pairs = list(map(get_list_of_pairs, pairs_lines))
        else:
            cropped_face_folder = "./QMUL-SurvFace/Face_Verification_Test_Set/verification_images"
            all_pairs = pairs_survface()

         #all_pairs = [('./sc_cropped/090_cam3_2.jpg', './frontal_cropped/130_frontal.jpg', 1), ('./sc_cropped/120_cam1_2.jpg', './frontal_cropped/130_frontal.jpg', 1), ('./sc_cropped/116_cam2_1.jpg', './frontal_cropped/130_frontal.jpg', 1), ('./sc_cropped/095_cam5_1.jpg', './frontal_cropped/130_frontal.jpg', 1), ('./sc_cropped/113_cam2_3.jpg', './frontal_cropped/130_frontal.jpg', 1), ('./sc_cropped/095_cam5_2.jpg', './frontal_cropped/110_frontal.jpg', 0), ('./sc_croppe 
        all_elems = np.unique(np.array(all_pairs)[:, :2].flatten())
        if mode == "lfw":
           size = int(target_size)
        else:
           size = 112 
        data_loader = torch.utils.data.DataLoader(CommonTestDataset(cropped_face_folder, all_elems, (size, size)),
                             batch_size=batch_size, shuffle=False)
        return all_pairs, data_loader

    def calculate_auc(self, mode="scface_all"):
        all_pairs, data_loader = self.get_data(mode)
        image_name2feature = extract_features(self.model, data_loader)
        labels, scores = [], []
        for index, cur_pair in tqdm(enumerate(all_pairs)):
            image_name1 = cur_pair[0]
            image_name2 = cur_pair[1]
            label = cur_pair[2]
            feat1 = image_name2feature[image_name1]
            feat2 =  image_name2feature[image_name2]
            cur_score  = np.sqrt(np.sum((feat2 - feat1) ** 2))
            #feat1 = feat1 / np.linalg.norm(feat1)
            #feat2 = feat2 / np.linalg.norm(feat2)
            #cur_score = np.dot(feat1, feat2)
            labels.append(label)
            scores.append(cur_score)
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
        print(metrics.auc(fpr, tpr))
        #leng = len(scores) // 2
        #scipy.io.savemat(f'./dist0.mat', {'matrix_name': scores[:leng]})
        #scipy.io.savemat(f'./dist1.mat', {'matrix_name': scores[leng:]})

    def calculate_conf_matrix(self,  mode="scface_all"):
        all_pairs, data_loader = self.get_data(mode)
        image_name2feature = extract_features(self.model, data_loader)
        embeds = np.array(list(image_name2feature.values())[:10])
        #pd.DataFrame(metrics.pairwise.cosine_distances(embeds, embeds.T)).to_csv("./arc_pairwise.csv")
        pd.DataFrame(np.dot(embeds, embeds.T)).to_csv("./app2_pairwise.csv")

    def test_scface(self, dist=1):
        image_path = "/home/arina/src/scface/sc_cropped" 
        models_path =  "/home/arina/src/scface/models_croppped"
        #model_path = "/home/arina/src/weights_adap/ArcFaceEpoch_9.pt"
        #model_path =  "/home/arina/src/WEIGHTS.pt"
        #model_path = "/home/arina/pretrained_Res50_Arcface_webface/Epoch_17.pt" 
        protocol_probes = [elem for elem in os.listdir(image_path) if int(elem[-5]) == dist and int(elem[-7]) <= 5]
        data_loader = torch.utils.data.DataLoader(CommonTestDataset(models_path, os.listdir(models_path)),
                             batch_size=16, shuffle=False)
        #model = FullModel().get_backbone(path=model_path, pretrained=True).to(device)
        gallery = pd.DataFrame(extract_features(self.model, data_loader))
        #pd.DataFrame(gallery).to_csv("gallery.csv")
        data_loader_probe = torch.utils.data.DataLoader(CommonTestDataset(image_path, protocol_probes),
                             batch_size=16, shuffle=False)
        probes = pd.DataFrame(extract_features(self.model, data_loader_probe)) 
        #pd.DataFrame(probes).to_csv("probes.csv") 
        normalized = probes.to_numpy()
        normalized_gallery = gallery.to_numpy()
        predictions = np.argmax(np.dot(normalized.T, normalized_gallery), 1)
        probes, gallery = probes.T, gallery.T
        probes["label"] = list(map(lambda k: k[:3], probes.index))
        gallery["label"] = list(map(lambda k: k[:3], gallery.index))
        #print(np.where(gallery.iloc[predictions].label.values == probes.label.values))
        #print(model_path, dist, sum(gallery.iloc[predictions].label.values == probes.label.values) / len(probes.label.values))
        return sum(gallery.iloc[predictions].label.values == probes.label.values) / len(probes.label.values)

    def test(self):
        res = []
        '''for mode in self.test_dataset:
            all_pairs, data_loader = self.get_data(mode)
            image_name2feature = extract_features(self.model, data_loader)
            mean, std = test_one_model(all_pairs, image_name2feature)
            print(mode, mean, std)
            res.append(mean)
        for i in range(1, 4, 1):
            res.append(self.test_scface(i))
            print(res[-1])'''
        print(self.test_scface(1))
        return res


def test_lfw(): 
    with open('/home/arina/src/pairs.txt') as f:
        pairs_lines = list(map(lambda k: k.rstrip().split("\t"), f.readlines()[1:]))

    cropped_face_folder = "/home/arina/src/lfw_crop"
    model_path = "/home/arina/src"
    batch_size = 512


    all_pairs = list(map(get_list_of_pairs, pairs_lines))
    all_elems = np.unique(np.array(all_pairs)[:, :2].flatten())
    data_loader = torch.utils.data.DataLoader(CommonTestDataset(cropped_face_folder, all_elems, (7, 7)),
                             batch_size=batch_size, shuffle=False)

    model = FullModel().get_backbone(path=os.path.join(model_path, "ArcFaceEpoch_16.pt"), pretrained=True).to(device)
    image_name2feature = extract_features(model, data_loader)
    mean, std = test_one_model(all_pairs, image_name2feature)
    print(mean, std)

# QMUL
def test_survface():
    cropped_face_folder = "./QMUL-SurvFace/Face_Verification_Test_Set/verification_images"
    model_path = "/home/arina/src/Epoch_17-2.pt"
    #model_path = "/home/arina/pretrained_Res50_Arcface_ms1m/Epoch_17.pt" 
    batch_size = 512
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    all_pairs = pairs_survface()
    all_elems = np.unique(np.array(all_pairs)[:, :2].flatten())
    data_loader = torch.utils.data.DataLoader(CommonTestDataset(cropped_face_folder, all_elems),
                             batch_size=batch_size, shuffle=False)

    model = FullModel().get_backbone(path=model_path, pretrained=True).to(device)
    image_name2feature = extract_features(model, data_loader)
    mean, std = test_one_model(all_pairs, image_name2feature)
    print(mean, std)

# SCFACE
def pairs_scface(dist=[1, 2, 3], night=False):
    pairs = []
    not_proc = []
    night_cam_check = [6, 7, 8] if night else [1, 2, 3, 4, 5]
    image_path = "./sc_cropped" 
    models_path =  "./frontal_cropped" 
    with open("./lrfr/for_scores.lst") as f:
        all_idx = f.readlines()
    all_idx = list(map(lambda x: x[:-1], all_idx))
    all_idx = {elem.split()[0]: elem.split()[1] for elem in all_idx}
    with open("./lrfr/for_models.lst") as f:
        all_models = {elem[1]: elem[0] for elem in list(map(lambda x: x.split(), f.readlines()))}
    #print(list(set(all_idx.values()), (list(map(lambda k: k[:-4], os.listdir(image_path)))))
    keys_ = list(set(all_idx.keys()) & set(list(map(lambda k: k[:-4], os.listdir(image_path)))))
    for mode in ["positive", "negative"]: 
        for elem in tqdm(keys_):
            name = elem.split(".")[0]
            cam = list(filter(lambda x: x.startswith("cam"), name.split("_")))[0]

            #if name in all_idx.keys(): # and int(name[-1]) in dist and int(cam[-1]) in night_cam_check:
            label = all_idx[name]
            name2 = all_models[label] if mode == "positive" else all_models[random.choice([elem for elem in all_models.keys() if elem != label])]
            pairs.append((os.path.join(image_path, name + ".jpg"), os.path.join(models_path, name2 + ".jpg"), int(mode == "positive")))
    return pairs[:len(pairs) - len(pairs) % 10]

def test_scface1(dist=1, model_path="/home/arina/pretrained_Res50_Arcface_webface/Epoch_17.pt"):
    gallery_path, probes_path =  "./frontal_cropped", "./sc_cropped" 
    data_loader_gallery = torch.utils.data.DataLoader(CommonTestDataset(gallery_path, os.listdir(gallery_path)),
                             batch_size=16, shuffle=False)
    protocol_probes = [elem for elem in os.listdir(probes_path) if int(elem[-5]) == dist]
    print(len(protocol_probes))
    data_loader_probes = torch.utils.data.DataLoader(CommonTestDataset(probes_path, protocol_probes),
                             batch_size=16, shuffle=False)
    model = FullModel().get_backbone(path=model_path, pretrained=True).to(device)
    image_name2feature_gallery = extract_features(model, data_loader_gallery)
    print("Gallery len", len(image_name2feature_gallery))
    pd.DataFrame(image_name2feature_gallery).to_csv("gallery.csv")
    image_name2feature_probe = extract_features(model, data_loader_probes)
    print("Probes len", len(image_name2feature_probe))
    pd.DataFrame(image_name2feature_probe).to_csv("probes.csv") 

def test_scface(dist=1): 
    image_path = "/home/arina/src/scface/sc_cropped" 
    models_path =  "/home/arina/src/scface/models_croppped"
    model_path = "/home/arina/src/weights_adap/ArcFaceEpoch_9.pt"
    #model_path =  "/home/arina/src/WEIGHTS.pt"
    #model_path = "/home/arina/pretrained_Res50_Arcface_webface/Epoch_17.pt" 
    protocol_probes = [elem for elem in os.listdir(image_path) if int(elem[-5]) == dist and int(elem[-7]) <= 5]
    data_loader = torch.utils.data.DataLoader(CommonTestDataset(models_path, os.listdir(models_path)),
                             batch_size=16, shuffle=False)
    model = FullModel().get_backbone(path=model_path, pretrained=True).to(device)
    gallery = pd.DataFrame(extract_features(model, data_loader))
    pd.DataFrame(gallery).to_csv("gallery.csv")
    data_loader_probe = torch.utils.data.DataLoader(CommonTestDataset(image_path, protocol_probes),
                             batch_size=16, shuffle=False)
    probes = pd.DataFrame(extract_features(model, data_loader_probe)) 
    pd.DataFrame(probes).to_csv("probes.csv") 
    normalized = probes.apply(lambda k: k / np.linalg.norm(k)).to_numpy()
    normalized_gallery = gallery.apply(lambda k: k / np.linalg.norm(k)).to_numpy()
    predictions = np.argmax(np.dot(normalized.T, normalized_gallery), 1)
    probes, gallery = probes.T, gallery.T
    probes["label"] = list(map(lambda k: k[:3], probes.index))
    gallery["label"] = list(map(lambda k: k[:3], gallery.index))
    #print(np.where(gallery.iloc[predictions].label.values == probes.label.values))
    print(model_path, dist, sum(gallery.iloc[predictions].label.values == probes.label.values) / len(probes.label.values))

def test_scface0():
    cropped_face_folder = "./data"
    model_path = "/home/arina/src/AdapMarginArcFaceEpoch_032.pt"
    #model_path = "/home/arina/pretrained_Res50_Arcface_ms1m/Epoch_17.pt" 
    batch_size = 512
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    all_pairs = pairs_scface()
    all_elems = np.unique(np.array(all_pairs)[:, :2].flatten())
    data_loader = torch.utils.data.DataLoader(CommonTestDataset(cropped_face_folder, all_elems),
                             batch_size=batch_size, shuffle=False)

    model = FullModel().get_backbone(path=model_path, pretrained=True).to(device)
    image_name2feature = extract_features(model, data_loader)
    mean, std = test_one_model(all_pairs, image_name2feature)
    print(mean, std)

def test_all():
    res = {}
    for ep in range(10):
        tester = Tester(ep)
        res[f"Epoch {ep}"] = tester.test()
    df = pd.DataFrame(res)
    df.index = tester.test_dataset
    print(df)
    df.to_csv("epochs_res2.csv")

Tester().test()
#test_scface(1)
#Tester(7).calculate_auc("survface_112")
"""- train 18 epochs: ResNet 50 ir, 4 losses
- test LFW (112x112, 56x56, 28x28, 14x14)
- test SCface, QMUL-SurvFace 
- table with results 

Final Presentation 10.01, 13.01 - report, 20s 01 - code
1s 01 - meeting 

15 
parts: 1. methods + results 2. related work
"""

