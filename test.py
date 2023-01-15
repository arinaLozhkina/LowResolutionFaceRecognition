import argparse
import os
import yaml

import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from scipy.io import loadmat
from tqdm import tqdm

from data.test_dataset import CommonTestDataset
from model import FullModel


def extract_features(model, data_loader, device):
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
        accu_list = []
        for subset_idx in tqdm(range(10)):
            test_score_list = subsets_score_list[subset_idx]
            test_label_list = subsets_label_list[subset_idx]
            subset_train[subset_idx] = False
            train_score_list = subsets_score_list[subset_train].flatten()
            train_label_list = subsets_label_list[subset_train].flatten()
            subset_train[subset_idx] = True
            best_thres = getThreshold(train_score_list, train_label_list)
            positive_score_list = test_score_list[test_label_list == 1]
            negative_score_list = test_score_list[test_label_list == 0]
            true_pos_pairs = np.sum(positive_score_list > best_thres)
            true_neg_pairs = np.sum(negative_score_list < best_thres)
            accu_list.append((true_pos_pairs + true_neg_pairs) / len_subset)
        mean = np.mean(accu_list)
        std = np.std(accu_list, ddof=1) / np.sqrt(10) #ddof=1, division 9.
        return mean, std


def pairs_lfw_split(line_strs):
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


class testLFW(object):
    def __init__(self, path_model, path_data, path_pairs, target_size=112):
        self.device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')
        self.model = FullModel().get_backbone(path=path_model, pretrained=True).to(self.device)
        self.model.eval()
        self.batch_size = 16
        self.target_size, self.path_data, self.path_pairs = target_size, path_data, path_pairs

    def pairs_lfw(self, path):
        with open(path) as f:
            pairs_lines = list(map(lambda k: k.rstrip().split("\t"), f.readlines()[1:]))
        return list(map(pairs_lfw_split, pairs_lines))

    def get_data_verification(self):
        pairs = self.pairs_lfw(self.path_pairs)
        all_elems = np.unique(np.array(pairs)[:, :2].flatten())
        data_loader = torch.utils.data.DataLoader(CommonTestDataset(self.path_data, all_elems, self.target_size),
                                                  batch_size=self.batch_size, shuffle=False)
        return pairs, data_loader

    def test(self):
        pairs, data_loader = self.get_data_verification()
        image_name2feature = extract_features(self.model, data_loader, self.device)
        mean, std = test_one_model(pairs, image_name2feature)
        print("LFW - Mean:", mean, "Std:", std)


class testSurvFace(object):
    def __init__(self, path_model, path_data, path_pairs):
        self.device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')
        self.model = FullModel().get_backbone(path=path_model, pretrained=True).to(self.device)
        self.model.eval()
        self.batch_size = 16
        self.path_data, self.path_pairs = path_data, path_pairs

    def pairs_survface(self, path_pair):
        mode_pairs = []
        for idx, mode in enumerate(["positive", "negative"]):
            pairs = loadmat(os.path.join(path_pair, f"{mode}_pairs_names.mat"), squeeze_me=True)
            pairs = pairs[f'{mode}_pairs_names']
            nb_pairs = pairs.shape[0]
            mode_pairs.append(
                np.stack([pairs[:, 0].flatten(), pairs[:, 1].flatten(), np.ones(nb_pairs) * int(mode == "positive")]).T)
        return np.vstack(mode_pairs)

    def get_data_verification(self):
        pairs = self.pairs_survface(self.path_pairs)
        all_elems = np.unique(np.array(pairs)[:, :2].flatten())
        data_loader = torch.utils.data.DataLoader(CommonTestDataset(self.path_data, all_elems),
                                                  batch_size=self.batch_size, shuffle=False)
        return pairs, data_loader

    def test(self):
        pairs, data_loader = self.get_data_verification()
        image_name2feature = extract_features(self.model, data_loader, self.device)
        mean, std = test_one_model(pairs, image_name2feature)
        print("QMUL-SurvFace - Mean:", mean, "Std:", std)


class testSCface(object):
    def __init__(self, path_model, path_data, path_data_gallery, dist=1, time="Day"):
        self.device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')
        self.model = FullModel().get_backbone(path=path_model, pretrained=True).to(self.device)
        self.model.eval()
        self.batch_size = 16
        self.path_data, self.path_data_gallery = path_data, path_data_gallery
        self.dist = dist
        self.cameras = [1, 2, 3, 4, 5] if time == "Day" else [6, 7]

    def get_data_identification(self):
        protocol_probes = [elem for elem in os.listdir(self.path_data) if int(elem[-5]) == self.dist and int(elem[-7]) in self.cameras]
        data_loader_probe = torch.utils.data.DataLoader(CommonTestDataset(self.path_data, protocol_probes),
                                                        batch_size=16, shuffle=False)
        data_loader_gallery = torch.utils.data.DataLoader(CommonTestDataset(self.path_data_gallery, os.listdir(self.path_data_gallery)),
                                                  batch_size=16, shuffle=False)
        return data_loader_probe, data_loader_gallery

    def test(self):
        data_loader_probe, data_loader_gallery = self.get_data_identification()
        gallery_features = pd.DataFrame(extract_features(self.model, data_loader_gallery, self.device))
        probes_features = pd.DataFrame(extract_features(self.model, data_loader_probe, self.device))
        predictions = np.argmax(np.dot(probes_features.to_numpy().T, gallery_features.to_numpy()), 1)
        probes_features, gallery_features = probes_features.T, gallery_features.T
        probes_features["label"] = list(map(lambda k: k[:3], probes_features.index))
        gallery_features["label"] = list(map(lambda k: k[:3], gallery_features.index))
        print("SCface - Accuracy:", sum(gallery_features.iloc[predictions].label.values == probes_features.label.values) / len(probes_features.label.values))

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--dataset", help="Which LR dataset to use: LFW, QMUL-SurvFace, SCface", default="LFW")
    '''argParser.add_argument("--path_model", help="Path to weights", default='/home/arina/src/weights/ArcFace.pt')
    argParser.add_argument("--path_data", help="Path to test images", default='/home/arina/cropped_data/lfw/lfw_crop')
    argParser.add_argument("--path_pairs", help="Path to pairs if using LFW, QMUL-SurvFace", default='/home/arina/cropped_data/lfw/pairs.txt')
    argParser.add_argument("--path_pairs_negative", help="Path to negative pairs if using QMUL-SurvFace (the path_pairs is for positive pairs)",
                           default='/home/arina/cropped_data/qmul/Face_Verification_Test_Set/negative_pairs_names.mat')
    argParser.add_argument("--path_gallery", help="Path to gallery images if using SCface", default='/home/arina/cropped_data/scface/models_croppped')
    argParser.add_argument("--scface_dist", help="Distance to test SCface", default=1, type=int)'''
    args = argParser.parse_args()
    with open("data_conf.yaml", "r") as stream:
        try:
            data_conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
   
    config = data_conf[args.dataset]
    if args.dataset == "LFW":
        testLFW(config['path_model'], config['path_data'], config['path_pairs'], config['target_size']).test()
    elif args.dataset == "QMUL-SurvFace":
        testSurvFace(config['path_model'], config['path_data'], config['path_pairs']).test()
    elif args.dataset == "SCface":
        testSCface(config['path_model'], config['path_data'], config['path_data_gallery'], dist=config['scface_dist']).test()
    else:
        raise ValueError(f"Wrong dataset: {args.dataset}, choose one of: LFW, QMUL-SurvFace, SCface")
