
import argparse
import os
import yaml

import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from scipy.io import loadmat
from tqdm import tqdm
from scipy.interpolate import interp1d
from torchcam.methods import SmoothGradCAMpp, GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import normalize, resize, to_pil_image
from matplotlib import pyplot as plt
import torchvision

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
            for batch_idx, (images, filenames, m) in tqdm(enumerate(data_loader)):
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
        if type(num_thresholds) != list:
            threshold_list = score_min +  step * np.array(range(1, num_thresholds + 1))
        else:
            threshold_list = num_thresholds 
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
        return  best_thres, tpr, fpr


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
        far_levels = [0.3, 0.1, 0.01, 0.001]
        for subset_idx in tqdm(range(10)):
            test_score_list = subsets_score_list[subset_idx]
            test_label_list = subsets_label_list[subset_idx]
            subset_train[subset_idx] = False
            train_score_list = subsets_score_list[subset_train].flatten()
            train_label_list = subsets_label_list[subset_train].flatten()
            subset_train[subset_idx] = True
        
            best_thres, tar, far = getThreshold(train_score_list, train_label_list)
            if subset_idx == 0:
                interp = interp1d(far, tar)
                tar_at_far = [interp(x) for x in far_levels]
                for f, fa in enumerate(far_levels):
                    print('TPR@FAR={:.7f}: {:.4f}'.format(fa, tar_at_far[f]))
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
        self.batch_size = 1
        self.target_size, self.path_data, self.path_pairs = target_size, path_data, path_pairs
        self.path = f"./res/res_{path_model.split('/')[-1]}_{self.target_size}"
        if not os.path.exists(self.path):
            os.makedirs(self.path)

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

    def class_activation_map(self):
        pairs, data_loader = self.get_data_verification()
        for batch_idx, (images, filenames, m) in tqdm(enumerate(data_loader)):
                    #images[0] = cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB)
                    images = images.to(self.device)
                    layers = [f"body.{elem}.res_layer" for elem in range(24)]
                    cam_extractor = SmoothGradCAMpp(self.model, "body.23", input_shape=(3, 112, 112))
                    features = self.model(images)
                    activation_map = cam_extractor(features.squeeze(0).argmax().item(), features)
                    plt.imshow(to_pil_image(images[0])); plt.axis('off'); plt.tight_layout(); plt.savefig(os.path.join(self.path, f"./results_orig_{batch_idx}.png")); 
                    result = overlay_mask(to_pil_image(images[0]), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
                    plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.savefig(os.path.join(self.path, f"./results_{batch_idx}.png"));
                    if batch_idx > 5:
                        break 
   
    def get_margin(self):
        margins = []
        pairs, data_loader = self.get_data_verification()
        for batch_idx, (images, filenames, m) in tqdm(enumerate(data_loader)):
                margins = np.append(margins, m.cpu().numpy())
        with open('margins.npy', 'wb') as f:
            np.save(f, margins)

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
    
    def get_margin(self):
        margins = []
        pairs, data_loader = self.get_data_verification()
        for batch_idx, (images, filenames, m) in tqdm(enumerate(data_loader)):
                margins = np.append(margins, m.cpu().numpy())
        with open('margins.npy', 'wb') as f:
            np.save(f, margins)

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
        gallery_features.to_csv("./gallery.csv")
        probes_features = pd.DataFrame(extract_features(self.model, data_loader_probe, self.device))
        probes_features.to_csv("./probes.csv")
        predictions = np.argmax(np.dot(probes_features.to_numpy().T, gallery_features.to_numpy()), 1)
        probes_features, gallery_features = probes_features.T, gallery_features.T
        probes_features["label"] = list(map(lambda k: k[:3], probes_features.index))
        gallery_features["label"] = list(map(lambda k: k[:3], gallery_features.index))
        print("SCface - Accuracy:", sum(gallery_features.iloc[predictions].label.values == probes_features.label.values) / len(probes_features.label.values))

    def get_margin(self):
        margins = []
        data_loader, _ = self.get_data_identification()
        #pairs, data_loader = self.get_data_verification()
        for batch_idx, (images, filenames, m) in tqdm(enumerate(data_loader)):
                margins = np.append(margins, m.cpu().numpy())
        with open('margins.npy', 'wb') as f:
            np.save(f, margins)

class testTinyFace(object):
    """
    TinyFace: Face Recognition in Native Low-resolution Imagery
    https://github.com/mk-minchul/AdaFace/blob/master/validation_lq/validate_tinyface.py
    """
    def __init__(self, path_model, path_data, alignment_dir_name="aligned_pad_0.1_pad_high"):
        self.device = torch.device('cuda:0') if torch.cuda.is_available else 'cpu'
        self.model = FullModel().get_backbone(path=path_model, pretrained=True).to(self.device)
        self.model.eval()
        self.batch_size = 16
        self.path_data = path_data

        self.gallery_dict = loadmat(
            os.path.join(path_data, 'tinyface/Testing_Set/gallery_match_img_ID_pairs.mat'))
        self.probe_dict = loadmat(
            os.path.join(path_data, 'tinyface/Testing_Set/probe_img_ID_pairs.mat'))
        self.proto_gal_paths = [os.path.join(path_data, alignment_dir_name, 'Gallery_Match', p[0].item()) for p
                                in self.gallery_dict['gallery_set']]
        self.proto_prob_paths = [os.path.join(path_data, alignment_dir_name, 'Probe', p[0].item()) for p in
                                 self.probe_dict['probe_set']]
        self.proto_distractor_paths = self.get_all_files(
            os.path.join(path_data, alignment_dir_name, 'Gallery_Distractor'))

        self.image_paths = self.get_all_files(os.path.join(path_data, alignment_dir_name))
        self.image_paths = np.array(self.image_paths).astype(object).flatten()

        self.probe_paths = self.get_all_files(os.path.join(path_data, 'tinyface/Testing_Set/Probe'))
        self.probe_paths = np.array(self.probe_paths).astype(object).flatten()

        self.gallery_paths = self.get_all_files(os.path.join(path_data, 'tinyface/Testing_Set/Gallery_Match'))
        self.gallery_paths = np.array(self.gallery_paths).astype(object).flatten()

        self.distractor_paths = self.get_all_files(
            os.path.join(path_data, 'tinyface/Testing_Set/Gallery_Distractor'))
        self.distractor_paths = np.array(self.distractor_paths).astype(object).flatten()

        self.init_proto(self.probe_paths, self.gallery_paths, self.distractor_paths)

    def get_key(self, image_path):
        return os.path.splitext(os.path.basename(image_path))[0]

    def get_label(self, image_path):
        return int(os.path.basename(image_path).split('_')[0])

    def get_all_files(self, root, extension_list=['.jpg', '.png', '.jpeg']):
        all_files = list()
        for (dirpath, dirnames, filenames) in os.walk(root):
            all_files += [os.path.join(dirpath, file) for file in filenames]
        if extension_list is None:
            return all_files
        all_files = list(filter(lambda x: os.path.splitext(x)[1] in extension_list, all_files))
        return all_files

    def init_proto(self, probe_paths, match_paths, distractor_paths):
        index_dict = {}
        for i, image_path in enumerate(self.image_paths):
            index_dict[self.get_key(image_path)] = i

        self.indices_probe = np.array([index_dict[self.get_key(img)] for img in probe_paths])
        self.indices_match = np.array([index_dict[self.get_key(img)] for img in match_paths])
        self.indices_distractor = np.array([index_dict[self.get_key(img)] for img in distractor_paths])

        self.labels_probe = np.array([self.get_label(img) for img in probe_paths])
        self.labels_match = np.array([self.get_label(img) for img in match_paths])
        self.labels_distractor = np.array([-100 for img in distractor_paths])

        self.indices_gallery = np.concatenate([self.indices_match, self.indices_distractor])
        self.labels_gallery = np.concatenate([self.labels_match, self.labels_distractor])

    def get_data_identification(self):
        data_loader_probe = torch.utils.data.DataLoader(CommonTestDataset(self.path_data, self.image_paths[self.indices_probe]),
                                                        batch_size=self.batch_size, shuffle=False)
        data_loader_gallery = torch.utils.data.DataLoader(
            CommonTestDataset(self.path_data, self.image_paths[self.indices_gallery]),
            batch_size=self.batch_size, shuffle=False)
        return data_loader_probe, data_loader_gallery

    def test(self, ranks=[1,5,20]):
        data_loader_probe, data_loader_gallery = self.get_data_identification()
        gallery_features = pd.DataFrame(extract_features(self.model, data_loader_gallery, self.device))
        probes_features = pd.DataFrame(extract_features(self.model, data_loader_probe, self.device))
        compare_func = inner_product
        score_mat = compare_func(probes_features, gallery_features)

        label_mat = self.labels_probe[:,None] == self.labels_gallery[None,:]

        results, _, __ = DIR_FAR(score_mat, label_mat, ranks)
        print("TinyFace results", results)
        return results

    def test_acc(self):
        data_loader_probe, data_loader_gallery = self.get_data_identification()
        gallery_features = pd.DataFrame(extract_features(self.model, data_loader_gallery, self.device))
        probes_features = pd.DataFrame(extract_features(self.model, data_loader_probe, self.device))
        predictions = np.argmax(np.dot(probes_features.to_numpy().T, gallery_features.to_numpy()), 1)
        probes_features, gallery_features = probes_features.T, gallery_features.T
        probes_features["label"] = list(map(lambda k: k[:3], probes_features.index))
        gallery_features["label"] = list(map(lambda k: k[:3], gallery_features.index))
        print("TinyFace - Accuracy:",
              sum(gallery_features.iloc[predictions].label.values == probes_features.label.values) / len(
                  probes_features.label.values))


def inner_product(x1, x2):
    x1, x2 = np.array(x1), np.array(x2)
    if x1.ndim == 3:
        raise ValueError('why?')
        x1, x2 = x1[:,:,0], x2[:,:,0]
    return np.dot(x1.T, x2)



def DIR_FAR(score_mat, label_mat, ranks=[1], FARs=[1.0], get_false_indices=False):
    '''
    Code borrowed from https://github.com/seasonSH/Probabilistic-Face-Embeddings

    Closed/Open-set Identification.
        A general case of Cummulative Match Characteristic (CMC)
        where thresholding is allowed for open-set identification.
    args:
        score_mat:            a P x G matrix, P is number of probes, G is size of gallery
        label_mat:            a P x G matrix, bool
        ranks:                a list of integers
        FARs:                 false alarm rates, if 1.0, closed-set identification (CMC)
        get_false_indices:    not implemented yet
    return:
        DIRs:                 an F x R matrix, F is the number of FARs, R is the number of ranks,
                              flatten into a vector if F=1 or R=1.
        FARs:                 an vector of length = F.
        thredholds:           an vector of length = F.
    '''
    assert score_mat.shape==label_mat.shape
    # assert np.all(label_mat.astype(np.float32).sum(axis=1) <=1 )
    # Split the matrix for match probes and non-match probes
    # subfix _m: match, _nm: non-match
    # For closed set, we only use the match probes
    match_indices = label_mat.astype(np.bool).any(axis=1)
    score_mat_m = score_mat[match_indices,:]
    label_mat_m = label_mat[match_indices,:]
    score_mat_nm = score_mat[np.logical_not(match_indices),:]
    label_mat_nm = label_mat[np.logical_not(match_indices),:]

    print('mate probes: %d, non mate probes: %d' % (score_mat_m.shape[0], score_mat_nm.shape[0]))

    # Find the thresholds for different FARs
    max_score_nm = np.max(score_mat_nm, axis=1)
    label_temp = np.zeros(max_score_nm.shape, dtype=np.bool)
    if len(FARs) == 1 and FARs[0] >= 1.0:
        # If only testing closed-set identification, use the minimum score as threshold
        # in case there is no non-mate probes
        thresholds = [np.min(score_mat) - 1e-10]
        openset = False
    else:
        # If there is open-set identification, find the thresholds by FARs.
        assert score_mat_nm.shape[0] > 0, "For open-set identification (FAR<1.0), there should be at least one non-mate probe!"
        thresholds = find_thresholds_by_FAR(max_score_nm, label_temp, FARs=FARs)
        openset = True

    # Sort the labels row by row according to scores
    sort_idx_mat_m = np.argsort(score_mat_m, axis=1)
    sorted_label_mat_m = np.ndarray(label_mat_m.shape, dtype=np.bool)
    for row in range(label_mat_m.shape[0]):
        sort_idx = (sort_idx_mat_m[row, :])[::-1]
        sorted_label_mat_m[row,:] = label_mat_m[row, sort_idx]

    # Calculate DIRs for different FARs and ranks
    if openset:
        gt_score_m = score_mat_m[label_mat_m]
        assert gt_score_m.size == score_mat_m.shape[0]

    DIRs = np.zeros([len(FARs), len(ranks)], dtype=np.float32)
    FARs = np.zeros([len(FARs)], dtype=np.float32)
    if get_false_indices:
        false_retrieval = np.zeros([len(FARs), len(ranks), score_mat_m.shape[0]], dtype=np.bool)
        false_reject = np.zeros([len(FARs), len(ranks), score_mat_m.shape[0]], dtype=np.bool)
        false_accept = np.zeros([len(FARs), len(ranks), score_mat_nm.shape[0]], dtype=np.bool)
    for i, threshold in enumerate(thresholds):
        for j, rank  in enumerate(ranks):
            success_retrieval = sorted_label_mat_m[:,0:rank].any(axis=1)
            if openset:
                success_threshold = gt_score_m >= threshold
                DIRs[i,j] = (success_threshold & success_retrieval).astype(np.float32).mean()
            else:
                DIRs[i,j] = success_retrieval.astype(np.float32).mean()
            if get_false_indices:
                false_retrieval[i,j] = ~success_retrieval
                false_accept[i,j] = score_mat_nm.max(1) >= threshold
                if openset:
                    false_reject[i,j] = ~success_threshold
        if score_mat_nm.shape[0] > 0:
            FARs[i] = (max_score_nm >= threshold).astype(np.float32).mean()

    if DIRs.shape[0] == 1 or DIRs.shape[1] == 1:
        DIRs = DIRs.flatten()

    if get_false_indices:
        return DIRs, FARs, thresholds, match_indices, false_retrieval, false_reject, false_accept, sort_idx_mat_m
    else:
        return DIRs, FARs, thresholds


# Find thresholds given FARs
# but the real FARs using these thresholds could be different
# the exact FARs need to recomputed using calcROC
def find_thresholds_by_FAR(score_vec, label_vec, FARs=None, epsilon=1e-5):
    #     Code borrowed from https://github.com/seasonSH/Probabilistic-Face-Embeddings

    assert len(score_vec.shape)==1
    assert score_vec.shape == label_vec.shape
    assert label_vec.dtype == np.bool
    score_neg = score_vec[~label_vec]
    score_neg[::-1].sort()
    # score_neg = np.sort(score_neg)[::-1] # score from high to low
    num_neg = len(score_neg)

    assert num_neg >= 1

    if FARs is None:
        thresholds = np.unique(score_neg)
        thresholds = np.insert(thresholds, 0, thresholds[0]+epsilon)
        thresholds = np.insert(thresholds, thresholds.size, thresholds[-1]-epsilon)
    else:
        FARs = np.array(FARs)
        num_false_alarms = np.round(num_neg * FARs).astype(np.int32)

        thresholds = []
        for num_false_alarm in num_false_alarms:
            if num_false_alarm==0:
                threshold = score_neg[0] + epsilon
            else:
                threshold = score_neg[num_false_alarm-1]
            thresholds.append(threshold)
        thresholds = np.array(thresholds)

    return thresholds



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

    for target in [112, 14, 7]:
        testLFW(config['path_model'], config['path_data'], config['path_pairs'], target).class_activation_map()
    """
    if args.dataset == "LFW":
        testLFW(config['path_model'], config['path_data'], config['path_pairs'], config['target_size']).test()
    elif args.dataset == "TinyFace":
        testTinyFace(config['path_model'], config['path_data']).test()
    elif args.dataset == "QMUL-SurvFace":
        testSurvFace(config['path_model'], config['path_data'], config['path_pairs']).test()
    elif args.dataset == "SCface":
        testSCface(config['path_model'], config['path_data'], config['path_data_gallery'], dist=config['scface_dist']).test()
    else:
        raise ValueError(f"Wrong dataset: {args.dataset}, choose one of: LFW, QMUL-SurvFace, SCface")
    """ 
