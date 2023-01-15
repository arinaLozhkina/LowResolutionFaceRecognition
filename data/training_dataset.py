import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def transform(img):
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    image = (img.transpose((2, 0, 1)) - 127.5) * 0.0078125
    return torch.from_numpy(image.astype(np.float32))

class ImageDatasetTriplet(Dataset):
    def __init__(self, data_root, train_file):
        self.data_root = data_root
        self.all_data = defaultdict(list)
        train_file_buf = open(train_file)
        lines = train_file_buf.readlines()
        labels = {elem: idx for idx, elem in enumerate(os.listdir(self.data_root))}
        for line in lines:
            image_path, image_label = line.strip().split(' ')
            self.all_data[labels[image_label]].append(image_path)

    def __len__(self):
        return len(self.all_data.keys()) * 3

    def __getitem__(self, index):
        anchor_id = random.choice(list(self.all_data.keys()))
        anchor_img = random.choice(self.all_data[anchor_id])
        img1 = random.choice([elem for elem in self.all_data[anchor_id] if elem != anchor_img])
        img_names = [anchor_img, img1]
        labels = [anchor_id, anchor_id]
        for idx in range(4):
            label = random.choice([elem for elem in list(self.all_data.keys()) if elem not in labels])
            labels.append(label)
            img_names.append(random.choice(self.all_data[label]))
        images = []
        for img_name in img_names:
            img_path = os.path.join(self.data_root, img_name)
            image = cv2.imread(img_path)
            images.append(transform(image))
        assert len(images) == len(labels) == 6
        return images[0], images[1], images[2], images[3], images[4], images[5], labels[0], labels[1], labels[2], labels[3], labels[4], labels[5]

class ImageDataset(Dataset):
    def __init__(self, data_root, train_file, mode="DeriveNet"):
        self.mode = mode
        self.data_root = data_root
        self.train_list = []
        train_file_buf = open(train_file)
        lines = train_file_buf.readlines()
        labels = {elem: idx for idx, elem in enumerate(os.listdir(self.data_root))}
        for line in lines:
            image_path, image_label = line.strip().split(' ')
            self.train_list.append((image_path, labels[image_label]))

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, index):
        image_path, image_label = self.train_list[index]
        image_path = os.path.join(self.data_root, image_path)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if self.mode == "Approach2":
            m = self.get_sharpness(image)
        else:
            m = 0
        image = transform(image)
        return image, image_label, m

    def get_sharpness(self, img):
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        gnorm = np.sqrt(laplacian ** 2)
        return 1 / np.average(gnorm)
