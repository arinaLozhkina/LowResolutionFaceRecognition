import os
import random
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.append("./data")

from data.test_dataset import square_crop, get_sharpness


def transform(img):
    """
    Augmentation and Normalization of training dataset.
    """
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    image = (img.transpose((2, 0, 1)) - 127.5) * 0.0078125
    return torch.from_numpy(image.astype(np.float32))


class ImageDataset(Dataset):
    def __init__(self, data_root, train_file, mode="Approach2"):
        self.mode = mode
        self.data_root = data_root
        self.train_list = []
        train_file_buf = open(train_file)
        lines = train_file_buf.readlines()
        labels = {elem: idx for idx, elem in enumerate(os.listdir(self.data_root))}
        for line in lines:
            image_path, image_label = line.strip().split()
            self.train_list.append((image_path, labels[image_label]))
        self.cross_idx = np.random.randint(0, 5, self.__len__())
        self.new_size = {1: 56, 2: 28, 3: 14, 4: 7}

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, index):
        image_path, image_label = self.train_list[index]
        image_path = os.path.join(self.data_root, image_path)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        image = cv2.resize(square_crop(image), (112, 112))
        if self.mode == "Approach2":
            m = get_sharpness(image)
        image = transform(image)
        return image, image_label, m
