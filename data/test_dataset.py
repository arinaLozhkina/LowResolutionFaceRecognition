import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def square_crop(im, S=112):
    """
    Crops images preserving the proportions of faces by padding if necessary.
    """
    if im.shape[0] > im.shape[1]:
        height = S
        width = int(float(im.shape[1]) / im.shape[0] * S)
    else:
        width = S
        height = int(float(im.shape[0]) / im.shape[1] * S)
    resized_im = cv2.resize(im, (width, height))
    det_im = np.zeros((S, S, 3), dtype=np.uint8)
    det_im[:resized_im.shape[0], :resized_im.shape[1], :] = resized_im
    return det_im


def get_sharpness(img):
    """
    Calculates Discrete Laplacian Norm
    """
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    gnorm = np.sqrt(laplacian ** 2)
    return 1 / np.average(gnorm)


class CommonTestDataset(Dataset):
    """ Data processor for model evaluation.

    Attributes:
        image_root(str): root directory of test set.
        image_list(list): the image list file.
        target_size(int): the required size of the image.
    """

    def __init__(self, image_root, image_list, target_size=112):
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
        image = square_crop(image) if image.shape[1] != 112 else image
        if self.target_size != 112:
            image = cv2.resize(cv2.resize(image, (self.target_size, self.target_size)), (112, 112))
        image = image.transpose((2, 0, 1))
        image = (image - self.mean) / self.std
        m = get_sharpness(image)
        image = torch.from_numpy(image.astype(np.float32))
        return image, short_image_path, m
