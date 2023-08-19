import os

import numpy as np
import pyiqa
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.training_dataset import ImageDataset
from model import FullModel


class Training(object):
    def __init__(self):
        super(Training, self).__init__()
        self.loss_type = "Approach2"  # Type of Head for training from scratch. One of: CosFace, SphereFace, ArcFace, Approach2
        self.weights_path = '/home/arina/LowResolutionFaceRecognition/src/weights_good/Approach2_0_160169.pt'
        self.data_root = '/home/arina/LowResolutionFaceRecognition/cropped_data/webface/casia-112x112'
        self.train_file = '/home/arina/LowResolutionFaceRecognition/cropped_data/webface/train_new.txt'
        self.out_dir = "./weights"

        self.save_freq = 5000
        self.batch_size = 64
        self.epoch = 20
        self.learning_rate = 0.1
        self.milestones = [5, 10, 15]
        self.gamma = 0.5

        self.dataset = ImageDataset(self.data_root, self.train_file, self.loss_type)
        self.data_loader = DataLoader(self.dataset, self.batch_size, shuffle=True)
        self.device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')

        self.criterion_cross_entropy = torch.nn.CrossEntropyLoss().to(self.device)
        self.model = FullModel(feat_dim=512, head_type=self.loss_type, pretrained=False)
        self.model = self.model.to(self.device)
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.SGD(parameters, lr=self.learning_rate,
                                   momentum=0.9, weight_decay=1e-4)
        # self.optimizer.load_state_dict(torch.load(self.weights_path)['optimizer_state_dict'])
        self.lr_schedule = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.milestones, gamma=self.gamma)

    def get_margin(self):
        margins = []
        bri = []
        iqa_metric = pyiqa.create_metric('brisque', device=self.device)
        for batch_idx, (img, label, m) in tqdm(enumerate(self.data_loader)):
            margins.append(m.cpu().numpy())
            bri.append(iqa_metric(img).cpu().numpy())
        with open('margins.npy', 'wb') as f:
            np.save(f, np.concatenate(margins))
        with open('bri.npy', 'wb') as f:
            np.save(f, np.concatenate(bri))

    def train_epoch(self, epoch):
        for batch_idx, (img, label, m) in tqdm(enumerate(self.data_loader)):
            img, label = img.to(self.device), label.to(self.device)
            em, embeddings = self.model.forward(img, label, m)
            loss = self.criterion_cross_entropy(em, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if (batch_idx + 1) % self.save_freq == 0 or (batch_idx + 1) == len(self.data_loader):
                torch.save({'state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()},
                           os.path.join(self.out_dir, f'{self.loss_type}_{epoch}_{batch_idx}.pt'))

    def train(self):
        self.model.train()
        for epoch in range(self.epoch):
            print("Epoch:", epoch)
            self.train_epoch(epoch)

    def extract_features(self):
        self.model.eval()
        image_name2feature = {"filename": [], "feature": [], "margin": []}
        with torch.no_grad():
            for batch_idx, (images, label, m) in tqdm(enumerate(self.data_loader)):
                images, label = images.to(self.device), label.to(self.device)
                em, features = self.model(images, label, m)
                features = F.normalize(features)
                features = features.cpu().numpy()
                for filename, feature in zip(label, features):
                    image_name2feature["filename"].append(filename)
                    image_name2feature["feature"].append(feature)
                    image_name2feature["margin"].append(m)
        return image_name2feature


if __name__ == '__main__':
    Training().train()
