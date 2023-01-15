import argparse
import os

import torch
import torchvision.transforms as T
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.training_dataset import ImageDataset
from model import FullModel


class Training(object):
    def __init__(self):
        super(Training, self).__init__()
        argParser = argparse.ArgumentParser()
        argParser.add_argument("--loss_type", default="CosFace",
                help="Type of Head for training from scratch. One of: CosFace, SphereFace, ArcFace, Approach1, Approach2")
        argParser.add_argument("--cross_resolution_batch", default=True, type=bool, 
                help="The bool value if we use cross resolution batch (downsample 50% of batch)")

        argParser.add_argument("--data_root", default='/home/arina/cropped_data/webface/casia-112x112')
        argParser.add_argument("--train_file", default='/home/arina/cropped_data/webface/train.txt')
        argParser.add_argument("--out_dir", default="./weights")

        argParser.add_argument("--save_freq", default=3000, type=int)
        argParser.add_argument("--batch_size", default=16, type=int)
        argParser.add_argument("--epoch", default=18, type=int)
        argParser.add_argument("--learning_rate", default=0.1, type=float)
        argParser.add_argument("--milestones", default=[5, 10, 15])
        argParser.add_argument("--gamma", default=0.5, type=float)
        self.args = argParser.parse_args()

        self.dataset = ImageDataset(self.args.data_root, self.args.train_file, self.args.loss_type)
        self.data_loader = DataLoader(self.dataset, self.args.batch_size, shuffle=True)
        self.device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')
        self.criterion_cross_entropy = torch.nn.CrossEntropyLoss().to(self.device)
        self.model = FullModel(head_type=self.args.loss_type)
        self.model = self.model.to(self.device)
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.SGD(parameters, lr=self.args.learning_rate,
                                   momentum=0.9, weight_decay=1e-4)
        self.lr_schedule = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.args.milestones, gamma=self.args.gamma)

    def train_epoch(self):
        for batch_idx, (img, label, m) in enumerate(self.data_loader):
            img, label = img.to(self.device), label.to(self.device)
            if batch_idx > self.args.batch_size // 2 and self.args.cross_resolution_batch:
                img = T.Resize((112, 112))(T.Resize((16, 16))(img))
            if self.args.loss_type == "Approach1":
                m = torch.ones(label.shape[0]) * 0.5 if batch_idx > self.args.batch_size // 2 \
                    else torch.ones(label.shape[0]) * 0.35
            em, embeddings = self.model.forward(img, label, m)
            loss = self.criterion_cross_entropy(em, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if (batch_idx + 1) % self.args.save_freq == 0 or (batch_idx + 1) == len(self.data_loader):
                torch.save({'state_dict': self.model.state_dict()}, os.path.join(self.args.out_dir, f'{self.args.loss_type}.pt'))
        self.lr_schedule.step()

    def train(self):
        self.model.train()
        for epoch in tqdm(range(self.args.epoch)):
            self.train_epoch()


if __name__ == '__main__':
    Training().train()

