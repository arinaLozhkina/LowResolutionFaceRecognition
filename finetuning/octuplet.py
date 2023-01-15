import argparse
import os
import sys
sys.path.append('..')

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.training_dataset import ImageDatasetTriplet
from model import FullModel


class Training(object):
    def __init__(self):
        super(Training, self).__init__()
        argParser = argparse.ArgumentParser()
        argParser.add_argument("--data_root", default='/home/arina/cropped_data/webface/casia-112x112')
        argParser.add_argument("--train_file", default='/home/arina/cropped_data/webface/train.txt')
        argParser.add_argument("--pretrain_model", default='/home/arina/src/weights/ArcFace.pt')
        argParser.add_argument("--out_dir", default="../weights")

        argParser.add_argument("--save_freq", default=3000, type=int)
        argParser.add_argument("--batch_size", default=16, type=int)
        argParser.add_argument("--epoch", default=18, type=int)
        argParser.add_argument("--learning_rate", default=0.1, type=float)
        argParser.add_argument("--milestones", default=[5, 10, 15])
        argParser.add_argument("--gamma", default=0.5, type=float)
        self.args = argParser.parse_args()

        self.dataset = ImageDatasetTriplet(self.args.data_root, self.args.train_file)
        self.data_loader = DataLoader(self.dataset, self.args.batch_size, shuffle=True)
        self.device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')
        self.model = FullModel().get_backbone(path=self.args.pretrain_model, pretrained=True)
        self.model = self.model.to(self.device)
        self.criterion_triplet = torch.nn.TripletMarginWithDistanceLoss(distance_function=
                                            lambda x, y: 1.0 - F.cosine_similarity(x, y)).to(self.device)
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.SGD(parameters, lr=self.args.learning_rate,
                                   momentum=0.9, weight_decay=1e-4)
        self.lr_schedule = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.args.milestones, gamma=self.args.gamma)

    def train_epoch_triplet(self):
        for batch_idx, (
        anchor, pos, neg1, neg2, neg3, neg4, anchor_id, pos_id, neg_id1, neg_id2, neg_id3, neg_id4) in enumerate(
                self.data_loader):
            anchor, pos, neg1, neg2, neg3, neg4 = anchor.to(self.device), pos.to(self.device), neg1.to(
                self.device), neg2.to(self.device), neg3.to(self.device), neg4.to(self.device)
            out_anchor = self.model.forward(anchor)
            out_anchor_low = self.model.forward(T.Resize(size=(112, 112))(T.Resize(size=(20, 20))(anchor)))
            out_pos = self.model.forward(pos)
            out_pos_low = self.model.forward(T.Resize(size=(112, 112))(T.Resize(size=(20, 20))(pos)))
            out_neg1 = self.model.forward(neg1)
            out_neg2 = self.model.forward(T.Resize(size=(112, 112))(T.Resize(size=(20, 20))(neg2)))
            out_neg3 = self.model.forward(neg3)
            out_neg4 = self.model.forward(T.Resize(size=(112, 112))(T.Resize(size=(20, 20))(neg4)))

            hrhr = self.criterion_triplet(out_anchor, out_pos, out_neg1)
            lrhr = self.criterion_triplet(out_anchor_low, out_pos, out_neg3)
            hrlr = self.criterion_triplet(out_anchor, out_pos_low, out_neg2)
            lrlr = self.criterion_triplet(out_anchor_low, out_pos_low, out_neg4)
            loss = hrhr + hrlr  + lrhr + lrlr
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if (batch_idx + 1) % self.args.save_freq == 0 or (batch_idx + 1) == len(self.data_loader):
                torch.save({'state_dict': self.model.state_dict()}, os.path.join(self.args.out_dir, "Octuplet.pt"))
        self.lr_schedule.step()

    def train(self):
        self.model.train()
        for epoch in tqdm(range(self.args.epoch)):
            self.train_epoch_triplet()


if __name__ == '__main__':
    Training().train()

