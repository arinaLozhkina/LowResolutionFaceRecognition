import argparse
import os
import sys
sys.path.append('..')

import torch
import torchvision.transforms as T
from torch import linalg as LA
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.training_dataset import ImageDataset
from head.dmargin import DMargin
from model import FullModel


class ReconstructModel(torch.nn.Module):
    def __init__(self, feat_dim=512, image_size=112):
        super(ReconstructModel, self).__init__()
        self.fc = torch.nn.Sequential(torch.nn.Linear(feat_dim, 1024),
                                    torch.nn.Linear(1024, 2048),
                                    torch.nn.Linear(2048, image_size * image_size * 3))

    def forward(self, x):
        return self.fc(x)


class ClassificationModel(torch.nn.Module):
    def __init__(self, feat_dim=512, num_classes=10575):
        super(ClassificationModel, self).__init__()
        self.lin1 = torch.nn.Linear(feat_dim, feat_dim)
        self.lin2 = torch.nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        out = self.lin1(x)
        out1 = self.lin2(out)
        return out1


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
        argParser.add_argument("--lambda_loss", default=1e-3, help="The weight of ReCent Loss", type=float)
        self.args = argParser.parse_args()

        self.dataset = ImageDataset(self.args.data_root, self.args.train_file)
        self.data_loader = DataLoader(self.dataset, self.args.batch_size, shuffle=True)
        self.device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')
        self.model = FullModel().get_backbone(path=self.args.pretrain_model, pretrained=True).to(self.device)
        self.criterion_cross_entropy = torch.nn.CrossEntropyLoss().to(self.device)
        self.reconstruct_model = ReconstructModel().to(self.device)
        self.classification_model = ClassificationModel(num_classes=self.dataset.__len__()).to(self.device)
        self.derived_margin_softmax_loss = DMargin(device=self.device).to(self.device)
        parameters = [p for p in list(self.reconstruct_model.parameters()) + list(self.classification_model.parameters())
                      if p.requires_grad]
        self.optimizer = optim.SGD(parameters, lr=self.args.learning_rate,
                                   momentum=0.5, weight_decay=1e-4)
        self.lr_schedule = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.args.milestones, gamma=self.args.gamma)
        self.centers = torch.zeros(10575, 112 * 112 * 3)
        self.counts = torch.zeros(10575)

    def reconstruct_model_loss(self, reconstruction, img, center):
        return self.args.lambda_loss * torch.sum((LA.norm(torch.flatten(img, start_dim=1) - reconstruction, dim=1,
                                                     ord=2) ** 2 + LA.norm(reconstruction - center, dim=1, ord=2) ** 2))

    def train_epoch(self):
        for batch_idx, (img, label, m) in enumerate(self.data_loader):
            img, label = img.to(self.device), label.to(self.device)
            low_embeddings = self.model.forward(T.Resize(size=(112, 112))(T.Resize(size=(16, 16))(img)))
            cur_centers = []
            reconstruction = self.reconstruct_model(low_embeddings)
            for cur_idx, cur_label in enumerate(label):
                cur_label = int(cur_label.detach().cpu().numpy())
                self.counts[cur_label] += 1
                self.centers[cur_label] += reconstruction.clone()[cur_idx].detach().cpu()
                cur_centers.append((self.centers[cur_label] / self.counts[cur_label]).to(self.device))
            reconstruction_loss = self.reconstruct_model_loss(reconstruction, img, torch.stack(cur_centers))
            softmax_loss = self.derived_margin_softmax_loss(low_embeddings, label, self.centers, self.counts)
            loss = (reconstruction_loss + softmax_loss) / len(label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if (batch_idx + 1) % self.args.save_freq == 0 or (batch_idx + 1) == len(self.data_loader):
                torch.save({'state_dict': self.model.state_dict()}, os.path.join(self.args.out_dir, 'DeriveNet.pt'))
        self.lr_schedule.step()

    def train(self):
        self.model.train()
        for epoch in tqdm(range(self.args.epoch)):
            self.train_epoch()


if __name__ == '__main__':
    Training().train()

