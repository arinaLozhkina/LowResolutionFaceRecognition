import torch.nn
import torch.nn.functional as F

import torch.nn
import torch.nn.functional as F


class DMargin(torch.nn.Module):
    def __init__(self, feat_dim=512, num_class=10575, device=torch.device("cpu")):
        super(DMargin, self).__init__()
        self.device = device
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.w = torch.nn.Parameter(torch.Tensor(feat_dim, num_class))
        torch.nn.init.xavier_normal_(self.w)
        self.cosine_similarity = lambda a, b: torch.dot(a, b) / torch.linalg.norm(a) / torch.linalg.norm(b)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, feats, labels, centers, counts):
        original = feats.mm(self.w)
        dists = []
        for label in labels:
            idx = int(label.detach().cpu().numpy())
            cur_center = centers[idx] / counts[idx]
            dists.append([])
            for elem in range(len(counts)):
                mean = centers[elem] / counts[elem]
                dists[-1].append(self.cosine_similarity(cur_center, mean))
        added = original + torch.FloatTensor(dists).to(self.device)
        onehot = F.one_hot(labels, self.num_class)
        logits = torch.where(onehot == 1, added, original)
        return self.criterion(logits, labels)
