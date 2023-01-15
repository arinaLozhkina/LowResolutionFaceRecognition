import torch.nn
import torch.nn.functional as F

class CosFace(torch.nn.Module):
    def __init__(self, feat_dim, num_class, margin=0.35, s=30):
        super(CosFace, self).__init__()
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.s = s
        self.m = margin
        self.w = torch.nn.Parameter(torch.Tensor(feat_dim, num_class))
        torch.nn.init.xavier_normal_(self.w)

    def forward(self, feats, labels):
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)
        cos_theta = F.normalize(feats, dim=1).mm(self.w)
        cos_theta = cos_theta.clamp(-1, 1)

        cos_theta_m = cos_theta - self.m
        onehot = F.one_hot(labels, self.num_class)
        logits = self.s * torch.where(onehot == 1, cos_theta_m, cos_theta)
        return logits