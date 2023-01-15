import torch.nn.functional as F
import torch

class ArcFace(torch.nn.Module):
    def __init__(self, feat_dim=512, num_class=10575, s=64., margin=0.35):
        super(ArcFace, self).__init__()
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
        with torch.no_grad():
            theta_m = torch.acos(cos_theta.clamp(-1+1e-5, 1-1e-5))
            y_ = labels.view(-1, 1)
            theta_m.scatter_(1, y_, self.m, reduce='add')
            theta_m.clamp_(1e-5, 3.14159)
            d_theta = torch.cos(theta_m) - cos_theta

        logits = self.s * (cos_theta + d_theta)
        return logits