import torch

from head.adaptiveface import AdaptiveArcFace
from head.arcface import ArcFace
from head.cosface import CosFace
from head.sphereface import SphereFace
from resnet import Resnet


class FullModel(torch.nn.Module):
    def __init__(self, head_type="ArcFace", feat_dim=512, num_class=10575, path=None, pretrained=False):
        super(FullModel, self).__init__()
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.backbone = self.get_backbone(path, pretrained)
        self.head_type = head_type
        if head_type == "ArcFace":
            self.head = ArcFace(feat_dim, num_class, s=64., margin=0.35)
        elif head_type == "CosFace":
            self.head = CosFace(feat_dim, num_class, margin=0.35, s=30)
        elif head_type == "SphereFace":
            self.head = SphereFace(feat_dim, num_class, margin=0.35, s=32)
        elif head_type in ["Approach1", "Approach2"]:
            self.head = AdaptiveArcFace(feat_dim, num_class, s=64.)
        else:
            raise ValueError(f"Wrong Head Type: {head_type}")

    def get_backbone(self, path=None, pretrained=False):
        backbone = Resnet(drop_ratio=0.4, mode='ir', feat_dim=self.feat_dim, out_h=7, out_w=7)
        if pretrained:
            assert path is not None
            model_dict = backbone.state_dict()
            pretrained_dict = torch.load(path)['state_dict']
            new_pretrained_dict = {}
            for k in model_dict:
                if 'backbone.' + k in pretrained_dict.keys():
                    new_pretrained_dict[k] = pretrained_dict['backbone.' + k]
                else:
                    new_pretrained_dict[k] = pretrained_dict['module.backbone.' + k]
            model_dict.update(new_pretrained_dict)
            backbone.load_state_dict(model_dict)
        return backbone

    def forward(self, data, label, m=None):
        feat = self.backbone.forward(data)
        if self.head_type[:-1] == "Approach":
            pred = self.head.forward(feat, label, m)
        else:
            pred = self.head.forward(feat, label)
        return pred, feat
