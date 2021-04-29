import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.layers import Conv2d

class CosineSimLoss(nn.Module):
    def __init__(self,
                 with_norm=True,
                 negative=False,
                 pairwise=False):
        super().__init__()
        self.with_norm = with_norm
        self.negative = negative
        self.pairwise = pairwise
    
    def forward(self, cls_score, label, mask=None):
        if self.with_norm:
            cls_score = F.normalize(cls_score, p=2, dim=1)
            label = F.normalize(label, p=2, dim=1)
        if mask is not None:
            assert self.pairwise
        if self.pairwise:
            cls_score = cls_score.flatten(2)
            label = label.flatten(2)
            prod = torch.einsum('bci,bcj->bij', cls_score, label)
            if mask is not None:
                assert prod.shape == mask.shape
                prod *= mask.float()
            prod = prod.flatten(1)
        else:
            prod = torch.sum(
                cls_score * label, dim=1).view(cls_score.size(0), -1)
        if self.negative:
            loss = -prod.mean(dim=-1)
        else:
            loss = 2 - 2 * prod.mean(dim=-1)
        return loss


class SimSiamHead(nn.Module):
    """Classification head for I3D.
    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_feat (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 in_channels,
                 conv_mid_channels=2048,
                 conv_out_channles=2048,
                 num_convs=0,
                 kernel_size=1,
                 num_projection_fcs=3,
                 projection_mid_channels=2048,
                 projection_out_channels=2048,
                 num_predictor_fcs=2,
                 predictor_mid_channels=512,
                 predictor_out_channels=2048,
                 spatial_type='avg'):
        super().__init__()
        self.in_channels = in_channels
        self.num_convs = num_convs
        self.loss_feat = CosineSimLoss(negative=False)

        convs = []
        last_channels = in_channels
        for i in range(num_convs):
            is_last = i == num_convs - 1
            out_channels = conv_out_channles if is_last else conv_mid_channels
            convs.append(
                Conv2d(
                    last_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            last_channels = out_channels
        if len(convs) > 0:
            self.convs = nn.Sequential(*convs)
        else:
            self.convs = nn.Identity()

        projection_fcs = []
        for i in range(num_projection_fcs):
            is_last = i == num_projection_fcs - 1
            out_channels = projection_out_channels if is_last else \
                projection_mid_channels
            projection_fcs.append(nn.Linear(last_channels, out_channels))
            # no relu on output
            if not is_last:
                projection_fcs.append(nn.ReLU())
            last_channels = out_channels
        if len(projection_fcs):
            self.projection_fcs = nn.Sequential(*projection_fcs)
        else:
            self.projection_fcs = nn.Identity()

        predictor_fcs = []
        for i in range(num_predictor_fcs):
            is_last = i == num_predictor_fcs - 1
            out_channels = predictor_out_channels if is_last else \
                predictor_mid_channels
            predictor_fcs.append(nn.Linear(last_channels, out_channels))
            if not is_last:
                predictor_fcs.append(nn.ReLU())
        last_channels = out_channels
        if len(predictor_fcs):
            self.predictor_fcs = nn.Sequential(*predictor_fcs)
        else:
            self.predictor_fcs = nn.Identity()

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))


    def forward_projection(self, x):
        x = self.convs(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        z = self.projection_fcs(x)

        return z

    def forward(self, x):
        """Defines the computation performed at every call.
        Args:
            x (torch.Tensor): The input data.
        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        x = self.convs(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        z = self.projection_fcs(x)
        p = self.predictor_fcs(z)

        return z, p

    def loss(self, p1, z1, p2, z2, mask12=None, mask21=None, weight=1.):
        assert mask12 is None
        assert mask21 is None

        loss_feat = self.loss_feat(p1, z2.detach()) * 0.5 + self.loss_feat(
            p2, z1.detach()) * 0.5
        loss_feat = loss_feat * weight
        return loss_feat