import torch
import torch.nn as nn
import torchvision

import math
from torch.nn import functional as F

from loss import ClassificationLoss
from loss import NTxentLoss


def get_loss_func(name, projection_dim, fusion,temperature):
    if name == "classification":

        cls = ClassificationLoss(projection_dim=projection_dim, n_classes=19, fusion=fusion)

        return cls

    elif name == "contrastive":
        nxt = NTxentLoss(temperature=temperature)

    else:
        raise ValueError('Invalid Loss function.')
