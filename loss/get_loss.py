import torch
import torch.nn as nn
import torchvision

import math
from torch.nn import functional as F

from loss import ClassificationLoss


def get_loss_func(name, projection_dim, fusion):
    if name == "classification":

        cls = ClassificationLoss(projection_dim=projection_dim, n_classes=19, fusion=fusion)

        return cls

    elif name == "contrastive":
        pass

    else:
        raise ValueError('Invalid Loss function.')
