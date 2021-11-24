import torch
import torch.nn as nn
import torchvision
from utils.fusion import fusion_concat,fusion_avg,fusion_sum,fusion_max
import math
from torch.nn import functional as F


def get_loss_func(name,config):

    if name == "classification":

        cls = ClassificationLoss(projection_dim=config["projection_dim"], n_classes=19)

        return cls

    elif name == "contrastive":
        pass

    else:
        raise ValueError('Invalid optimizer.')