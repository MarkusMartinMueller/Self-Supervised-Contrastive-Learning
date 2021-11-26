import torch
import torch.nn as nn
import torchvision

import math
from torch.nn import functional as F


from loss.classification_loss import ClassificationLoss



def get_loss_func(name,config):

    if name == "classification":

        cls = ClassificationLoss(projection_dim=config["projection_dim"], n_classes=19)

        return cls

    elif name == "contrastive":
        pass

    else:
        raise ValueError('Invalid Loss function.')