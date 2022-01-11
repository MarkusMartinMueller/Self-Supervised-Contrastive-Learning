import torch
import torch.nn as nn
import torchvision
from utils import fusion_concat, fusion_avg, fusion_sum, fusion_max
import numpy as np


class ClassificationLoss(nn.Module):
    """
       Classification loss
    """

    def __init__(self, projection_dim, n_classes, fusion):
        super(ClassificationLoss, self).__init__()

        """

        projection_dim needs to be adjusted for fusion_concat because the dimensions is doubled
        """

        self.n_classes = n_classes
        self.fusion_method = fusion

        if fusion == "concat":
            self.projection_dim = 2 * projection_dim
        else:
            self.projection_dim = projection_dim

        self.fc = nn.Linear(self.projection_dim, self.n_classes)
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, fused, labels):

        logits = self.fc(fused)

        loss = self.criterion(logits, labels)

        return loss  # ,map

# if __name__ == "__main__":

#     cls = ClassificationLoss(projection_dim=128, n_classes=19)
#     inputs_s1 = torch.randn((4, 128))
#     inputs_s2 = torch.randn((4, 128))

#     labels = torch.randn((4, 19))
#     t = torch.Tensor([0.5])  # threshold
#     labels = (labels > t).float() * 1

#     fused = fusion_concat(inputs_s1, inputs_s2)

#     loss = cls(fused, labels)

#     print(loss)