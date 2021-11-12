import torch
import torch.nn as nn
import torchvision
from utils.fusion import avg, sum, max, concat

class ClassificationLoss(nn.Module):
    """
       Classification loss
    """
    def __init__(self, projection_dim, n_classes):
        super(ClassificationLoss, self).__init__()
        self.projection_dim = projection_dim
        self.n_classes = n_classes

        self.fc = nn.Linear(self.projection_dim, self.n_classes)


    def forward(self,fused , labels):

        logits = self.fc(fused)

        criterion = nn.BCEWithLogitsLoss()

        loss = criterion(logits, labels)

        return loss

if __name__ == "__main__":

    cls = ClassificationLoss(projection_dim=128, n_classes=19)
    inputs_s1 = torch.randn((4, 128))
    inputs_s2 = torch.randn((4, 128))

    labels = torch.randn((4, 19))
    t = torch.Tensor([0.5])  # threshold
    labels = (labels > t).float() * 1

    fused = avg(inputs_s1, inputs_s2)

    loss = cls(fused, labels)

    print(loss)