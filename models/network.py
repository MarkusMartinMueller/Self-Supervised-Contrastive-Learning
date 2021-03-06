"""
  File: network.py
  Authors: Markus Mueller (markus.m.mueller@campus.tu-berlin.de)
  Created: 202-11-09
"""
import h5py
from torch.utils.data import DataLoader

import torch
from torch import nn

# local imports
from models import ResNet50_S1, ResNet50_S2, ResNet50_joint
from data import dataGenBigEarthLMDB_joint
from utils import fusion_concat, fusion_avg, fusion_sum, fusion_max
from loss import ClassificationLoss, NTxentLoss


class TwoBranch(nn.Module):

    def __init__(self, encoder_s1, encoder_s2, encoder_joint, projection_dim, n_features, type):
        super(TwoBranch, self).__init__()
        """
        As encoder the commonly used ResNet50 is adopted to obtain hi = f(xi)
        As projection head a mlp with one hidden layer is used to obtain g(hi) = W_2(sigma(W_1hi) with sigma = ReLU
        """

        self.encoder_s1 = encoder_s1
        self.encoder_s2 = encoder_s2
        self.encoder_joint = encoder_joint
        self.n_features = n_features
        self.type = type

        # add mlp projection head

        self.projection_head_s1 = nn.Sequential(

            nn.Linear(self.n_features, projection_dim),
            nn.ReLU()
        )

        self.projection_head_s2 = nn.Sequential(
            nn.Linear(self.n_features, projection_dim),
            nn.ReLU()

        )

    def forward(self, s_1, s_2):
        """
        s_1: torch.tensor, sentinel 1 input [batch_size,bands,120,120]
        s_2: torch.tensor, sentinel 2 input [batch_size,bands,120,120]
        type: string separate or joint

        separate: two different models for each modality
        joint: same model with a preceding 1x1 conv to get the same number of channels

        """

        if self.type == "separate":
            return self.forward_separate(s_1, s_2)
        if self.type == "joint":
            return self.forward_joint(s_1, s_2)

    def forward_separate(self, s_1, s_2):

        h_i = self.encoder_s1(s_1)
        h_j = self.encoder_s2(s_2)

        ##flatten to e.g Resnet50 [batch_size,2048]
        h_i = torch.flatten(h_i, 1)
        h_j = torch.flatten(h_j, 1)

        ##output dim could be e.g. [batch_size,128]
        projection_i = self.projection_head_s1(h_i)
        projection_j = self.projection_head_s2(h_j)

        return h_i, h_j, projection_i, projection_j

    def forward_joint(self, s_1, s_2):

        h_i = self.encoder_joint(s_1)
        h_j = self.encoder_joint(s_2)

        ##flatten to e.g Resnet50 [batch_size,2048]
        h_i = torch.flatten(h_i, 1)
        h_j = torch.flatten(h_j, 1)

        ##output dim could be e.g. [batch_size,128]
        projection_i = self.projection_head_s1(h_i)
        projection_j = self.projection_head_s2(h_j)

        return h_i, h_j, projection_i, projection_j


def get_model(path_type, n_features, projection_dim, out_channels):
    """
    :path_type:  str, either named joint or separate to indicate the model
    :rtype:
    """
    resnet_s1 = ResNet50_S1()
    resnet_s2 = ResNet50_S2()
    resnet_joint = ResNet50_joint(out_channels=out_channels)
    net = TwoBranch(resnet_s1, resnet_s2, resnet_joint, n_features=n_features, projection_dim=projection_dim,
                    type=path_type)

    return net


if __name__ == "__main__":
    from utils.utils import MetricTracker
    from models import ResNet50_bands_12
    train_csv = "C:/Users/Markus/Desktop/splits_mm_serbia/train.csv"
    val_csv = "C:/Users/Markus/Desktop/splits_mm_serbia/val.csv"
    test_csv = "C:/Users/Markus/Desktop/splits_mm_serbia/test.csv"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using {} device".format(device))

    train_dataGen = dataGenBigEarthLMDB_joint(
        bigEarthPthLMDB_S2="C:/Users/Markus/Desktop/project/data/BigEarth_Serbia_Summer_S2.lmdb",
        bigEarthPthLMDB_S1="C:/Users/Markus/Desktop/project/data/BigEarth_Serbia_Summer_S1.lmdb",
        state='val',
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv
    )
    train_data_loader = DataLoader(train_dataGen, batch_size=128, num_workers=0, shuffle=True, pin_memory=False)
    net = get_model(path_type="joint", n_features=2048, projection_dim=128, out_channels=32)
    check = torch.load('C:/Users/Markus/Desktop/project/logs/MM_paper/fourth_try/check_model_best.pth.tar',map_location=torch.device('cpu'))
    net.load_state_dict(check["state_dict"])


    #net = ResNet50_bands_12()
    image = next(iter(train_data_loader))
    #inputs = image["bands"]
    print(image["bands_S1"].size(0))
    inputs_s1 = image["bands_S1"]
    inputs_s2 = image["bands_S2"]
    labels = image['labels']
    h_i, h_j, projection_i, projection_j = net(inputs_s1, inputs_s2)
    cls = ClassificationLoss(projection_dim=128, n_classes=19, fusion="avg")

    #logits = net(inputs)
    #loss = cls(logits,labels)








    simclr = NTxentLoss(0.8,device="cpu")




    loss_tracker = MetricTracker()

    #simclr_loss = simclr(projection_i, projection_j)
    #class_loss = cls(logits,labels)

    #loss_concat = cls(fusion_concat(projection_i, projection_j),labels)
    loss_avg = cls(fusion_avg(projection_i, projection_j), labels)
    loss_sum = cls(fusion_sum(projection_i, projection_j), labels)
    loss_max = cls(fusion_max(projection_i, projection_j), labels)

    loss_tracker.update(loss_avg.item())
    print('Train Loss: {:.6f}'.format(
        loss_tracker.avg
    ))
    print(h_i.shape, h_j.shape, projection_i.shape, projection_j.shape)

    #print(fusion_concat(projection_i, projection_j).shape)
    print(fusion_avg(projection_i, projection_j).shape)
    print(fusion_sum(projection_i, projection_j).shape)
    print(fusion_max(projection_i, projection_j).shape)

    #print(loss_concat)
    print(loss_avg)
    print(loss_sum)
    print(loss_max)
