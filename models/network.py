"""
  File: network.py
  Authors: Markus Mueller (m.markus.mueler@campus.tu-berlin.de)
  Created: 202-11-09
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader



#local imports
from ResNet import ResNet50_S1, ResNet50_S2, ResNet50_joint
from data.data_one_way import dataGenBigEarthLMDB_joint
from data.data_two_way import dataGenBigEarthLMDB
from utils.fusion import fusion_concat,fusion_avg,fusion_sum,fusion_max


class TwoBranch(nn.Module):

    def __init__(self, encoder_s1, encoder_s2, encoder_joint, projection_dim, n_features,type):
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
                                             nn.Linear(self.n_features, projection_dim)
                                                )

        self.projection_head_s2 = nn.Sequential(
                                                nn.Linear(self.n_features, projection_dim)

                                                )

    def forward(self, s_1, s_2):
        """
        s_1: torch.tensor, sentinel 1 input [batch_size,bands,60,60]
        s_2: torch.tensor, sentinel 2 input [batch_size,bands,120,120]
        type: string separate or joint

        separate: two different models for each modality
        joint: same model with a preceding 1x1 conv to get the same number of channels

        """

        if self.type == "separate":
            return self.forward_separate(s_1, s_2)
        if self.type == "joint":
            return self.forward_joint(s_1, s_2)


    def forward_separate(self,s_1,s_2):

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







if __name__ == "__main__":


    train_csv = "C:/Users/Markus/Desktop/splits_mm_serbia/train.csv"
    val_csv = "C:/Users/Markus/Desktop/splits_mm_serbia/val.csv"
    test_csv = "C:/Users/Markus/Desktop/splits_mm_serbia/test.csv"

    train_dataGen = dataGenBigEarthLMDB_joint(
        bigEarthPthLMDB_S2="C:/Users/Markus/Desktop/project/data/BigEarth_Serbia_Summer_S2.lmdb",
        bigEarthPthLMDB_S1="C:/Users/Markus/Desktop/project/data/BigEarth_Serbia_Summer_S1.lmdb",
        state='val',

        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv
    )
    train_data_loader = DataLoader(train_dataGen, batch_size=4, num_workers=0, shuffle=False, pin_memory=False)

    image = next(iter(train_data_loader))

    #inputs_s1 = torch.randn((4, 10, 224, 224))
    #inputs_s2 = torch.randn((4, 2, 224, 224))

    inputs_s1 = image["bands_S1"]
    inputs_s2 = image["bands_S2"]

    resnet_s1 = ResNet50_S1()
    resnet_s2 = ResNet50_S2()
    resnet_joint = ResNet50_joint()
    net = TwoBranch(resnet_s2, resnet_s1, resnet_joint, n_features=2048, projection_dim=128,type= "joint")

    h_i, h_j, projection_i, projection_j = net(inputs_s2, inputs_s1)
    print(h_i.shape, h_j.shape, projection_i.shape, projection_j.shape)

    print(fusion_concat(projection_i, projection_j).shape)
    print(fusion_avg(projection_i, projection_j).shape)
    print(fusion_sum(projection_i, projection_j).shape)
    print(fusion_max(projection_i, projection_j).shape)




