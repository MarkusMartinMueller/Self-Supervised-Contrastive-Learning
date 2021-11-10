"""
  File: network.py
  Authors: Markus Mueller (m.markus.mueler@campus.tu-berlin.de)
  Created: 202-11-09
"""



import torch
import torch.nn as nn
import torchvision.models as models
from ResNet import ResNet50_S1, ResNet50_S2, ResNet50_joint

class TwoBranch(nn.Module):
    def __init__(self,encoder_s1,encoder_s2,encoder_joint,projection_dim,n_features):
        super(TwoBranch,self).__init__()
        """
        As encoder the commonly used ResNet50 is adopted to obtain hi = f(xi)
        As projection head a mlp with one hidden layer is used to obtain g(hi) = W_2(sigma(W_1hi) with sigma = ReLU
        """


        self.encoder_s1 = encoder_s1
        self.encoder_s2 = encoder_s2
        self.encoder_joint = encoder_joint
        self.n_features = n_features

        # add mlp projection head

        self.projection_head_s1 = nn.Sequential(nn.Linear(self.n_features,self.n_features,bias = False),
                                             nn.ReLU(),
                                             nn.Linear(self.n_features,projection_dim)
                                                )

        self.projection_head_s2 = nn.Sequential(nn.Linear(self.n_features, self.n_features, bias=False),
                                                nn.ReLU(),
                                                nn.Linear(self.n_features, projection_dim)

                                                )

    def forward(self, s_1, s_2, type = "joint"):
        """
        s_1: torch.tensor, sentinel 1 input
        s_2: torch.tensor, sentinel 2 input
        type: string separate or joint

        separate: two different models for each modality
        joint: same model with a preceding 1x1 conv to get the same number of channels

        """

        if type == "separate":
            return self.forward_separate(s_1,s_2)
        if type == "joint":
            return self.forward_joint(s_1, s_2)


    def forward_separate(self,s_1,s2):

        h_i = self.encoder_s1(s_1)
        h_j = self.encoder_s2(s_2)

        ##flatten to e.g Resnet50 [batch_size,2048]
        h_i = torch.flatten(h_i, 1)
        h_j = torch.flatten(h_j, 1)

        ##output dim could be e.g. [128]
        projection_i = self.projection_head_s1(h_i)
        projection_j = self.projection_head_s2(h_j)

        return h_i, h_j, projection_i, projection_j

    def forward_joint(self, s_1, s_2):

        h_i = self.encoder_joint(s_1)
        h_j = self.encoder_joint(s_2)

        ##flatten to e.g Resnet50 [batch_size,2048]
        h_i = torch.flatten(h_i, 1)
        h_j = torch.flatten(h_j, 1)

        ##output dim could be e.g. [128]
        projection_i = self.projection_head_s1(h_i)
        projection_j = self.projection_head_s2(h_j)

        return h_i, h_j, projection_i, projection_j







if __name__ == "__main__":

    inputs_s1 = torch.randn((4, 10, 120, 120))
    inputs_s2 = torch.randn((4, 2, 120, 120))
    resnet_s1 = ResNet50_S1()
    resnet_s2 = ResNet50_S2()
    resnet_joint = ResNet50_joint()
    net= TwoBranch( resnet_s2,resnet_s1,resnet_joint,n_features = 2048, projection_dim= 128)

    h_i, h_j, projection_i, projection_j = net(inputs_s2,inputs_s1)
    print(h_i.shape, h_j.shape, projection_i.shape, projection_j.shape)




