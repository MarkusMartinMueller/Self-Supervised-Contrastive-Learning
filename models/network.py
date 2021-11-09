import torch
import torch.nn as nn
import torchvision
import os

class TwoBranch(nn.Module):
    def __init__(self,encoder,projection_dim,n_features):
        super(TwoBranch,self).__init__()
        """
        As encoder the commonly used ResNet50 is adopted to obtain hi = f(xi) 
        As projection head a mlp with one hidden layer is used to obtain g(hi) = W_2(sigma(W_1hi) with sigma = ReLU
        """


        self.encoder = nn.Sequential(*list(encoder.children())[:-1]) # remove fc layer/classifier from encoder
        self.n_features = n_features





        # add mlp projection head

        self.projection_head = nn.Sequential(nn.Linear(self.n_features,self.n_features,bias = False),
                                             nn.ReLU(),
                                             nn.Linear(self.n_features,projection_dim)


        )

    def forward(self, s_1, s_2):
        """
        s_1: sentinel 1 input tensors
        s_2: sentinel 2 input tensors

        """
        h_i = self.encoder(s_1)
        h_j = self.encoder(s_2)

        ##flatten to e.g Resnet50 [2048]
        h_i = torch.flatten(h_i)
        h_j = torch.flatten(h_j)


        ##output dim could be e.g. [128]
        projection_i = self.projection_head(h_i)
        projection_j = self.projection_head(torch.flatten(h_j)


        return h_i, h_j, projection_i, projection_j





