import torch
import torch.nn as nn
import torchvision.models as models

class TwoBranch(nn.Module):
    def __init__(self,encoder_s1,encoder_s2,projection_dim,n_features):
        super(TwoBranch,self).__init__()
        """
        As encoder the commonly used ResNet50 is adopted to obtain hi = f(xi), because sentinel 1 and sentinel 2 have diferent input channels two different backbones are used
        As projection head a mlp with one hidden layer is used to obtain g(hi) = W_2(sigma(W_1hi) with sigma = ReLU
        """


        self.encoder_s1 = encoder_s1
        self.encoder_s2 = encoder_s2
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
        h_i = self.encoder_s1(s_1)
        h_j = self.encoder_s2(s_2)

        ##flatten to e.g Resnet50 [12,2048] or [2,2048]
        h_i = torch.flatten(h_i,1)
        h_j = torch.flatten(h_j,1)


        ##output dim could be e.g. [128]
        projection_i = self.projection_head(h_i)
        projection_j = self.projection_head(h_j)


        return h_i, h_j, projection_i, projection_j






class ResNet50_S1(nn.Module):
    def __init__(self,numCls = 19):
        super().__init__()

        resnet = models.resnet50(pretrained=False)


        self.conv1 = nn.Conv2d(10, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )




    def forward(self, x):
        x = self.encoder(x)


        return x

class ResNet50_S2(nn.Module):
    def __init__(self,numCls = 19):
        super().__init__()

        resnet = models.resnet50(pretrained=False)


        self.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )




    def forward(self, x):
        x = self.encoder(x)


        return x


if __name__ == "__main__":

    inputs_s1 = torch.randn((4, 10, 120, 120))
    inputs_s2 = torch.randn((4, 2, 120, 120))
    resnet_s1 = ResNet50_S1()
    resnet_s2 = ResNet50_S2()
    net= TwoBranch( resnet_s2,resnet_s1,n_features = 2048, projection_dim= 128)

    h_i, h_j, projection_i, projection_j = net(inputs_s2,inputs_s1)
    print(h_i.shape, h_j.shape, projection_i.shape, projection_j.shape)




