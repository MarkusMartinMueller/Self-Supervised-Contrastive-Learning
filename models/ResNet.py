import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50_S2(nn.Module):
    """
    ResNet 50 Encoder for Sentinel 2 tensors
    """
    def __init__(self):
        super(ResNet50_S2, self).__init__()

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

class ResNet50_S1(nn.Module):
    """
        ResNet 50 Encoder for Sentinel 1 tensors
        """
    def __init__(self):
        super(ResNet50_S1,self).__init__()

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



class Conv1(nn.Module):
    """
        1x1 Convolution for the joint model to adjust the channel number to equal channel across modalitites
        e.g. SAR image batch with torch tensor (batch_size = 200, channels = 2, 120,120) to (batch_size = 200, channels = 32, 120,120)
    """

    def __init__(self, in_channels,out_channels=32):
        super(Conv1,self).__init__()



        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self,x):

        x = self.conv(x)

        return x



class ResNet50_joint(nn.Module):
    """
    Joint Encoder for both modalities
    """

    def __init__(self, out_channels=32):
        super(ResNet50_joint, self).__init__()

        resnet = models.resnet50(pretrained=False)
        self.out_channels = out_channels


        self.conv1 = nn.Conv2d(self.out_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
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

        input_channels = x.shape[1]

        #if in_channels != self.out_channels:
            #conv0 = nn.Conv2d(in_channels, self.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            #x = conv0(x)



        conv1 = Conv1(input_channels,out_channels =32)


        x = conv1(x)   # 1x1 convolution to adjust channel size
        x = self.encoder(x)

        return x