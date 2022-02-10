import torch
from torch import nn
from torchvision import models
import torch.nn.init as init


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data)



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

        self.apply(weights_init_kaiming)


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
        self.apply(weights_init_kaiming)


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
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))#kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):

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

        self.conv_SAR = Conv1(2, out_channels=self.out_channels)
        self.conv_RGB = Conv1(10, out_channels=self.out_channels)
        self.conv_first = nn.Conv2d(self.out_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                    bias=False)
        self.encoder = nn.Sequential(

            self.conv_first,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.apply(weights_init_kaiming)
    def forward(self, x):
        input_channels = x.shape[1]

        if input_channels == 2:
            x = self.conv_SAR(x)  # 1x1 convolution to adjust channel size

        elif input_channels == 10:
            x = self.conv_RGB(x)

        # conv1 = Conv1(input_channels,out_channels =self.out_channels)

        x = self.encoder(x)

        return x

class ResNet50_bands_12(nn.Module):
    """
        ResNet 50 Encoder for 12 bands input
        """
    def __init__(self):
        super(ResNet50_bands_12,self).__init__()

        resnet = models.resnet50(pretrained=False)


        self.conv1 = nn.Conv2d(12, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
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

        self.apply(weights_init_kaiming)
    def forward(self, x):
        x = self.encoder(x)

        ## output should be [batch_size,n_features]
        x = torch.squeeze(x)
        return x