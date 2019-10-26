import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BaselineNet(nn.Module):
    def __init__(self, args):
        super(BaselineNet, self).__init__()

        """ backbone """
        backbone = models.resnet18(pretrained=args.pretrained)
        self.backbone = nn.Sequential(*(list(backbone.children())[:-2]))

        """ De-convolution layers"""
        # first block
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        nn.init.xavier_uniform_(self.deconv1.weight)

        # second block
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        nn.init.xavier_uniform_(self.deconv2.weight)

        # third block
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        nn.init.xavier_uniform_(self.deconv3.weight)

        # fourth block
        self.deconv4 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        nn.init.xavier_uniform_(self.deconv4.weight)

        # fifth block
        self.deconv5 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        nn.init.xavier_uniform_(self.deconv5.weight)

        """ Convolution classifier """
        # sixth block
        self.conv6 = nn.Conv2d(
            in_channels=16,
            out_channels=9,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        nn.init.xavier_uniform_(self.conv6.weight)

    def forward(self, imgs):

        x = self.backbone(imgs)

        x = self.deconv1(x)
        x = F.relu(x)

        x = self.deconv2(x)
        x = F.relu(x)

        x = self.deconv3(x)
        x = F.relu(x)

        x = self.deconv4(x)
        x = F.relu(x)

        x = self.deconv5(x)
        x = F.relu(x)

        x = self.conv6(x)

        return x
