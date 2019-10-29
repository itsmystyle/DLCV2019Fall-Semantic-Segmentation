import torch
import torch.nn as nn
from torchvision import models


class UNet(nn.Module):
    def __init__(self, args):
        super(UNet, self).__init__()

        def ConvReLU(in_channels, out_channels, kernel, padding):
            """ create convolution layer with relu """
            return nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    padding=padding,
                ),
                nn.ReLU(inplace=True),
            )

        def DeConvReLU(in_channels, out_channels, kernel=4, stride=2, padding=1):
            """ create convolution layer with relu """
            return nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.ReLU(inplace=True),
            )

        backbone = models.resnet18(pretrained=args.pretrained)
        self.backbone_layers = list(backbone.children())

        self.conv1 = nn.Sequential(*self.backbone_layers[:3])  # [32, 64, 176, 224]
        self.conv2 = nn.Sequential(*self.backbone_layers[3:5])  # [32, 64, 88, 112]
        self.conv3 = self.backbone_layers[5]  # [32, 128, 44, 56]
        self.conv4 = self.backbone_layers[6]  # [32, 256, 22, 28]
        self.conv5 = self.backbone_layers[7]  # [32, 512, 11, 14]

        self.deconv4 = DeConvReLU(512, 256)
        self.deconv3 = DeConvReLU(256, 128)
        self.deconv2 = DeConvReLU(128, 64)
        self.deconv1 = DeConvReLU(64, 64)
        self.deconvf = DeConvReLU(64, 32)

        self.conv4_1 = ConvReLU(512, 256, 1, 0)
        self.conv3_1 = ConvReLU(256, 128, 1, 0)
        self.conv2_1 = ConvReLU(128, 64, 1, 0)
        self.conv1_1 = ConvReLU(128, 64, 1, 0)

        self.conv = nn.Conv2d(
            in_channels=32, out_channels=9, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, x):
        layer1 = self.conv1(x)
        layer2 = self.conv2(layer1)
        layer3 = self.conv3(layer2)
        layer4 = self.conv4(layer3)
        layer5 = self.conv5(layer4)

        d_layer4 = self.deconv4(layer5)
        d_layer4 = torch.cat((layer4, d_layer4), dim=1)
        d_layer4 = self.conv4_1(d_layer4)

        d_layer3 = self.deconv3(d_layer4)
        d_layer3 = torch.cat((layer3, d_layer3), dim=1)
        d_layer3 = self.conv3_1(d_layer3)

        d_layer2 = self.deconv2(d_layer3)
        d_layer2 = torch.cat((layer2, d_layer2), dim=1)
        d_layer2 = self.conv2_1(d_layer2)

        d_layer1 = self.deconv1(d_layer2)
        d_layer1 = torch.cat((layer1, d_layer1), dim=1)
        d_layer1 = self.conv1_1(d_layer1)

        x = self.deconvf(d_layer1)
        x = self.conv(x)

        return x
