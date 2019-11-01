import torch
import torch.nn as nn
from torchvision import models


class ImprovedUNet(nn.Module):
    def __init__(self, args):
        super(ImprovedUNet, self).__init__()

        def ConvReLU(in_channels, out_channels, kernel, padding, stride=1):
            """ create convolution layer with relu """
            layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    padding=padding,
                    stride=stride,
                ),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
            )
            for m in layer:
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
            return layer

        def DeConvReLU(in_channels, out_channels, kernel=4, stride=2, padding=1, output_padding=0):
            """ create convolution layer with relu """
            layer = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                ),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
            )
            for m in layer:
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.xavier_uniform_(m.weight)
            return layer

        backbone = models.resnet18(pretrained=args.pretrained)
        self.backbone_layers = list(backbone.children())

        # top-level features extractor
        self.layer0_1 = ConvReLU(3, 64, 3, 1)
        self.layer0_2 = ConvReLU(64, 64, 3, 1)

        # resnet18 encoder
        self.layer1 = nn.Sequential(*self.backbone_layers[:3])  # [64, 176, 224]
        self.layer2 = nn.Sequential(*self.backbone_layers[3:5])  # [64, 88, 112]
        self.layer3 = self.backbone_layers[5]  # [128, 44, 56]
        self.layer4 = self.backbone_layers[6]  # [256, 22, 28]
        self.layer5 = self.backbone_layers[7]  # [512, 11, 14]
        self.layer6 = ConvReLU(512, 1024, 1, 0, 2)  # [1024, 6, 7]

        # filters
        self.layer6_1x1 = ConvReLU(1024, 1024, 3, 1)
        self.layer5_1x1 = ConvReLU(1024, 256, 3, 1)
        self.layer4_1x1 = ConvReLU(512, 128, 3, 1)
        self.layer3_1x1 = ConvReLU(256, 64, 3, 1)
        self.layer2_1x1 = ConvReLU(128, 64, 3, 1)
        self.layer1_1x1 = ConvReLU(128, 64, 3, 1)
        self.layer0_1x1 = ConvReLU(128, 64, 3, 1)

        # decoder
        self.delayer6 = DeConvReLU(1024, 512, padding=(2, 1), output_padding=(1, 0))
        self.delayer5 = DeConvReLU(256, 256)
        self.delayer4 = DeConvReLU(128, 128)
        self.delayer3 = DeConvReLU(64, 64)
        self.delayer2 = DeConvReLU(64, 64)
        self.delayer1 = DeConvReLU(64, 64)

        self.conv = nn.Conv2d(
            in_channels=64, out_channels=9, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, x):
        layer0 = self.layer0_1(x)
        layer0 = self.layer0_2(layer0)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        layer6 = self.layer6(layer5)

        layer6 = self.layer6_1x1(layer6)
        x = self.delayer6(layer6)
        x = torch.cat([layer5, x], dim=1)

        x = self.layer5_1x1(x)
        x = self.delayer5(x)
        x = torch.cat([layer4, x], dim=1)

        x = self.layer4_1x1(x)
        x = self.delayer4(x)
        x = torch.cat([layer3, x], dim=1)

        x = self.layer3_1x1(x)
        x = self.delayer3(x)
        x = torch.cat([layer2, x], dim=1)

        x = self.layer2_1x1(x)
        x = self.delayer2(x)
        x = torch.cat([layer1, x], dim=1)

        x = self.layer1_1x1(x)
        x = self.delayer1(x)
        x = torch.cat([layer0, x], dim=1)

        x = self.layer0_1x1(x)

        x = self.conv(x)

        return x
