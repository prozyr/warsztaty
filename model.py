import math
import os
from typing import Any, cast, Dict, List, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F_torch
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor

""" SRResNet """
class SRResNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
            num_rcb: int = 16,
            upscale: int = 4,
    ) -> None:
        super(SRResNet, self).__init__()
        # Low frequency information extraction layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, (9, 9), (1, 1), (4, 4)),
            nn.PReLU(),
        )

        # High frequency information extraction block
        trunk = []
        for _ in range(num_rcb):
            trunk.append(_ResidualConvBlock(channels))
        self.trunk = nn.Sequential(*trunk)

        # High-frequency information linear fusion layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

        # zoom block
        upsampling = []
        if upscale == 2 or upscale == 4 or upscale == 8:
            for _ in range(int(math.log(upscale, 2))):
                upsampling.append(_UpsampleBlock(channels, 2))
        else:
            raise NotImplementedError(f"Upscale factor `{upscale}` is not support.")
        self.upsampling = nn.Sequential(*upsampling)

        # reconstruction block
        self.conv3 = nn.Conv2d(channels, out_channels, (9, 9), (1, 1), (4, 4))

        # Initialize neural network weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1)
        x = self.upsampling(x)
        x = self.conv3(x)

        x = torch.clamp_(x, 0.0, 1.0)

        return x

class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(_ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        x = self.rcb(x)

        x = torch.add(x, identity)

        return x
    
class _UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(_UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample_block(x)

        return x
    

""" RDDBNet """""
class RRDBNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
            growth_channels: int = 32,
            num_rrdb: int = 23,
            upscale: int = 4,
    ) -> None:
        super(RRDBNet, self).__init__()
        self.upscale = upscale

        # The first layer of convolutional layer.
        self.conv1 = nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1))

        # Feature extraction backbone network.
        trunk = []
        for _ in range(num_rrdb):
            trunk.append(_ResidualResidualDenseBlock(channels, growth_channels))
        self.trunk = nn.Sequential(*trunk)

        # After the feature extraction network, reconnect a layer of convolutional blocks.
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))

        # Upsampling convolutional layer.
        if upscale == 2:
            self.upsampling1 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True)
            )
        if upscale == 4:
            self.upsampling1 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True)
            )
            self.upsampling2 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True)
            )
        if upscale == 8:
            self.upsampling1 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True)
            )
            self.upsampling2 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True)
            )
            self.upsampling3 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True)
            )

        # Reconnect a layer of convolution block after upsampling.
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        # Output layer.
        self.conv4 = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))

        # Initialize all layer
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.2
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    # The model should be defined in the Torch.script method.
    def _forward_impl(self, x: Tensor) -> Tensor:
        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1)

        if self.upscale == 2:
            x = self.upsampling1(F_torch.interpolate(x, scale_factor=2, mode="nearest"))
        if self.upscale == 4:
            x = self.upsampling1(F_torch.interpolate(x, scale_factor=2, mode="nearest"))
            x = self.upsampling2(F_torch.interpolate(x, scale_factor=2, mode="nearest"))
        if self.upscale == 8:
            x = self.upsampling1(F_torch.interpolate(x, scale_factor=2, mode="nearest"))
            x = self.upsampling2(F_torch.interpolate(x, scale_factor=2, mode="nearest"))
            x = self.upsampling3(F_torch.interpolate(x, scale_factor=2, mode="nearest"))

        x = self.conv3(x)
        x = self.conv4(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

class _ResidualDenseBlock(nn.Module):
    """Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(_ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels + growth_channels * 0, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(channels + growth_channels * 1, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(channels + growth_channels * 2, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(channels + growth_channels * 3, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(channels + growth_channels * 4, channels, (3, 3), (1, 1), (1, 1))

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1], 1)))
        out3 = self.leaky_relu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = self.leaky_relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out5 = self.identity(self.conv5(torch.cat([x, out1, out2, out3, out4], 1)))

        x = torch.mul(out5, 0.2)
        x = torch.add(x, identity)

        return x

""" CycleNet """
class CycleNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: int,
    ) -> None:
        super(CycleNet, self).__init__()
        self.main = nn.Sequential(
            # Initial convolution block
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, channels, (7, 7), (1, 1), (0, 0)),
            nn.InstanceNorm2d(channels, track_running_stats=True),
            nn.ReLU(True),

            # Downsampling
            nn.Conv2d(channels, int(channels * 2), (3, 3), (2, 2), (1, 1)),
            nn.InstanceNorm2d(int(channels * 2), track_running_stats=True),
            nn.ReLU(True),
            nn.Conv2d(int(channels * 2), int(channels * 4), (3, 3), (2, 2), (1, 1)),
            nn.InstanceNorm2d(int(channels * 4), track_running_stats=True),
            nn.ReLU(True),

            # Residual blocks
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),
            _ResidualBlock(int(channels * 4)),

            # Upsampling
            nn.ConvTranspose2d(int(channels * 4), int(channels * 2), (3, 3), (2, 2), (1, 1), (1, 1)),
            nn.InstanceNorm2d(int(channels * 2), track_running_stats=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(int(channels * 2), channels, (3, 3), (2, 2), (1, 1), (1, 1)),
            nn.InstanceNorm2d(channels, track_running_stats=True),
            nn.ReLU(True),

            # Output layer
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_channels, (7, 7), (1, 1), (0, 0)),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.main(x)

        return x

class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super(_ResidualBlock, self).__init__()

        self.res = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (0, 0)),
            nn.InstanceNorm2d(channels, track_running_stats=True),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (0, 0)),
            nn.InstanceNorm2d(channels, track_running_stats=True),
        )

""" SRCNN """

""" FSRCNN """

