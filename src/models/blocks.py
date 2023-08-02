# -*- coding:utf-8 -*-

from typing import Optional, Callable

from torch import Tensor, nn
from torchvision.models.resnet import conv1x1, conv3x3


class TransposeBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    contraction: float = 0.25

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 2048,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 2048.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if upsample:
            self.conv2 = nn.ConvTranspose2d(width, width, kernel_size=2, stride=stride, bias=False)
        else:
            self.conv2 = nn.ConvTranspose2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, int(planes * self.contraction))
        self.bn3 = norm_layer(int(planes * self.contraction))
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        print(out.shape)
        if self.upsample is not None:
            identity = self.upsample(x)
            print(identity.shape)
        # todo 处理转置卷积维度不一致的问题
        out += identity
        out = self.relu(out)

        return out
