from functools import partial
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import merge_tb, split_tb


def conv3x3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """kernel size (3, 3, 3) building block for ResNet.
    """
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=(1, stride, stride),
        padding=1,
        bias=False,
    )


def conv1x1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """kernel size (1, 1, 1) building block for ResNet.
    """
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=1, stride=(1, stride, stride), bias=False
    )


class BasicBlock(nn.Module):
    """Basic ResNet building block"""

    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: type,
        layers: Sequence[int],
        block_inplanes: Sequence[int],
        group_norm_groups: int,
        model_width: int,
        n_input_channels: int = 1,
        no_max_pool: bool = True,
        shortcut_type: str = "B",
        widen_factor: float = 1.0,
        stride: int = 1,
    ):
        """ResNet consisting of multiple ResNet building blocks. Also see `generate_resnet()` and `_make_layer()`.
        """
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(
            n_input_channels,
            self.in_planes,
            kernel_size=3,
            stride=(1, stride, stride),
            padding=1,
            bias=False,
        )

        self.gn = nn.GroupNorm(group_norm_groups, model_width)

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            group_norm_groups, block, block_inplanes[0], layers[0], shortcut_type
        )
        self.layer2 = self._make_layer(
            group_norm_groups,
            block,
            block_inplanes[1],
            layers[1],
            shortcut_type,
            stride=stride,
        )
        self.layer3 = self._make_layer(
            group_norm_groups,
            block,
            block_inplanes[2],
            layers[2],
            shortcut_type,
            stride=1,
        )
        self.layer4 = self._make_layer(
            group_norm_groups,
            block,
            block_inplanes[3],
            layers[3],
            shortcut_type,
            stride=1,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(
        self, x: torch.Tensor, planes: int, stride: int
    ) -> Tensor:
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(
            out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)
        )
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(
        self,
        group_norm_groups: int,
        block: type,
        planes: int,
        blocks: int,
        shortcut_type: str,
        stride: int = 1,
    ) -> nn.Sequential:
        """Create one ResNet layer consisting of multiple instances of `block`.

        Parameters
        ----------
        group_norm_groups : int
            groups for nn.GroupNorm.
        block : type
            class to instatiate to create blocks.
        planes : int
            output and intermediate channel count for all blocks.
        blocks : int
            number of blocks to create
        shortcut_type : str
            whether to average pooling (A) or 1x1 convolution (B).
        stride : int, optional
            stride for blocks and downsampling, by default 1

        Returns
        -------
        nn.Sequential
            module for this layer.
        """
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.GroupNorm(group_norm_groups, planes * block.expansion),
                )

        layers = []
        layers.append(
            block(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                downsample=downsample,
            )
        )
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)

        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        out = merge_tb(x)
        out = self.gn(out.float())
        out = split_tb(out, x.shape[0])

        return out


def generate_resnet(
    model_depth: int, model_width: int, group_norm_groups: int, **kwargs
) -> ResNet:
    """Create a ResNet instance. The layers are hard-coded depending on the `model_depth`. See `ResNet` for all parameters.

    Parameters
    ----------
    model_depth : int
        config model.resnet.model_depth
    model_width : int
        config model.model_width, variable C in the paper.
    group_norm_groups : int
        number of groups for GroupNorm

    Returns
    -------
    ResNet
        ResNet model
    """

    layers = {
        10: [1, 1, 1, 1],
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3],
    }

    assert model_depth in layers.keys()

    block_inplanes = [model_width] * 4

    return ResNet(
        BasicBlock,
        layers[model_depth],
        block_inplanes,
        group_norm_groups,
        model_width,
        **kwargs
    )

