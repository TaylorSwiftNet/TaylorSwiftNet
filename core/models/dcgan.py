from typing import Tuple
from torch import Tensor
import torch.nn as nn
from einops import rearrange

from .utils import Sine


class Encoder(nn.Module):
    def __init__(self, model_width: int):
        """Convolutional network encoder.

        Parameters
        ----------
        model_width : int
            parameter C in the paper
        """
        super(Encoder, self).__init__()

        in_ = [1, 8, 16, 32, 32, 64]
        out_ = [8, 16, 32, 32, 64, model_width]
        stride = [1, 1, 2, 1, 2, 1]

        modules = []
        for i in range(6):
            modules.append(
                nn.Conv3d(
                    in_channels=in_[i],
                    out_channels=out_[i],
                    kernel_size=(1, 3, 3),
                    stride=(1, stride[i], stride[i]),
                    padding=(0, 1, 1),
                    bias=False,
                )
            )
            modules.append(nn.GroupNorm(4, out_[i]))
            modules.append(Sine() if i < 5 else nn.Tanh())

        self.main = nn.Sequential(*modules)

    def forward(self, _input: Tensor) -> Tuple[Tensor, list]:
        return self.main(_input), []  # do not use skip features


class Decoder(nn.Module):
    def __init__(self, model_width: int):
        """Convolutional network decoder.

        Parameters
        ----------
        model_width : int
            parameter C in the paper
        """
        super(Decoder, self).__init__()

        in_ = [model_width, 64, 32, 32, 16, 8]
        out_ = [64, 32, 32, 16, 8, 1]
        stride = [1, 2, 1, 2, 1, 1]

        modules = []
        for i in range(6):
            modules.append(
                nn.ConvTranspose3d(
                    in_channels=in_[i],
                    out_channels=out_[i],
                    kernel_size=(1, 3, 3),
                    stride=(1, stride[i], stride[i]),
                    padding=(0, 1, 1),
                    output_padding=(0, stride[i] - 1, stride[i] - 1),
                    bias=False,
                )
            )
            modules.append(nn.GroupNorm((4 if i < 5 else 1), out_[i]))
            modules.append(Sine() if i < 5 else nn.Tanh())

        self.main = nn.Sequential(*modules)

    def forward(self, _input: Tuple[Tensor, list]) -> Tensor:
        x, skip = _input  # we do not use skip features
        batch_size = x.shape[0]

        x = rearrange(x, "b c t h w -> (b t) c 1 h w")
        x = self.main(x)
        x = rearrange(x, "(b t) c 1 h w -> b c t h w", b=batch_size)

        return x
