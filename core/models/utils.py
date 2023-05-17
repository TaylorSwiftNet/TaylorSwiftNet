import torch.nn as nn
import torch
from torch import Tensor


def return_activation(activation_str: str) -> nn.Module:
    if activation_str == "lrelu":
        activation = nn.LeakyReLU(0.2, inplace=True)
    elif activation_str == "sin":
        activation = Sine()
    elif activation_str == "relu":
        activation = nn.ReLU()
    else:
        raise Exception(
            "This activation function was not expected: {}".format(activation_str)
        )
    return activation


class Sine(nn.Module):
    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.w0 * x)


def merge_tb(tensor: Tensor) -> Tensor:
    """
    this function receives a tensor with shape BxCxTxHxW and returns a tensor with shape (B*T)xCx1xHxW
    :param tensor:
    :return:
    """

    b, c, t, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1, 3, 4)
    tensor = tensor.contiguous().view(-1, c, 1, h, w)

    return tensor


def split_tb(tensor: Tensor, batch_size: int) -> Tensor:
    """
    this function receives a tensor with shape (B*T)xCx1xHxW and returns a tensor with shape BxCxTxHxW
    :param tensor:
    :param batch_size:
    :return:
    """

    b, c, t, h, w = tensor.shape
    tensor = tensor.view(batch_size, b // batch_size, c, h, w).contiguous()
    tensor = tensor.permute(0, 2, 1, 3, 4)

    return tensor

