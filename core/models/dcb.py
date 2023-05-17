from typing import List
import torch
import torch.nn as nn
from torch import Tensor
from yacs.config import CfgNode

from core.models.utils import return_activation


class DCB(nn.Module):
    def __init__(self, cfg: CfgNode, order: int):
        """
        Implementation of the Delta Convolutional Block for TemporalDynamic.

        Config
        ----------
        model.width
        dataset.seen_frames
        model.conv_3d_depth
        model.activation_
        """
        super(DCB, self).__init__()
        self.in_channels = cfg.model.width
        self.temporal_dim = cfg.dataset.seen_frames
        self.cfg = cfg
        self.order = order
        self.conv_1_depth = cfg.model.conv_3d_depth

        self.conv_group = 1

        activation = return_activation(cfg.model.activation_)

        kernel_size = (3, 3, 3)
        padding = (1, 1, 1)

        conv_1_dict = {
            "in_channels": self.in_channels,
            "out_channels": self.in_channels,
            "kernel_size": kernel_size,
            "stride": 1,
            "padding": padding,
            "groups": self.conv_group,
        }

        kernel_size = (self.temporal_dim, 3, 3)
        padding = (0, 1, 1)

        conv_2_dict = {
            "in_channels": self.in_channels,
            "out_channels": self.in_channels,
            "kernel_size": kernel_size,
            "stride": 1,
            "padding": padding,
            "groups": self.conv_group,
        }

        self.dcb_blocks = nn.ModuleDict()

        for order_idx in range(self.order):
            dcb_block_list = []

            # handling temporal preserving convolutions
            for _ in range(self.conv_1_depth):
                dcb_block_list.append(nn.Conv3d(**conv_1_dict))
                dcb_block_list.append(activation)

            self.dcb_blocks.update(
                {f"block_{order_idx}_conv_1": nn.Sequential(*dcb_block_list)}
            )

            dcb_block_list = []

            # handling temporal squeezing convolution
            dcb_block_list.append(nn.Conv3d(**conv_2_dict))
            dcb_block_list.append(activation)

            self.dcb_blocks.update(
                {f"block_{order_idx}_conv_2": nn.Sequential(*dcb_block_list)}
            )

    def forward(self, _input: Tensor) -> List[Tensor]:
        derivatives = []
        for order_idx in range(self.order):

            out_conv_1 = self.dcb_blocks[f"block_{order_idx}_conv_1"](_input)
            derivative = self.dcb_blocks[f"block_{order_idx}_conv_2"](
                out_conv_1
            ).unsqueeze(1)
            derivatives.append(derivative)

            _input = _input + out_conv_1

        return derivatives


def test_dcb_parameters(cfg: CfgNode):
    from torchsummary import summary

    from core.tools.utils import count_parameters

    dcb = DCB(cfg, 4).to(torch.device("cuda"))
    summary(
        dcb,
        input_size=(
            cfg.model.width,
            cfg.dataset.seen_frames,
            cfg.dataset.encoded_image_size,
            cfg.dataset.encoded_image_size,
        ),
    )

    print("\n")
    print(dcb)
    print("\n")

    print(f"model parameters: {count_parameters(dcb) / 1e6:.2f} M")


def test_dcb_profiling(cfg: CfgNode):
    import time

    from tqdm import tqdm

    device = torch.device("cuda")

    dcb = DCB(cfg, 4).to(device)

    _input = torch.rand(
        1,
        cfg.model.width,
        cfg.dataset.seen_frames,
        cfg.dataset.encoded_image_size,
        cfg.dataset.encoded_image_size,
    ).to(device)

    times = []
    for i in tqdm(range(100)):
        start = time.time()
        _ = dcb(_input)
        end = time.time()
        times.append(end - start)

    average_time = sum(times) / len(times)
    print(f"\ndcb profiling: {average_time * 1000:.2f} ms\n")


if __name__ == "__main__":
    from core.tools.utils import create_cfg

    cfg = create_cfg()

    test_dcb_profiling(cfg)

