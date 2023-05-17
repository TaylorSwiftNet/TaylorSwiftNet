import torch
import torch.nn as nn
from torch import Tensor
from scipy.special import factorial
from yacs.config import CfgNode

from core.models.dcb import DCB
from core.models.utils import merge_tb, split_tb


class TemporalDynamicModel(nn.Module):
    def __init__(self, cfg: CfgNode):
        """
        Implementation of the TemporalModel.

        Config
        ----------
        dataset.seen_frames 
        dataset.unseen_frames 
        system.device 
        model.taylor_order_r 
        model.group_norm_groups 
        model.width 

        Submodules
        ----------
        DCB
        """

        super(TemporalDynamicModel, self).__init__()
        self.cfg = cfg
        self.seen_frames = cfg.dataset.seen_frames

        self.unseen_frames = cfg.dataset.unseen_frames

        self.device = torch.device(cfg.system.device)

        self.taylor_order = cfg.model.taylor_order_r
        self.tpd = DCB(cfg=cfg, order=self.taylor_order)

        self.gn = nn.GroupNorm(
            num_groups=cfg.model.group_norm_groups, num_channels=cfg.model.width,
        ).to(self.device)

        taylor_index = nn.Parameter(
            torch.arange(self.taylor_order).to(torch.float64).to(self.device) + 1,
            requires_grad=False,
        )
        t_s = nn.Parameter(
            torch.arange(self.seen_frames).to(torch.float64).to(self.device) + 1,
            requires_grad=False,
        )
        t_u = nn.Parameter(
            torch.arange(self.unseen_frames).to(torch.float64).to(self.device)
            + self.seen_frames
            + 1,
            requires_grad=False,
        )

        taylor_index = taylor_index.reshape(1, self.taylor_order, 1, 1, 1, 1)
        t_s = self.seen_frames
        t_u = t_u.reshape(1, 1, 1, self.unseen_frames, 1, 1)

        taylor_factorial = factorial(taylor_index.cpu()).to(self.device)
        self.taylor_coef = (1 / taylor_factorial) * (t_u - t_s) ** taylor_index
        self.taylor_coef = self.taylor_coef.to(torch.float32)
        self.t_u = t_u

    def forward(self, _input: Tensor, sampler_indices=None) -> Tensor:
        last_frame = _input[:, :, -1, :, :][:, :, None, :, :]

        out = torch.cat(self.tpd(_input), dim=1)

        coefs = self.taylor_coef

        out = out * coefs.to(out.device)
        out = torch.sum(out, dim=1)
        out += last_frame

        out = merge_tb(out)
        out = self.gn(out.float())
        out = split_tb(out, _input.shape[0])

        return out
