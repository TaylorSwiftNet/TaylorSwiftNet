from typing import List, Optional, Union

import torch
from torch import Tensor
import torch.nn as nn
from yacs.config import CfgNode

from core.models.dcgan import Decoder, Encoder
from core.models.resnet import generate_resnet
from core.models.temporal_dynamic import TemporalDynamicModel
from core.models.utils import merge_tb
from core.tools.data_classes import GeneralLoss
from core.tools.utils import find_model_state, init_weights


class GeneralModel(nn.Module):
    def __init__(self, cfg: CfgNode):
        """
        The TaylorSwiftNet model.

        Config
        ----------
        trainer.batch_size
        system.device
        model.width
        model.resnet.model_depth
        model.group_norm_groups
        dataset.unseen_frames
        dataset.seen_frames

        Submodules
        ----------
        TemporalDynamicModel
        """

        super(GeneralModel, self).__init__()
        self.cfg = cfg
        self.batch_size = cfg.trainer.batch_size
        self.device = torch.device(cfg.system.device)
        self.eps = 1e-8

        self.mse_loss = nn.MSELoss(reduction="none")
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.encoder = Encoder(cfg.model.width)
        self.encoder.apply(init_weights)

        self.decoder = Decoder(cfg.model.width)
        self.decoder.apply(init_weights)

        self.resnet = generate_resnet(
            cfg.model.resnet.model_depth,
            cfg.model.width,
            cfg.model.group_norm_groups,
            n_input_channels=self.cfg.model.width,
        )

        self.unseen_frames = cfg.dataset.unseen_frames
        self.seen_frames = cfg.dataset.seen_frames

        self.temporal_dynamic = TemporalDynamicModel(cfg)
        self.temporal_dynamic.apply(init_weights)

    def skip_connection_handler(self, skip: List[Tensor], t_dim: int, t=None):
        t_sq_skip = []

        assert t is None

        for feature in skip:
            feature = torch.mean(feature, dim=2).unsqueeze(2).repeat(1, 1, t_dim, 1, 1)
            feature = merge_tb(feature)
            t_sq_skip.append(feature)

        return t_sq_skip

    def forward(self, _input: Tensor, t, sampler_indices=None) -> List:
        skip = None
        distribution = None

        encoded = self.encoder(_input)
        current_hidden_state, skip = encoded
        current_hidden_state = self.resnet(current_hidden_state)

        future_temporal_hidden = self.temporal_dynamic(
            current_hidden_state, sampler_indices=sampler_indices
        )

        decoder_input = future_temporal_hidden

        decoded = self.decoder([decoder_input, skip])
        return [decoded, distribution]

    def loss(
        self,
        output: List[Tensor],
        target: Tensor,
        t,
        loss_weights: Optional[Tensor] = None,
    ) -> GeneralLoss:
        moment_loss = None
        kl_loss = None

        predictions, distribution = output

        batch_size = predictions.shape[0]
        mse_loss = self.mse_loss(predictions, target)
        mse_loss = torch.mean(mse_loss.reshape(batch_size, -1), dim=1).sum()

        if loss_weights is None:
            loss_weights = torch.Tensor(
                [1 / self.unseen_frames] * self.unseen_frames
            ).to(self.device)

        total_loss = (
            loss_weights.reshape(1, -1).expand(batch_size, -1)
            * self.mse_loss(predictions, target)
            .reshape(batch_size, self.unseen_frames, -1)
            .mean(dim=2)
        ).sum()

        total_loss_weight = batch_size

        return GeneralLoss(
            total_loss=dict(loss=total_loss, weight=total_loss_weight),
            mse_loss=mse_loss,
            temporal_mse=None,
            moment_loss=moment_loss,
            kl_loss=kl_loss,
        )

    @staticmethod
    def kl_criterion(mu, logvar):
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # batch_size = mu.shape[0]
        # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # KLD /= batch_size
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0
        ).sum()

        return kld_loss


def make_general_model(cfg: CfgNode) -> Union[GeneralModel, nn.DataParallel]:
    """
    Factory for creating a GeneralModel instance and optionally resuming a run.

    Config
    ------
    model.resume
    system.device

    Submodules
    -------
    GeneralModel
    find_model_state
    """
    # Autoregressive was not enabled
    model = GeneralModel(cfg)

    if torch.cuda.device_count() > 1:
        print("Created parallel model")
        model = nn.DataParallel(model)

    if cfg.model.resume:
        if cfg.system.device == "cpu":
            state_dict = torch.load(
                find_model_state(cfg), map_location=torch.device("cpu")
            )
        else:
            state_dict = torch.load(find_model_state(cfg))

        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        else:
            state_dict = state_dict["model"].state_dict()

        clean_state_dict = {}

        # Remove artifacts from multi-gpu support
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module.") :]

            clean_state_dict[k] = v

        model.load_state_dict(clean_state_dict)
        print(f"The state dict {find_model_state(cfg)} has been loaded.")

    model = model.to(torch.device(cfg.system.device))
    return model


def test_model_parameters(cfg: CfgNode):
    from core.tools.utils import count_parameters

    model = GeneralModel(cfg)
    print("\n")
    print(model)
    print("\n")

    print(f"model parameters: {count_parameters(model) / 1e6:.2f} M")


if __name__ == "__main__":
    from core.tools.utils import create_cfg

    cfg = create_cfg()

    test_model_parameters(cfg)
