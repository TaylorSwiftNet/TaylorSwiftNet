import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim_metric
from torch import Tensor
from torch.utils.data import Dataset
from yacs.config import CfgNode

from core.tools.data_classes import GeneralLoss, Metrics


class GeneralDataset(Dataset):
    def __init__(self, cfg: CfgNode):
        """
        Superclass for different datasets. Also implements general logic for performing a forward
        pass and evaluating, which may be overwritten by a subclass.

        Config
        ------
        dataset.unseen_frames
        system.device
        dataset.reinforce_memory
        """
        super(GeneralDataset, self).__init__()
        self.bbg_config_list = None
        self.unseen_frames = cfg.dataset.unseen_frames
        self.device = torch.device(cfg.system.device)
        self.reinforce_memory = cfg.dataset.reinforce_memory
        self.length = None

    def __len__(self):
        pass

    def __getitem__(self, item: int):
        pass

    @staticmethod
    @torch.no_grad()
    def evaluate(prediction: Tensor, target: Tensor) -> Dict[str, Any]:
        """Calculate relevant validation metrics: MSE, MAE, SSIM, TemporalSSIM

        Parameters
        ----------
        prediction : Tensor
            prediction
        target : Tensor
            ground truth

        Returns
        -------
        dict
            calculated metrics
        """

        # we will divide the batch size at the end
        b, c, t, h, w = prediction.shape

        mse_batch = torch.mean(
            torch.pow(prediction - target, 2).reshape(-1, h, w), dim=0
        ).sum()
        mae_batch = torch.mean(
            torch.abs(prediction - target).reshape(-1, h, w), dim=0
        ).sum()

        prediction = prediction.detach().cpu().numpy().astype(np.float32)
        target = target.detach().cpu().numpy().astype(np.float32)

        temporal_ssim = [[] for _ in range(t)]

        # calculating structural similarity index
        ssim_batch = 0
        for i in range(b):
            for j in range(c):
                for k in range(t):
                    ssim_batch += ssim_metric(target[i, j, k], prediction[i, j, k])
                    temporal_ssim[k] += [
                        ssim_metric(target[i, j, k], prediction[i, j, k])
                    ]

        ssim_batch = ssim_batch / (c * t)

        # calculating binary cross entropy
        return dict(
            mse_metric=mse_batch,
            mae_metric=mae_batch,
            ssim_metric=ssim_batch,
            temporal_ssim=temporal_ssim,
        )

    def prepare_data(self, batch_: List[Tensor]):
        batch, target = batch_[:2]
        batch = batch.to(self.device)
        target = target.to(self.device)

        t = (
            (torch.arange(self.unseen_frames) + 1)
            .to(self.device)
            .reshape(1, 1, -1, 1, 1)
        )

        t = t.repeat(batch.shape[0], 1, 1, 1, 1)
        return batch, target, t

    def feed_forward(
        self,
        batch_: List[Tensor],
        model: "GeneralModel",
        metrics: Optional[Metrics] = None,
        parallel_model: Optional[nn.DataParallel] = None,
        loss_weights: Optional[Tensor] = None,
        no_loss: bool = False,
    ) -> Tuple[List[Tensor], Optional[GeneralLoss]]:
        """Perform a forward pass through `model` using `batch_` as model input. This is a function
        of the dataset to allow for dataset-specific validation metrices and data preparation.

        Parameters
        ----------
        batch_ : List[Tensor]
            training data batch
        model : GeneralModel
            model to forward through
        metrics : Optional[Metrics], optional
            object for metric accumulation, by default None
        parallel_model : Optional[nn.DataParallel], optional
            parallel model for multi-gpu training, by default None
        loss_weights : Optional[Tensor], optional
            weights for loss function, by default None
        no_loss : bool, optional
            whether to calculate the loss, by default False

        Returns
        -------
        Tuple[List[Tensor], Optional[GeneralLoss]]
            model prediction and loss if requested
        """

        if parallel_model is None:
            parallel_model = model

        batch, target, t = self.prepare_data(batch_)

        out = model(batch, t)

        if not no_loss:
            loss = parallel_model.loss(out, target, t, loss_weights)  # type: ignore
        else:
            loss = None

        if metrics is not None:
            predictions = out[0]

            eval_metrics = self.evaluate(predictions, target)
            metrics.append([loss.__dict__, eval_metrics])
        return out, loss

    def generate_a_random_config(self):
        return []

    def set_config_list(self):
        if self.bbg_config_list is not None:
            random.shuffle(self.bbg_config_list)
            length_of_seen_samples = int(
                len(self.bbg_config_list) * self.reinforce_memory
            )
            self.bbg_config_list = self.bbg_config_list[:length_of_seen_samples]
            length_of_list = self.length - length_of_seen_samples
        else:
            length_of_list = self.length
            self.bbg_config_list = []

        for i in range(length_of_list):
            config = self.generate_a_random_config()
            self.bbg_config_list.append(config)
