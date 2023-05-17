from collections import defaultdict
from statistics import mean
from typing import Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_metric
from torch.utils.data import DataLoader
from tqdm import tqdm
from yacs.config import CfgNode

from core.datasets.general_dataset import GeneralDataset
from core.models.general_model import GeneralModel


class Tester:
    def __init__(self, cfg: CfgNode, model: GeneralModel, test_dataset: GeneralDataset):
        """Class for evaluation a model on a dataset.
        """

        self.cfg = cfg
        self.model = model
        self.test_db = test_dataset

    def evaluate(self, num_workers: int = -1) -> Dict[str, float]:
        """Calculate MAE, MSE and SSIM on the test dataset.

        Parameters
        ----------
        num_workers : int, optional
            num workers. If not stated, use config default.

        Returns
        -------
        Dict[str, float]
            test metrics
        """

        if num_workers < 0:
            num_workers = self.cfg.system.num_workers

        loader = DataLoader(
            self.test_db,
            8,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=True,
        )

        all_eval_metrics = defaultdict(float)

        for i, (X, y) in enumerate(tqdm(loader)):

            y = y.to(self.cfg.system.device)

            out, _ = self.test_db.feed_forward((X, y), self.model)

            eval_metrics = self.evaluate_batch(out[0], y)

            for k, v in eval_metrics.items():
                all_eval_metrics[k] += v

        mae = all_eval_metrics["mae_metric"] / all_eval_metrics["count"]
        mse = all_eval_metrics["mse_metric"] / all_eval_metrics["count"]
        ssim = all_eval_metrics["ssim_metric"] / all_eval_metrics["count"]

        print(f"Test MAE: \t{mae:.4f}")
        print(f"Test MSE: \t{mse:.4f}")
        print(f"Test SSIM: \t{ssim:.4f}")

        return dict(mae=mae, mse=mse, ssim=ssim)

    @staticmethod
    @torch.no_grad()
    def evaluate_batch(
        prediction: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, Union[float, int]]:
        """Calculate evaluation metrics. 

        MSE and MAE are summed over batch, height and width dimension and averaged over channel and
        time dimension. SSIM is summed over batch dimension and averaged over channel and time
        dimension. This means, that the returned values need to be devided by the batch size to get
        correct results.

        Parameters
        ----------
        prediction : Tensor
            Tensor of shape (batch, channels, time, height, width).
        target : Tense
            same shape as `prediction`.

        Returns
        -------
        dict
            dictionary containing "ssim_metric", "mse_metric, "mae_metric" and "count" (number of
            samples in this batch).
        """
        b, c, t, h, w = prediction.shape

        prediction = prediction * 0.5 + 0.5  # [-1, 1] -> [0, 1]
        target = target * 0.5 + 0.5

        mse_batch = F.mse_loss(prediction, target, reduction="none")  # (b c t h w)
        mae_batch = F.l1_loss(prediction, target, reduction="none")  # (b c t h w)

        mse_batch = torch.mean(torch.sum(mse_batch, dim=(0, -1, -2)))
        mae_batch = torch.mean(torch.sum(mae_batch, dim=(0, -1, -2)))

        prediction = prediction.detach().cpu().numpy().astype(np.float32)
        target = target.detach().cpu().numpy().astype(np.float32)

        ssim_batch = 0
        for i in range(b):
            ssim_batch += mean(
                ssim_metric(target[i, j, k], prediction[i, j, k])
                for j in range(c)
                for k in range(t)
            )

        # calculating binary cross entropy
        return dict(
            mse_metric=mse_batch.item(),
            mae_metric=mae_batch.item(),
            ssim_metric=ssim_batch,
            count=b,
        )
