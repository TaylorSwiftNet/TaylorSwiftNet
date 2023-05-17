import os

import numpy as np
from torch import Tensor
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

from core.datasets.general_dataset import GeneralDataset
from core.models.general_model import GeneralModel
from core.tools.data_classes import Metrics
from core.tools.utils import *

scheduler_dict = {
    "plateau": ReduceLROnPlateau,
    "cyclic": CyclicLR,
}


def setup_learning_from_checkpoint(cfg: CfgNode):
    """
    Create optimizer and scheduler for resuming from checkpoint.

    Config
    ------
    model.resume_epoch

    Submodules
    ----------
    find_model_state
    """

    model_state = torch.load(find_model_state(cfg))
    model_optimizer = model_state["optimizer"]
    scheduler = model_state["scheduler"]
    epoch = model_state["epoch"]
    if cfg.model.resume_epoch == -1:
        start_epoch = epoch
    else:
        start_epoch = cfg.model.resume_epoch

    return model_optimizer, scheduler, start_epoch


def setup_learning_from_config(cfg: CfgNode, model: GeneralModel):
    """
    Create optimizer and scheduler based on config.
    
    Config
    ------
    trainer.optimizer.type
    trainer.optimizer.lr
    trainer.optimizer.weight_decay
    trainer.scheduler.type
    trainer.scheduler.config.**
    """

    if cfg.trainer.optimizer.type == "adam":
        model_optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.trainer.optimizer.lr,
            weight_decay=cfg.trainer.optimizer.weight_decay,
        )
    elif cfg.trainer.optimizer.name == "adamw":
        model_optimizer = optim.AdamW(model.parameters(), lr=cfg.trainer.optimizer.lr,)
    else:
        ValueError("Illegal optimizer type")

    if cfg.trainer.scheduler.type == "cyclic":
        scheduler = scheduler_dict[cfg.trainer.scheduler.type](
            model_optimizer, **cfg.trainer.scheduler.config
        )
    else:
        # plateau
        scheduler = scheduler_dict["plateau"](
            model_optimizer,
            mode=cfg.trainer.scheduler.mode,
            factor=cfg.trainer.scheduler.factor,
            patience=cfg.trainer.scheduler.patience,
            min_lr=1e-7,
        )

    return model_optimizer, scheduler, 0


class Trainer:
    """
    Main class for training.
    
    Config
    ------
    trainer.batch_size
    trainer.num_epochs
    system.device
    dataset.seen_frames
    dataset.unseen_frames
    dataset.total_frames
    dataset.demo.enable
    dataset.demo.length
    system.num_workers
    experiment_dir
    model.name
    model.resume
    tqdm_length
    system.debugger
    trainer.log_interval
    trainer.optimizer.enable_gradient_clip
    trainer.optimizer.gradient_clip_value
    trainer.scheduler.type
    trainer.enable_histogram
    trainer.save_interval
    trainer.scheduler.metric

    Submodules
    ----------
    setup_learning_from_checkpoint
    setup_learning_from_config
    """

    def __init__(
        self,
        cfg: CfgNode,
        model: GeneralModel,
        train_dataset: GeneralDataset,
        test_dataset: GeneralDataset,
        experiment_num: int,
    ):
        self.cfg = cfg
        self.batch_size = cfg.trainer.batch_size
        self.epochs = cfg.trainer.num_epochs
        self.device = torch.device(cfg.system.device)
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = test_dataset

        if torch.cuda.device_count() > 1:
            self._model = model.module
        else:
            self._model = model

        self.seen_frames = cfg.dataset.seen_frames
        self.unseen_frames = cfg.dataset.unseen_frames
        self.pid = os.getpid()

        train_sampler = None
        val_sampler = None
        if cfg.dataset.demo.enable:
            indices = np.arange(len(train_dataset))
            np.random.shuffle(indices)
            train_sampler = SubsetRandomSampler(indices[: cfg.dataset.demo.length])

        self.train_data_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True if not cfg.dataset.demo.enable else False,
            drop_last=True,
            num_workers=cfg.system.num_workers,
            sampler=train_sampler,
        )
        self.val_data_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=True,
            drop_last=True,
            num_workers=cfg.system.num_workers,
            sampler=val_sampler,
        )

        if self.cfg.dataset.total_frames > 20:
            self.val_data_loader.dataset.set_config_list()

        self.experiment_root = os.path.join(cfg.experiment_dir, cfg.model.name)
        list_of_dirs = ["image", "save", "plot", "summary", "text", "gif"]
        iterative_check_path(self.experiment_root, list_of_dirs)

        self.experiment_num = experiment_num
        iterative_check_path(self.experiment_root, list_of_dirs, self.experiment_num)

        self.model_optimizer, self.scheduler, self.start_epoch = (
            setup_learning_from_checkpoint(self.cfg)
            if cfg.model.resume
            else setup_learning_from_config(self.cfg, self.model)
        )

        self.metric_keys = [
            "total_loss",
            "mse_loss",
            "mse_metric",
            "mae_metric",
            "ssim_metric",
            "temporal_ssim",
        ]

        self.train_metrics = Metrics(
            keys=self.metric_keys,
            train=True,
            iterations=len(self.train_data_loader),
            batch_size=self.train_data_loader.batch_size,
            unseen_frames=cfg.dataset.unseen_frames,
        )
        self.val_metrics = Metrics(
            keys=self.metric_keys,
            train=False,
            iterations=len(self.val_data_loader),
            batch_size=self.val_data_loader.batch_size,
            unseen_frames=cfg.dataset.unseen_frames,
        )
        self.train_bar = None
        self.val_bar = None

    def train_model(self):

        for epoch in range(self.start_epoch, self.epochs):
            self.train_one_epoch(epoch)


    def train_one_epoch(self, epoch: int):
        set_seed(epoch)
        self.train_data_loader.dataset.set_config_list()
        self.model.train()

        self.train_bar = tqdm(
            self.train_data_loader,
            ncols=self.cfg.tqdm_length,
            leave=bool(epoch == self.epochs - 1),
        )
        for i, batch in enumerate(self.train_bar):
            self.train_one_batch(batch, epoch, i)
            if self.cfg.system.debugger and i > 3:
                break

        if epoch % self.cfg.trainer.log_interval == 0:
            self.evaluate_model(epoch)
            self.val_metrics.end_of_epoch()

        self.train_metrics.end_of_epoch()
        self.end_of_epoch(epoch)


    def train_one_batch(self, batch: List[Tensor], epoch: int, _iter: int):
        """ feed forward and back propagation """
        out, loss = self.train_dataset.feed_forward(
            batch, self.model, self.train_metrics, self._model, None,
        )

        self.model_optimizer.zero_grad()
        (loss.total_loss["loss"] / loss.total_loss["weight"]).backward()

        if self.cfg.trainer.optimizer.enable_gradient_clip:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.trainer.optimizer.gradient_clip_value
            )

        nan_happened = torch.isnan(loss.total_loss["loss"]).any()
        if nan_happened:
            print("Loss is NaN")
            raise ValueError("loss in NaN!")

        self.model_optimizer.step()

        step_metrics = self.train_metrics.end_of_step()
        if step_metrics is not None:
            step_loss = step_metrics["loss"]
            step_ssim = step_metrics["ssim"]

            self.train_bar.set_postfix(
                {
                    "train_loss": step_loss,
                    "train_ssim": step_ssim,
                    "iter": _iter,
                    "epoch": epoch + 1,
                }
            )

    def end_of_epoch(self, epoch: int):
        if self.cfg.trainer.scheduler.type == "cyclic":
            self.scheduler.step()
        else:
            self.scheduler.step(self.update_scheduler())


        if epoch % self.cfg.trainer.log_interval == 0:
            if epoch % self.cfg.trainer.save_interval == 0:
                checkpoint = {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.model_optimizer,
                    "scheduler": self.scheduler,
                }
                torch.save(
                    checkpoint,
                    os.path.join(
                        self.experiment_root,
                        f"save/{self.experiment_num}/epoch_{epoch:04d}.pt",
                    ),
                )

            self.model.eval()
            
    def evaluate_model(self, epoch: int):
        self.model.eval()
        # deactivating the autograd engine
        with torch.no_grad():
            self.val_bar = tqdm(
                self.val_data_loader,
                ncols=self.cfg.tqdm_length,
                leave=bool(epoch == self.epochs - 1),
            )
            for i, batch in enumerate(self.val_bar):
                self.evaluate_one_batch(batch, epoch)
                if self.cfg.system.debugger and i > 3:
                    break

    def evaluate_one_batch(self, batch: List[Tensor], epoch: int):
        _ = self.val_dataset.feed_forward(
            batch, self.model, self.val_metrics, self._model, None,
        )

        step_metrics = self.val_metrics.end_of_step()
        if step_metrics is not None:
            step_loss = step_metrics["loss"]
            step_ssim = step_metrics["ssim"]
            self.val_bar.set_postfix(
                {"val_loss": step_loss, "val_ssim": step_ssim, "epoch": epoch}
            )

    def update_scheduler(self):
        if self.cfg.trainer.scheduler.use_train:
            step = self.train_metrics.all_metrics[self.cfg.trainer.scheduler.metric][-1]
            if isinstance(step, list):
                step = step[-1]
        else:
            step = self.val_metrics.all_metrics[self.cfg.trainer.scheduler.metric][-1]
            if isinstance(step, list):
                step = step[-1]

        return step
