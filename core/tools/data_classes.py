from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


from torch import Tensor

TotalLoss = Dict[str, Union[Tensor, int]]


@dataclass
class GeneralLoss:
    total_loss: TotalLoss
    mse_loss: Tensor
    temporal_mse: Optional[Tensor]
    moment_loss: Optional[Tensor]
    kl_loss: Optional[Tensor]

    def __init__(
        self,
        total_loss: TotalLoss,
        mse_loss: Tensor,
        temporal_mse: Optional[Tensor] = None,
        moment_loss: Optional[Tensor] = None,
        kl_loss: Optional[Tensor] = None,
    ):
        """Manage losses calculated during training.
        """
        self.total_loss = total_loss
        self.mse_loss = mse_loss
        if temporal_mse is not None:
            self.temporal_mse = temporal_mse
        if moment_loss is not None:
            self.moment_loss = moment_loss
        if kl_loss is not None:
            self.kl_loss = kl_loss

    def __add__(self, other: "GeneralLoss"):
        """Add to GeneralLoss objects. 

        Note: Weighted total loss is not supported!
        """
        total_loss = {
            "loss": self.total_loss["loss"] + other.total_loss["loss"],
            "weight": 1,
        }
        mse_loss = self.mse_loss + other.mse_loss
        temporal_mse = None
        moment_loss = None
        kl_loss = None
        try:
            temporal_mse = self.temporal_mse + other.temporal_mse
        except:
            pass
        try:
            moment_loss = self.moment_loss + other.moment_loss
        except:
            pass
        try:
            kl_loss = self.kl_loss + other.kl_loss
        except:
            pass

        return GeneralLoss(total_loss, mse_loss, temporal_mse, moment_loss, kl_loss)

    def __truediv__(self, other: "GeneralLoss"):
        total_loss = {"loss": self.total_loss["loss"] / other, "weight": 1}
        mse_loss = self.mse_loss / other
        temporal_mse = None
        moment_loss = None
        kl_loss = None
        try:
            temporal_mse = self.temporal_mse / other
        except:
            pass
        try:
            moment_loss = self.moment_loss / other
        except:
            pass
        try:
            kl_loss = self.kl_loss / other
        except:
            pass

        return GeneralLoss(total_loss, mse_loss, temporal_mse, moment_loss, kl_loss)


@dataclass
class Metrics:
    def __init__(
        self,
        keys: List[str],
        train: bool,
        iterations: int,
        batch_size: int,
        unseen_frames: int,
    ):
        """Dataclass for collecting and aggregating metrics.
        """
        self.keys = keys
        self.metrics = {}
        self.all_metrics = {}
        self.train = train
        self.batch_size = batch_size
        self.unseen_frames = unseen_frames

        for key in keys:
            if key in ["temporal_mse", "temporal_ssim"]:
                self.metrics.update({key: [[] for _ in range(self.unseen_frames)]})
            elif key in ["derivative_metric_l1", "derivative_metric_l2"]:
                self.metrics.update({key: [[] for _ in range(2)]})
            else:
                self.metrics.update({key: 0})
            self.all_metrics.update({"all_" + key: []})
        self.iter = -1
        self.iterations = iterations
        self.total_loss_weight = 0
        self.step_counter = -1

    def append(self, list_of_dict: List[Dict[str, Any]]):
        for _dict in list_of_dict:
            for key, value in _dict.items():
                if key in list(self.metrics.keys()):
                    if isinstance(value, list):
                        for i in range(len(self.metrics[key])):
                            self.metrics[key][i] += [
                                v.item() if hasattr(v, "item") else v for v in value[i]
                            ]
                    elif isinstance(value, dict):
                        value, weight = value["loss"], value["weight"]
                        self.metrics[key] += (
                            value.item() if hasattr(value, "item") else value
                        )
                        self.total_loss_weight += (
                            weight.item() if hasattr(weight, "item") else weight
                        )
                    else:
                        self.metrics[key] += (
                            value.item() if hasattr(value, "item") else value
                        )
                else:
                    raise Exception(
                        "expected to receive a key initialized before, key: {} , keys: {}".format(
                            key, list(self.metrics.keys())
                        )
                    )

    def end_of_step(self):
        self.step_counter += 1
        if self.step_counter % 5 == 0:
            loss = self.metrics["total_loss"] / self.total_loss_weight
            ssim = self.metrics["ssim_metric"] / (self.iterations * self.batch_size)
            return {"loss": loss, "ssim": ssim}
        return

    def end_of_epoch(self):
        for key in self.all_metrics:
            if key[4:] in ["temporal_mse", "temporal_ssim"]:
                self.all_metrics[key].append(
                    [sum(value) / len(value) for value in self.metrics[key[4:]]]
                )
                self.metrics[key[4:]] = [[] for _ in range(self.unseen_frames)]
            elif key[4:] in ["derivative_metric_l1", "derivative_metric_l2"]:
                self.all_metrics[key].append(
                    [sum(value) / len(value) for value in self.metrics[key[4:]]]
                )
                self.metrics[key[4:]] = [[] for _ in range(2)]

            elif key[4:] == "total_loss":
                self.all_metrics[key].append(
                    self.metrics[key[4:]] / self.total_loss_weight
                )
                self.metrics[key[4:]] = 0
                self.total_loss_weight = 0
            else:
                self.all_metrics[key].append(
                    self.metrics[key[4:]] / (self.iterations * self.batch_size)
                )
                self.metrics[key[4:]] = 0

    def values(self):
        return list(self.all_metrics.values())

    def __iter__(self):
        return self

    def __next__(self):
        self.iter += 1
        if self.iter < len(self.keys):
            return self.keys[self.iter], list(self.all_metrics.values())[self.iter]

        else:
            self.iter = -1
            raise StopIteration
