import gzip
import os
import random
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torchvision import transforms
from yacs.config import CfgNode

from .general_dataset import GeneralDataset


def load_mnist(root, train: bool) -> np.ndarray:
    """Load MNIST dataset for generating training data."""
    if train:
        path = os.path.join(root, "train-images-idx3-ubyte.gz")
    else:
        path = os.path.join(root, "t10k-images-idx3-ubyte.gz")
    with gzip.open(path, "rb") as f:
        mnist = np.frombuffer(f.read(), np.uint8, offset=16)
        mnist = mnist.reshape(-1, 28, 28)
    return mnist


def load_fixed_set(root: str) -> np.ndarray:
    """Load the fixed dataset"""
    filename = "mnist_test_seq.npy"
    path = os.path.join(root, filename)
    dataset = np.load(path)
    dataset = dataset[..., np.newaxis]
    return dataset


class MovingMNIST(GeneralDataset):
    def __init__(
        self,
        cfg: CfgNode,
        train: bool,
        transform: Optional[Callable] = None,
        num_objects=[2],
    ):
        """
        Dataset class for MovingMNIST. The sequences are generated on the fly based on the static
        MNIST dataset. By default 10,000 samples of length `cfg.datasets.total_frames` are 
        generated.

        Config
        ------
        dataset.root
        dataset.seen_frames
        dataset.unseen_frames
        dataset.total_frames
        dataset.image_size

        Submodules
        ----------
        GeneralDataset
        """
        super(MovingMNIST, self).__init__(cfg=cfg)
        root = cfg.dataset.root

        self.seen_frames = cfg.dataset.seen_frames
        self.unseen_frames = cfg.dataset.unseen_frames
        self.total_frames = cfg.dataset.total_frames

        self.dataset = None

        if train:
            self.mnist = load_mnist(root, True)
        else:
            if num_objects[0] != 2 or self.total_frames > 20:
                self.mnist = load_mnist(root, False)
            else:
                self.dataset = load_fixed_set(root)

        self.length = int(1e4) if self.dataset is None else self.dataset.shape[1]

        self.train = train
        self.num_objects = num_objects
        self.transform = transform
        # For generating data
        self.image_size_ = cfg.dataset.image_size
        self.digit_size_ = 28
        self.step_length_ = 0.1
        self.rand_ind_array = []

    def get_random_trajectory(
        self,
        seq_length: int,
        x: Optional[float] = None,
        y: Optional[float] = None,
        theta: Optional[float] = None,
        step: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if step is None:
            step = self.step_length_
        # Generate a random sequence of a MNIST digit
        canvas_size = self.image_size_ - self.digit_size_
        if x is None:
            x = random.random()
        if y is None:
            y = random.random()
        if theta is None:
            theta = random.random() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)
        for i in range(seq_length):
            # Take a step along velocity.
            y += v_y * step
            x += v_x * step

            # Bounce off edges.
            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y
            start_y[i] = y
            start_x[i] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def generate_moving_mnist(self, config=None) -> np.ndarray:
        """Get random trajectories for the digits and generate a video."""
        if config is None:
            num_digits = random.choice(self.num_objects)
        else:
            num_digits = config[-1]

        data = np.zeros(
            (self.total_frames, self.image_size_, self.image_size_), dtype=np.float32
        )
        for n in range(num_digits):
            # Trajectory
            if config is None:
                start_y, start_x = self.get_random_trajectory(self.total_frames)
                ind = random.randint(0, self.mnist.shape[0] - 1)
            else:
                start_y, start_x = self.get_random_trajectory(
                    self.total_frames, *config[0][n][:-1]
                )
                ind = config[0][n][-1]

            digit_image = self.mnist[ind]
            for i in range(self.total_frames):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.digit_size_
                right = left + self.digit_size_
                # Draw digit
                data[i, top:bottom, left:right] = np.maximum(
                    data[i, top:bottom, left:right], digit_image
                )

        return data / 255

    def generate_a_random_config(self):
        assert self.train or self.num_objects[0] != 2 or self.total_frames > 20
        num_digits = random.choice(self.num_objects)
        random_config = []

        for n in range(num_digits):
            x = random.random()
            y = random.random()
            theta = random.random() * 2 * np.pi

            ind = random.randint(0, self.mnist.shape[0] - 1)
            random_config.append([x, y, theta, ind])
        return [random_config, num_digits]

    def __getitem__(self, idx: int, evaluate: bool = False) -> Tuple[Tensor, Tensor]:
        """generate a sample"""

        if self.train or self.num_objects[0] != 2 or self.total_frames > 20:
            if self.bbg_config_list is None:
                images = self.generate_moving_mnist()
            else:
                images = self.generate_moving_mnist(self.bbg_config_list[idx])
        else:
            images = self.dataset[:, idx, ...]

        if self.transform is not None:
            clip = [self.transform(frame) for frame in images]
        images = torch.stack(clip, 0).permute(1, 0, 2, 3)

        _input = images[:, : self.seen_frames]
        target = images[:, self.seen_frames : self.total_frames]

        return _input, target

    def __len__(self):
        return self.length


def make_db(cfg: CfgNode, train: bool) -> GeneralDataset:
    """
    Generate a MovingMNIST instance with Tensor-conversion and normalization applied.

    Submodules
    ----------
    MovingMNIST
    """

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),]
    )

    return MovingMNIST(cfg, train, transform)
