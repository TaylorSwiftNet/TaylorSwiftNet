import argparse
import os
import random
from pathlib import Path
from typing import List, Optional

import cv2 as cv
import imageio
import numpy as np
import torch
import torchvision
from torch import Tensor, nn
from yacs.config import CfgNode

from configs.default_config import get_config_defaults


def init_weights(m: nn.Module):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        if hasattr(m, "weight"):
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def set_seed(seed: int, fully_deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if fully_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_gif(
    generated_samples: Tensor,
    target_samples: Tensor,
    path: str,
    val_samples=None,
    val_targets=None,
):
    border = 3
    b, c, t, h, w = generated_samples.shape

    norm_gs = normalize(generated_samples).cpu().numpy().astype(np.uint8)
    norm_ts = normalize(target_samples).cpu().numpy().astype(np.uint8)
    gs_wb = add_boarder(norm_gs, border)
    ts_wb = add_boarder(norm_ts, border)

    compare_results = np.concatenate((gs_wb, ts_wb), axis=-1)

    if val_samples is not None and val_targets is not None:
        _norm_gs = normalize(val_samples).cpu().numpy().astype(np.uint8)
        _norm_ts = normalize(val_targets).cpu().numpy().astype(np.uint8)

        gs_wb = add_boarder(_norm_gs, border)
        ts_wb = add_boarder(_norm_ts, border)

        _compare_results = np.concatenate((gs_wb, ts_wb), axis=-1)
        compare_results = np.concatenate((compare_results, _compare_results), axis=-2)

    if c == 3:
        imageio.mimsave(path, np.moveaxis(compare_results.squeeze(), 0, -1))
    elif c == 2:
        blue_ch = np.ones_like(compare_results.squeeze()[0:1]) * 255
        compare_results = np.concatenate([compare_results.squeeze(), blue_ch], axis=0)
        imageio.mimsave(path, np.moveaxis(compare_results.squeeze(), 0, -1))
    else:
        imageio.mimsave(path, compare_results.squeeze())


def add_boarder(images: np.ndarray, border: int) -> np.ndarray:
    """
    Add a border to images

    Parameters
    ----------
    images: ndarray
        input images
    border: int
        border you want to wrap around the image

    Returns
    -------
    ndarray
        compared results with border
    """

    if len(images.shape) == 5:
        b, c, t, h, w = images.shape
    elif len(images.shape) == 4:
        c = 1
        b, t, h, w = images.shape
    elif len(images.shape) == 3:
        b, c = 1, 1
        t, h, w = images.shape
    else:
        b, c, t = 1, 1, 1
        h, w = images.shape
    images = images.reshape(b, c, t, h, w)

    images_wb = np.zeros((b, c, t, h + 2 * border, w + 2 * border)).astype(np.uint8)
    for i in range(b):
        for j in range(c):
            for k in range(t):
                images_wb[i][j][k] = cv.copyMakeBorder(
                    images[i][j][k],
                    border,
                    border,
                    border,
                    border,
                    borderType=cv.BORDER_CONSTANT,
                    value=127,
                )
    return images_wb


def save_image(list_of_samples: List[Tensor], path: str, save: bool = True) -> Tensor:
    selected_t_list = []
    for samples in list_of_samples:
        selected_t = []
        for j in range(samples.shape[0]):
            selected_t += [samples[j, :, j % samples.shape[2]]]
        selected_t_list += [
            torchvision.utils.make_grid(
                torch.stack(selected_t),
                nrow=1,
                padding=2,
                normalize=True,
                pad_value=100,
            )
        ]
    samples_stack = torch.cat(selected_t_list, dim=-1)

    np_grid = (samples_stack * 255).cpu().numpy().astype(np.uint8)
    np_grid = np.moveaxis(np_grid, 0, -1)

    if save:
        if np_grid.shape[2] == 3:
            cv.imwrite(path, cv.cvtColor(np_grid, cv.COLOR_BGR2RGB))
        elif np_grid.shape[2] == 2:
            blue_ch = np.ones_like(np_grid[..., 0:1]) * 255
            np_grid = np.concatenate([np_grid, blue_ch], axis=-1)
            cv.imwrite(path, cv.cvtColor(np_grid, cv.COLOR_BGR2RGB))

    images_tensor = torch.from_numpy(np_grid)
    return images_tensor.permute(2, 0, 1)


def find_latest_experiment(path: str):
    list_of_experiments = os.listdir(path)
    list_of_int_experiments = []
    for exp in list_of_experiments:
        try:
            int_exp = int(exp)
        except ValueError:
            continue
        list_of_int_experiments.append(int_exp)

    if len(list_of_int_experiments) == 0:
        return 0

    return max(list_of_int_experiments)


def print_config(cfg: CfgNode):
    for key, value in cfg.items():
        print(key, value)


def iterative_check_path(root: str, _list: list, _num=None):
    for _l in _list:
        if _num is None:
            path = os.path.join(root, _l)
        else:
            path = os.path.join(root, _l, str(_num))
        check_path(path)


def check_path(path: str):
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except FileNotFoundError:
            check_path(os.path.dirname(path))
            os.mkdir(path)


def normalize(x: Tensor, ret_uint: bool = True):
    if ret_uint:
        return torch.mul(torch.add(torch.mul(x, 0.5), 0.5), 255).type(torch.uint8)
    else:
        return torch.add(torch.mul(x, 0.5), 0.5)


def parse_args() -> argparse.Namespace:
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument(
        "--cfg", dest="cfg_file", help="optional config file", default=None, type=str
    )
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        help="set config keys",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


BASE_PATH = (Path(__file__).parent.parent.parent).resolve()


def create_cfg(
    cfg_file: Optional[str] = None, set_cfgs: Optional[List[str]] = None
) -> CfgNode:
    if cfg_file is None and set_cfgs is None:
        args = parse_args()
        cfg_file = args.cfg_file
        set_cfgs = args.set_cfgs

    cfg = get_config_defaults()

    print("BASE PATH", BASE_PATH)

    if cfg_file is not None:
        print("Config path:", BASE_PATH / cfg_file)
        cfg.merge_from_file(BASE_PATH / cfg_file)

        model_name = cfg_file.split("/")[-1]
        if model_name.endswith(".yaml"):
            model_name = model_name[: -len(".yaml")]

        cfg.merge_from_list(["model.name", model_name])

    if set_cfgs is not None:
        cfg.merge_from_list(set_cfgs)

    return cfg


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def find_model_state(cfg: CfgNode) -> str:
    """
    Find model state path for a given config file.
    
    Config
    ------
    model.model_state_path
    model.name
    model.resume_experiment_num
    """
    if cfg.model.model_state_path == "":
        path = os.path.join("experiments", cfg.model.name, "save")
        if cfg.model.resume_experiment_num == -1:
            path = os.path.join(path, sorted(os.listdir(path))[-1])
        else:
            path = os.path.join(path, str(cfg.model.resume_experiment_num))

        if len(os.path.splitext(os.listdir(path)[0].split("_")[1])[0]) == 4:
            path = os.path.join(path, sorted(os.listdir(path))[-1])
        else:
            epoch_nums = [
                int(os.path.splitext(epoch.split("_")[1])[0])
                for epoch in os.listdir(path)
            ]
            path = os.path.join(path, f"epoch_{max(epoch_nums)}.pt")
        return path
    else:
        return cfg.model.model_state_path
