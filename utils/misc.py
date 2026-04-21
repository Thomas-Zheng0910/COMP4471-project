"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

from functools import wraps
from time import time
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from scipy import interpolate


@torch.jit.script
def max_stack(tensors: List[torch.Tensor]) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0]
    return torch.stack(tensors, dim=-1).max(dim=-1).values


def last_stack(tensors: List[torch.Tensor]) -> torch.Tensor:
    return tensors[-1]


def first_stack(tensors: List[torch.Tensor]) -> torch.Tensor:
    return tensors[0]


@torch.jit.script
def softmax_stack(
    tensors: List[torch.Tensor], temperature: float = 1.0
) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0]
    return F.softmax(torch.stack(tensors, dim=-1) / temperature, dim=-1).sum(dim=-1)


@torch.jit.script
def mean_stack(tensors: List[torch.Tensor]) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0]
    return torch.stack(tensors, dim=-1).mean(dim=-1)


@torch.jit.script
def sum_stack(tensors: List[torch.Tensor]) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0]
    return torch.stack(tensors, dim=-1).sum(dim=-1)


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()


def format_seconds(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def get_params(module, lr, wd):
    skip_list = {}
    skip_keywords = {}
    if hasattr(module, "no_weight_decay"):
        skip_list = module.no_weight_decay()
    if hasattr(module, "no_weight_decay_keywords"):
        skip_keywords = module.no_weight_decay_keywords()
    has_decay = []
    no_decay = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            (name in skip_list)
            or any((kw in name for kw in skip_keywords))
            or len(param.shape) == 1
            or name.endswith(".gamma")
            or name.endswith(".beta")
            or name.endswith(".bias")
        ):
            no_decay.append(param)
        else:
            has_decay.append(param)

    group1 = {
        "params": has_decay,
        "weight_decay": wd,
        "lr": lr,
        "weight_decay_init": wd,
        "weight_decay_base": wd,
        "lr_base": lr,
    }
    group2 = {
        "params": no_decay,
        "weight_decay": 0.0,
        "lr": lr,
        "weight_decay_init": 0.0,
        "weight_decay_base": 0.0,
        "weight_decay_final": 0.0,
        "lr_base": lr,
    }
    return [group1, group2], [lr, lr]


def log(t, eps: float = 1e-5):
    return torch.log(t.clamp(min=eps))


def l2norm(t):
    return F.normalize(t, dim=-1)


def exists(val):
    return val is not None


def identity(t, *args, **kwargs):
    return t


def divisible_by(numer, denom):
    return (numer % denom) == 0


def first(arr, d=None):
    if len(arr) == 0:
        return d
    return arr[0]


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def maybe(fn):
    @wraps(fn)
    def inner(x):
        if not exists(x):
            return x
        return fn(x)

    return inner


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


def _many(fn):
    @wraps(fn)
    def inner(tensors, pattern, **kwargs):
        return (fn(tensor, pattern, **kwargs) for tensor in tensors)

    return inner


rearrange_many = _many(rearrange)
repeat_many = _many(repeat)
reduce_many = _many(reduce)


def add_padding_metas(out, image_metas):
    device = out.device
    # left, right, top, bottom
    paddings = [img_meta.get("paddings", [0] * 4) for img_meta in image_metas]
    paddings = torch.stack(paddings).to(device)
    outs = [F.pad(o, padding, value=0.0) for padding, o in zip(paddings, out)]
    return torch.stack(outs)


# left, right, top, bottom
def remove_padding(out, paddings):
    H, W = out.shape[-2:]
    outs = [
        o[..., padding[2] : H - padding[3], padding[0] : W - padding[1]]
        for padding, o in zip(paddings, out)
    ]
    return torch.stack(outs)


def remove_padding_metas(out, image_metas):
    B, C, H, W = out.shape
    device = out.device
    # left, right, top, bottom
    paddings = [
        torch.tensor(img_meta.get("paddings", [0] * 4)) for img_meta in image_metas
    ]
    return remove_padding(out, paddings)


def ssi_helper(tensor1, tensor2):
    stability_mat = 1e-4 * torch.eye(2, device=tensor1.device)
    tensor2_one = torch.stack([tensor2, torch.ones_like(tensor2)], dim=1)
    scale_shift = torch.inverse(tensor2_one.T @ tensor2_one + stability_mat) @ (
        tensor2_one.T @ tensor1.unsqueeze(1)
    )
    scale, shift = scale_shift.squeeze().chunk(2, dim=0)
    return scale, shift


def calculate_mean_values(names, values):
    name_values = {name: {} for name in names}
    for name, value in zip(names, values):
        name_values[name]["sum"] = name_values[name].get("sum", 0.0) + value
        name_values[name]["count"] = name_values[name].get("count", 0.0) + 1
    output_dict = {
        name: name_values[name]["sum"] / name_values[name]["count"]
        for name in name_values
    }
    return output_dict


def remove_leading_dim(infos):
    if isinstance(infos, dict):
        return {k: remove_leading_dim(v) for k, v in infos.items()}
    elif isinstance(infos, torch.Tensor):
        return infos.squeeze(0)
    else:
        return infos


def recursive_index(infos, index):
    if isinstance(infos, dict):
        return {k: recursive_index(v, index) for k, v in infos.items()}
    elif isinstance(infos, torch.Tensor):
        return infos[index]
    else:
        return infos


def to_cpu(infos):
    if isinstance(infos, dict):
        return {k: to_cpu(v) for k, v in infos.items()}
    elif isinstance(infos, torch.Tensor):
        return infos.detach()
    else:
        return infos


def recursive_to(infos, device, non_blocking, cls):
    if isinstance(infos, dict):
        return {k: recursive_to(v, device, non_blocking, cls) for k, v in infos.items()}
    elif isinstance(infos, list):
        return [recursive_to(v, device, non_blocking, cls) for v in infos]
    elif isinstance(infos, cls):
        return infos.to(device, non_blocking=non_blocking)
    else:
        return infos


def masked_mean(
    data: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dim: Optional[List[int]] = None,
    keepdim: bool = False,
) -> torch.Tensor:
    dim = dim if dim is not None else list(range(data.dim()))
    if mask is None:
        return data.mean(dim=dim, keepdim=keepdim)
    mask = mask.float()
    mask_sum = torch.sum(mask, dim=dim, keepdim=True)
    mask_mean = torch.sum(data * mask, dim=dim, keepdim=True) / torch.clamp(
        mask_sum, min=1.0
    )
    return mask_mean.squeeze(dim) if not keepdim else mask_mean


class ProfileMethod:
    def __init__(self, model, func_name, track_statistics=True, verbose=False):
        self.model = model
        self.func_name = func_name
        self.verbose = verbose
        self.track_statistics = track_statistics
        self.timings = []

    def __enter__(self):
        if self.verbose:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start_time = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.end_time = time()
            elapsed_time = self.end_time - self.start_time
            self.timings.append(elapsed_time)
            if self.track_statistics and len(self.timings) > 25:
                timings_array = np.array(self.timings)
                mean_time = np.mean(timings_array)
                std_time = np.std(timings_array)
                quantiles = np.percentile(timings_array, [0, 25, 50, 75, 100])
                print(
                    f"{self.model.__class__.__name__}.{self.func_name} took {elapsed_time:.4f} seconds"
                )
                print(f"Mean Time: {mean_time:.4f} seconds")
                print(f"Std Time: {std_time:.4f} seconds")
                print(
                    f"Quantiles: Min={quantiles[0]:.4f}, 25%={quantiles[1]:.4f}, Median={quantiles[2]:.4f}, 75%={quantiles[3]:.4f}, Max={quantiles[4]:.4f}"
                )
            else:
                print(
                    f"{self.model.__class__.__name__}.{self.func_name} took {elapsed_time:.4f} seconds"
                )


def profile_method(track_statistics=True, verbose=False):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with ProfileMethod(self, func.__name__, track_statistics, verbose):
                return func(self, *args, **kwargs)

        return wrapper

    return decorator


def squeeze_list(nested_list, dim, current_dim=0):
    if isinstance(nested_list, list) and len(nested_list) == 1 and current_dim == dim:
        return squeeze_list(nested_list[0], dim, current_dim + 1)
    elif isinstance(nested_list, list):
        return [squeeze_list(item, dim, current_dim + 1) for item in nested_list]
    else:
        return nested_list


def match_gt(tensor1, tensor2, padding1, padding2, mode: str = "bilinear"):
    """
    Transform each item in tensor1 batch to match tensor2's dimensions and padding.
    """
    batch_size = len(tensor1)
    transformed_tensors = []

    for i in range(batch_size):
        item1 = tensor1[i]
        item2 = tensor2[i]

        H2, W2 = item2.shape[-2:]

        # Remove padding from tensor1
        if padding1 is not None:
            pad_left, pad_right, pad_top, pad_bottom = (
                int(padding1[i, 0]),
                int(padding1[i, 1]),
                int(padding1[i, 2]),
                int(padding1[i, 3]),
            )
            H1, W1 = item1.shape[-2:]
            item1 = item1[
                ...,
                pad_top : H1 - pad_bottom if pad_bottom > 0 else H1,
                pad_left : W1 - pad_right if pad_right > 0 else W1,
            ]

        # Resize to match tensor2
        item1 = F.interpolate(
            item1.unsqueeze(0).float(),
            size=(H2, W2),
            mode=mode,
            align_corners=False,
        ).squeeze(0)

        transformed_tensors.append(item1)

    return torch.stack(transformed_tensors)


def match_intrinsics(intrinsics, image, depth_gt, padding1, padding2):
    """
    Adjust intrinsics to match the depth GT resolution after removing padding.
    """
    batch_size = intrinsics.shape[0]
    adjusted_intrinsics = []

    for i in range(batch_size):
        K = intrinsics[i].clone()
        H_img, W_img = image.shape[-2:]
        H_gt, W_gt = depth_gt[i].shape[-2:]

        if padding1 is not None:
            pad_left = int(padding1[i, 0])
            pad_top = int(padding1[i, 2])
            K[0, 2] = K[0, 2] - pad_left
            K[1, 2] = K[1, 2] - pad_top
            H_cropped = H_img - int(padding1[i, 2]) - int(padding1[i, 3])
            W_cropped = W_img - int(padding1[i, 0]) - int(padding1[i, 1])
        else:
            H_cropped, W_cropped = H_img, W_img

        scale_x = W_gt / W_cropped
        scale_y = H_gt / H_cropped
        K[0, 0] *= scale_x
        K[1, 1] *= scale_y
        K[0, 2] *= scale_x
        K[1, 2] *= scale_y

        adjusted_intrinsics.append(K)

    return torch.stack(adjusted_intrinsics)
