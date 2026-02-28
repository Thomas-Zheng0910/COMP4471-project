"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

from typing import Tuple

import torch
from torch.nn import functional as F


@torch.jit.script
def generate_rays(
    camera_intrinsics: torch.Tensor, image_shape: Tuple[int, int], noisy: bool = False
):
    batch_size, device, dtype = (
        camera_intrinsics.shape[0],
        camera_intrinsics.device,
        camera_intrinsics.dtype,
    )
    height, width = image_shape
    # Generate grid of pixel coordinates
    pixel_coords_x = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    pixel_coords_y = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    if noisy:
        pixel_coords_x += torch.rand_like(pixel_coords_x) - 0.5
        pixel_coords_y += torch.rand_like(pixel_coords_y) - 0.5
    pixel_coords = torch.stack(
        [pixel_coords_x.repeat(height, 1), pixel_coords_y.repeat(width, 1).t()], dim=2
    )  # (H, W, 2)
    pixel_coords = pixel_coords + 0.5

    # Calculate ray directions
    intrinsics_inv = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    intrinsics_inv[:, 0, 0] = 1.0 / camera_intrinsics[:, 0, 0]
    intrinsics_inv[:, 1, 1] = 1.0 / camera_intrinsics[:, 1, 1]
    intrinsics_inv[:, 0, 2] = -camera_intrinsics[:, 0, 2] / camera_intrinsics[:, 0, 0]
    intrinsics_inv[:, 1, 2] = -camera_intrinsics[:, 1, 2] / camera_intrinsics[:, 1, 1]
    homogeneous_coords = torch.cat(
        [pixel_coords, torch.ones_like(pixel_coords[:, :, :1])], dim=2
    )  # (H, W, 3)
    ray_directions = torch.matmul(
        intrinsics_inv, homogeneous_coords.permute(2, 0, 1).flatten(1)
    )  # (3, H*W)
    ray_directions = F.normalize(ray_directions, dim=1)  # (B, 3, H*W)
    ray_directions = ray_directions.permute(0, 2, 1)  # (B, H*W, 3)

    theta = torch.atan2(ray_directions[..., 0], ray_directions[..., -1])
    phi = torch.acos(ray_directions[..., 1])
    angles = torch.stack([theta, phi], dim=-1)
    return ray_directions, angles


@torch.jit.script
def spherical_zbuffer_to_euclidean(spherical_tensor: torch.Tensor) -> torch.Tensor:
    theta = spherical_tensor[..., 0]  # Extract polar angle
    phi = spherical_tensor[..., 1]  # Extract azimuthal angle
    z = spherical_tensor[..., 2]  # Extract zbuffer depth

    x = z * torch.tan(theta)
    y = z / torch.tan(phi) / torch.cos(theta)

    euclidean_tensor = torch.stack((x, y, z), dim=-1)
    return euclidean_tensor


@torch.jit.script
def spherical_to_euclidean(spherical_tensor: torch.Tensor) -> torch.Tensor:
    theta = spherical_tensor[..., 0]
    phi = spherical_tensor[..., 1]
    r = spherical_tensor[..., 2]
    x = r * torch.sin(phi) * torch.sin(theta)
    y = r * torch.cos(phi)
    z = r * torch.cos(theta) * torch.sin(phi)
    euclidean_tensor = torch.stack((x, y, z), dim=-1)
    return euclidean_tensor


@torch.jit.script
def euclidean_to_spherical(spherical_tensor: torch.Tensor) -> torch.Tensor:
    x = spherical_tensor[..., 0]
    y = spherical_tensor[..., 1]
    z = spherical_tensor[..., 2]
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.atan2(x / r, z / r)
    phi = torch.acos(y / r)
    euclidean_tensor = torch.stack((theta, phi, r), dim=-1)
    return euclidean_tensor


@torch.jit.script
def unproject_points(
    depth: torch.Tensor, camera_intrinsics: torch.Tensor
) -> torch.Tensor:
    batch_size, _, height, width = depth.shape
    device = depth.device

    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    pixel_coords = torch.stack((x_coords, y_coords), dim=-1)

    pixel_coords_homogeneous = torch.cat(
        (pixel_coords, torch.ones((height, width, 1), device=device)), dim=-1
    )
    pixel_coords_homogeneous = pixel_coords_homogeneous.permute(2, 0, 1).flatten(1)
    unprojected_points = torch.matmul(
        torch.inverse(camera_intrinsics), pixel_coords_homogeneous
    )
    unprojected_points = unprojected_points.view(batch_size, 3, height, width)
    unprojected_points = unprojected_points * depth
    return unprojected_points


@torch.jit.script
def flat_interpolate(
    flat_tensor: torch.Tensor,
    old: Tuple[int, int],
    new: Tuple[int, int],
    antialias: bool = True,
    mode: str = "bilinear",
) -> torch.Tensor:
    if old[0] == new[0] and old[1] == new[1]:
        return flat_tensor
    tensor = flat_tensor.view(flat_tensor.shape[0], old[0], old[1], -1).permute(
        0, 3, 1, 2
    )  # b c h w
    tensor_interp = F.interpolate(
        tensor,
        size=(new[0], new[1]),
        mode=mode,
        align_corners=False,
        antialias=antialias,
    )
    flat_tensor_interp = tensor_interp.view(
        flat_tensor.shape[0], -1, new[0] * new[1]
    ).permute(
        0, 2, 1
    )  # b (h w) c
    return flat_tensor_interp.contiguous()


@torch.jit.script
def dilate(image: torch.Tensor, kernel_size: int):
    device, dtype = image.device, image.dtype
    padding = kernel_size // 2
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32, device=image.device)
    dilated_image = F.conv2d(image.float(), kernel, padding=padding, stride=1)
    dilated_image = torch.where(
        dilated_image > 0,
        torch.tensor(1.0, device=device),
        torch.tensor(0.0, device=device),
    )
    return dilated_image.to(dtype)


@torch.jit.script
def erode(image: torch.Tensor, kernel_size: int):
    device, dtype = image.device, image.dtype
    padding = kernel_size // 2
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32, device=image.device)
    eroded_image = F.conv2d(image.float(), kernel, padding=padding, stride=1)
    eroded_image = torch.where(
        eroded_image == float(kernel_size * kernel_size),
        torch.tensor(1.0, device=device),
        torch.tensor(0.0, device=device),
    )
    return eroded_image.to(dtype)
