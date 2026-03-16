"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)

Simplified camera module - includes Pinhole camera needed for UniDepthV1 training.
"""

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from .coordinate import coords_grid
from .misc import recursive_to, squeeze_list


def invert_pinhole(K):
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    K_inv = torch.zeros_like(K)
    K_inv[..., 0, 0] = 1.0 / fx
    K_inv[..., 1, 1] = 1.0 / fy
    K_inv[..., 0, 2] = -cx / fx
    K_inv[..., 1, 2] = -cy / fy
    K_inv[..., 2, 2] = 1.0
    return K_inv


class Camera:
    """
    Base camera class. Use Pinhole for typical training data.
    """

    def __init__(self, params=None, K=None):
        if params.ndim == 1:
            params = params.unsqueeze(0)

        if K is None:
            K = (
                torch.eye(3, device=params.device, dtype=params.dtype)
                .unsqueeze(0)
                .repeat(params.shape[0], 1, 1)
            )
            K[..., 0, 0] = params[..., 0]
            K[..., 1, 1] = params[..., 1]
            K[..., 0, 2] = params[..., 2]
            K[..., 1, 2] = params[..., 3]

        self.params = params
        self.K = K
        self.overlap_mask = None
        self.projection_mask = None

    def project(self, xyz):
        raise NotImplementedError

    def unproject(self, uv):
        raise NotImplementedError

    def get_projection_mask(self):
        return self.projection_mask

    def get_overlap_mask(self):
        return self.overlap_mask

    def reconstruct(self, depth):
        id_coords = coords_grid(
            1, depth.shape[-2], depth.shape[-1], device=depth.device
        )
        rays = self.unproject(id_coords)
        return (
            rays / rays[:, -1:].clamp(min=1e-4) * depth.clamp(min=1e-4)
        )

    def resize(self, factor):
        self.K[..., :2, :] *= factor
        self.params[..., :4] *= factor
        return self

    def to(self, device, non_blocking=False):
        self.params = self.params.to(device, non_blocking=non_blocking)
        self.K = self.K.to(device, non_blocking=non_blocking)
        return self

    def get_rays(self, shapes, noisy=False):
        b, h, w = shapes
        uv = coords_grid(1, h, w, device=self.K.device, noisy=noisy)
        rays = self.unproject(uv)
        return rays / torch.norm(rays, dim=1, keepdim=True).clamp(min=1e-4)

    def get_pinhole_rays(self, shapes, noisy=False):
        b, h, w = shapes
        uv = coords_grid(b, h, w, device=self.K.device, homogeneous=True, noisy=noisy)
        rays = (invert_pinhole(self.K) @ uv.reshape(b, 3, -1)).reshape(b, 3, h, w)
        return rays / torch.norm(rays, dim=1, keepdim=True).clamp(min=1e-4)

    def flip(self, H, W, direction="horizontal"):
        new_cx = (
            W - self.params[:, 2] if direction == "horizontal" else self.params[:, 2]
        )
        new_cy = H - self.params[:, 3] if direction == "vertical" else self.params[:, 3]
        self.params = torch.stack(
            [self.params[:, 0], self.params[:, 1], new_cx, new_cy], dim=1
        )
        self.K[..., 0, 2] = new_cx
        self.K[..., 1, 2] = new_cy
        return self

    def clone(self):
        return deepcopy(self)

    def crop(self, left, top, right=None, bottom=None):
        self.K[..., 0, 2] -= left
        self.K[..., 1, 2] -= top
        self.params[..., 2] -= left
        self.params[..., 3] -= top
        return self

    def _pad_params(self):
        if self.params.shape[1] < 16:
            padding = torch.zeros(
                16 - self.params.shape[1],
                device=self.params.device,
                dtype=self.params.dtype,
            )
            padding = padding.unsqueeze(0).repeat(self.params.shape[0], 1)
            return torch.cat([self.params, padding], dim=1)
        return self.params

    @property
    def device(self):
        return self.K.device

    @property
    def hfov(self):
        return 2 * torch.atan(self.params[..., 2] / self.params[..., 0])

    @property
    def vfov(self):
        return 2 * torch.atan(self.params[..., 3] / self.params[..., 1])

    @property
    def max_fov(self):
        return 150.0 / 180.0 * np.pi, 150.0 / 180.0 * np.pi


class Pinhole(Camera):
    def __init__(self, params=None, K=None):
        assert params is not None or K is not None
        if params is None:
            params = torch.stack(
                [K[..., 0, 0], K[..., 1, 1], K[..., 0, 2], K[..., 1, 2]], dim=-1
            )
        super().__init__(params=params, K=K)

    def get_rays(self, shapes, noisy=False):
        """
        Override base class to correctly handle a batched Pinhole camera (K=[B,3,3]).
        The base Camera.get_rays uses coords_grid(1,...) which is designed for the
        BatchCamera.unproject iteration pattern. For a single Pinhole with K=[B,3,3]
        we must use get_pinhole_rays which generates per-sample coordinate grids.
        """
        
        return self.get_pinhole_rays(shapes, noisy=noisy)

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def project(self, pcd):
        b, _, h, w = pcd.shape
        pcd_flat = pcd.reshape(b, 3, -1)  # [B, 3, H*W]
        cam_coords = self.K @ pcd_flat
        pcd_proj = cam_coords[:, :2] / cam_coords[:, -1:].clamp(min=0.01)
        pcd_proj = pcd_proj.reshape(b, 2, h, w)
        invalid = (
            (pcd_proj[:, 0] >= 0)
            & (pcd_proj[:, 0] < w)
            & (pcd_proj[:, 1] >= 0)
            & (pcd_proj[:, 1] < h)
        )
        self.projection_mask = (~invalid).unsqueeze(1)
        return pcd_proj

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def unproject(self, uv):
        b, _, h, w = uv.shape
        uv_flat = uv.reshape(b, 2, -1)  # [B, 2, H*W]
        uv_homogeneous = torch.cat(
            [uv_flat, torch.ones(b, 1, h * w, device=uv.device)], dim=1
        )  # [B, 3, H*W]
        K_inv = torch.inverse(self.K.float())
        xyz = K_inv @ uv_homogeneous
        xyz = xyz / xyz[:, -1:].clip(min=1e-4)
        xyz = xyz.reshape(b, 3, h, w)
        self.unprojection_mask = xyz[:, -1:] > 1e-4
        return xyz

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def reconstruct(self, depth):
        b, _, h, w = depth.shape
        uv = coords_grid(b, h, w, device=depth.device)
        xyz = self.unproject(uv) * depth.clip(min=0.0)
        return xyz