# This is a demo app for showcasing the trained model
# NOTE: requires package 'flask' run `pip install flask` to install

# allows type hinting
from __future__ import annotations

# standard library imports
from pathlib import Path
from typing import Any
import os
import io
import json
import base64
import threading

# third-party imports
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps
import numpy as np

# PyTorch imports
import torch
import torch.nn.functional as F

# local imports
from model.unidepthv1.unidepthv1 import UniDepthV1
from utils.visualization import colorize

# Constants and utility functions
# ---------- Configuration Section ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# DEVICE: prefer explicit env var if provided, otherwise default to cuda:0 when available
# Use robust parsing to avoid passing None into torch.device.
if torch.cuda.is_available():
    device_env = os.environ.get("DEVICE", "cuda:0")
    DEVICE = torch.device(device_env)
else:
    DEVICE = torch.device("cpu")

# Build checkpoint candidate list from env var if present; otherwise empty list.
DEFAULT_CHECKPOINT_CANDIDATES = []
if os.environ.get("CHECKPOINT_PATH"):
    DEFAULT_CHECKPOINT_CANDIDATES.append(PROJECT_ROOT / os.environ.get("CHECKPOINT_PATH"))

# Default camera / lidar params used by the demo UI
DEFAULT_CAMERA_PARAMS = {
    "fov_deg": 60.0,
}

DEFAULT_LIDAR_PARAMS = {
    "depth": 0.0,
    "mask": 0.0,
    "confidence": 0.0,
    "enabled": True,
    "note": "Default placeholders are set; monocular infer() path does not consume LiDAR tensors.",
}

DEFAULT_MODEL_CONFIG = {
    "model": {
        "name": "UniDepthV1",
        "pixel_encoder": {
            "name": "convnextv2_large",
            "use_checkpoint": True,
        },
        "pixel_decoder": {
            "name": "Decoder",
            "hidden_dim": 512,
            "dropout": 0.0,
            "depths": [3, 2, 1],
            "use_lidar_fusion": True,
            "lidar_fusion_type": "token",
        },
        "num_heads": 8,
        "expansion": 4,
    },
    "training": {
        "lr": 1e-4,
        "wd": 0.01,
        "losses": {},
    },
    "data": {
        "image_shape": [480, 640],
    },
}
# Add an explicit inference resize macro (height, width)
DEFAULT_INFERENCE_SHAPE = tuple(DEFAULT_MODEL_CONFIG["data"]["image_shape"])
# ----------- configuration ENDs ------------

# ensure the model checkpoint
def resolve_checkpoint_path() -> Path:
    """
    Return the first existing candidate checkpoint path.
    Raises FileNotFoundError with a helpful message if none found.
    """
    for candidate in DEFAULT_CHECKPOINT_CANDIDATES:
        if candidate.exists():
            return candidate
    checked = "\n".join(str(path) for path in DEFAULT_CHECKPOINT_CANDIDATES) if DEFAULT_CHECKPOINT_CANDIDATES else "(none)"
    raise FileNotFoundError(f"No default checkpoint found. Checked:\n{checked}")

# load checkpoint
def load_checkpoint(model: UniDepthV1, ckpt_path: Path, device: torch.device) -> None:
    """
    Load a checkpoint into the model. Accepts several checkpoint layouts and
    removes potential "module." prefixes from keys.
    """
    checkpoint = torch.load(str(ckpt_path), map_location = device)
    # find candidate state dict within checkpoint
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError(f"Unexpected checkpoint format: {ckpt_path}")

    # normalize keys (remove DataParallel 'module.' prefix)
    raw_sd = { key.replace("module.", ""): value for key, value in state_dict.items() }

    model_sd_keys = set(model.state_dict().keys())
    filtered_sd = {}
    mapped_keys = []
    skipped_keys = []

    for k, v in raw_sd.items():
        if k in model_sd_keys:
            filtered_sd[k] = v
            mapped_keys.append(k)
            continue

        # common case: checkpoint is a backbone-only state_dict (e.g. convnext) whose keys
        # need to be placed under model.pixel_encoder.* in UniDepth. Try a few sensible mappings.
        candidates = [
            "pixel_encoder." + k,
            "pixel_encoder.pixel_encoder." + k,  # if nested naming used
            "pixel_encoder.backbone." + k,
            "pixel_encoder.encoder." + k,
            k.replace("encoder.", "pixel_encoder.") if k.startswith("encoder.") else None,
        ]
        mapped = False
        for cand in [c for c in candidates if c]:
            if cand in model_sd_keys:
                filtered_sd[cand] = v
                mapped = True
                mapped_keys.append(f"{k} -> {cand}")
                break
        if mapped:
            continue

        # drop typical classifier head / unrelated normalization keys (e.g. head.weight)
        # or any key that didn't match the model
        skipped_keys.append(k)

    # load filtered state dict
    missing_keys, unexpected_keys = model.load_state_dict(filtered_sd, strict = False)

    # Minimal logging to help debugging (printed to stdout)
    if mapped_keys:
        print(f"[LOAD] mapped {len(mapped_keys)} checkpoint keys (examples): {mapped_keys[:6]}")
    if skipped_keys:
        print(f"[LOAD] skipped {len(skipped_keys)} checkpoint keys (examples): {skipped_keys[:6]}")
    if missing_keys:
        print(f"[LOAD] model missing keys after load (will be randomly init or kept): {missing_keys[:6]}")
    if unexpected_keys:
        print(f"[LOAD] unexpected keys still present: {unexpected_keys[:6]}")

# build default intrinsics based on image shape and FOV
def build_default_intrinsics(height: int, width: int, fov_deg: float) -> torch.Tensor:
    fx = width / (2.0 * np.tan(np.radians(fov_deg / 2.0)))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype = torch.float32,
    )

# Utility: (web app)
# encode a numpy array as a PNG image and then base64 encode it for
# JSON transmission
def encode_png_base64(array: np.ndarray) -> str:
    image = Image.fromarray(array.astype(np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format = "PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Create the Flask app and define routes
def create_app() -> Flask:

    # Load the model checkpoint once at startup
    checkpoint_path = resolve_checkpoint_path()
    model = UniDepthV1(DEFAULT_MODEL_CONFIG)
    load_checkpoint(model, checkpoint_path, DEVICE)
    model.to(DEVICE).eval()

    # Use a lock to ensure thread safety during inference
    infer_lock = threading.Lock()
    app = Flask(__name__, template_folder = "templates", static_folder = "static")

    # Define routes
    @app.get("/")
    def index() -> str:
        return render_template(
            "index.html",
            checkpoint_path = str(checkpoint_path),
            camera_defaults_json = json.dumps(DEFAULT_CAMERA_PARAMS),
            lidar_defaults_json = json.dumps(DEFAULT_LIDAR_PARAMS),
        )

    # API endpoint to get default parameters
    @app.get("/defaults")
    def defaults() -> Any:
        return jsonify(
            {
                "checkpoint": str(checkpoint_path),
                "camera": DEFAULT_CAMERA_PARAMS,
                "lidar": DEFAULT_LIDAR_PARAMS,
            }
        )

    # API endpoint to perform depth prediction
    @app.post("/predict")
    def predict() -> Any:

        # fetch and validate the uploaded image file
        if "image" not in request.files:
            return jsonify({"error": "no image uploaded"}), 400
        image_file = request.files["image"]
        if image_file.filename == "":
            return jsonify({"error": "empty filename"}), 400

        # convert the uploaded image to RGB and remember original size
        try:
            # Respect EXIF orientation (fix portrait photos rotated 90°)
            img = Image.open(image_file)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
        except Exception as e:
            return jsonify({"error": f"failed to read image: {e}"}), 400

        # remember original (height, width) to resize prediction back later
        orig_w, orig_h = img.size

        # Resize image to configured inference shape (width, height order for PIL)
        inf_h, inf_w = DEFAULT_INFERENCE_SHAPE
        img = img.resize((int(inf_w), int(inf_h)), Image.LANCZOS)

        # Convert to numpy and build a normalized torch tensor (1,3,H,W)
        rgb = np.array(img).astype(np.float32) / 255.0  # normalize to [0,1]
        # ImageNet mean/std for normalization
        IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype = np.float32).reshape(1, 1, 3)
        IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype = np.float32).reshape(1, 1, 3)
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous()  # (1,3,H,W)

        # build intrinsics using the resized image shape and move inputs to model device
        intrinsics = build_default_intrinsics(
            height = int(rgb.shape[0]),
            width = int(rgb.shape[1]),
            fov_deg = float(DEFAULT_CAMERA_PARAMS["fov_deg"]),
        ).to(DEVICE)

        rgb_tensor = rgb_tensor.to(DEVICE)

        # infer depth and prepare the response
        with torch.no_grad():
            with infer_lock:
                # model.infer expected to accept (B,3,H,W) rgb tensor and intrinsics
                out = model.infer(rgb_tensor, intrinsics = intrinsics)
                # expect output dict to contain 'depth' tensor in metres: (B,1,H,W)
                depth = out.get("depth", None)
                if depth is None:
                    return jsonify({"error": "model output missing 'depth'"}), 500
                depth_t = depth.squeeze(0).squeeze(0).detach().cpu()  # (H, W) on CPU

        # Print / log stats for debugging (min/max depth) to help confirm scale
        dmin = float(torch.min(depth_t).item())
        dmax = float(torch.max(depth_t).item())
        print(f"[INFER] raw depth min={dmin:.4f} max={dmax:.4f} (metres)")

        # Resize predicted depth back to original uploaded image size using torch (preserve metric scale)
        depth_resized_t = F.interpolate(
            depth_t.unsqueeze(0).unsqueeze(0),
            size = (int(orig_h), int(orig_w)),
            mode = "bilinear",
            align_corners = False,
        ).squeeze(0).squeeze(0)
        depth_resized = depth_resized_t.numpy().astype(np.float32)

        # clamp to reasonable range before visualization
        vmin = dmin
        vmax = dmax
        depth_resized = np.clip(depth_resized, vmin, vmax)

        # convert depth to RGB using the project's colorize helper
        # colorize should return an HxWx3 uint8 numpy array (or tensor)
        depth_rgb_u8 = colorize(depth_resized, vmin = vmin, vmax = vmax)
        # ensure numpy uint8 array for PNG encoding
        if isinstance(depth_rgb_u8, torch.Tensor):
            depth_rgb_u8 = depth_rgb_u8.detach().cpu().numpy()
        if depth_rgb_u8.dtype != np.uint8:
            # if float in [0,1], scale to 0-255
            if depth_rgb_u8.max() <= 1.0:
                depth_rgb_u8 = (depth_rgb_u8 * 255.0).astype(np.uint8)
            else:
                depth_rgb_u8 = depth_rgb_u8.astype(np.uint8)

        # encode RGB image as PNG into base64 (no matplotlib)
        buf = io.BytesIO()
        Image.fromarray(depth_rgb_u8).save(buf, format = "PNG")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # return result JSON (keys match demo/templates/index.html expectations)
        H_out, W_out = depth_resized.shape
        return jsonify(
            {
                "depth_png_base64": img_b64,
                "depth_min": float(dmin),
                "depth_max": float(dmax),
                "shape_hw": [int(H_out), int(W_out)],
            }
        )

    return app

# Class object - global app instance
app = create_app()

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 6480, debug = False)