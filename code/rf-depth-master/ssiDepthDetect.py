"""Depth inference for the Python SSI depth pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import img_as_float32, transform

from ssiDepthChns import ssiDepthChns

if TYPE_CHECKING:
    from ssiDepthOptions import SSIDepthTrainOptions


FEATURE_NAMES = (
    "prior",
    "brightness_inverse",
    "saturation",
    "texture",
    "dark_channel",
    "vertical_gradient",
)

DEFAULT_WEIGHTS = np.array([0.34, 0.24, 0.12, 0.10, 0.14, 0.06], dtype=np.float32)
DEFAULT_BIAS = 0.0


def _resize_image(image: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Resize image while preserving value range.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.
    out_hw : tuple of int
        Output ``(height, width)``.

    Returns
    -------
    numpy.ndarray
        Resized image as ``float32``.
    """
    return transform.resize(
        image,
        out_hw,
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)


def compute_depth_features(image: np.ndarray, opts: "SSIDepthTrainOptions") -> np.ndarray:
    """Compute pixel-wise depth features.

    Parameters
    ----------
    image : numpy.ndarray
        Input image of shape ``(H, W, 3)`` or ``(H, W)``.
    opts : SSIDepthTrainOptions
        Detector options object.

    Returns
    -------
    numpy.ndarray
        Feature map of shape ``(H, W, F)`` with ``F=6``.
    """
    image = img_as_float32(image)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    chns_reg, _, _, _, *_ = ssiDepthChns(image, opts)

    prior = chns_reg[:, :, 0]
    rgb = chns_reg[:, :, 1:4]
    hsi = chns_reg[:, :, 4:7]

    laws_start = 10
    laws_end = min(27, chns_reg.shape[2])
    dark_idx = min(27, chns_reg.shape[2] - 1)

    brightness_inverse = 1.0 - np.mean(rgb, axis=2)
    saturation = hsi[:, :, 1]
    texture = np.mean(np.abs(chns_reg[:, :, laws_start:laws_end]), axis=2)
    dark_channel = chns_reg[:, :, dark_idx]

    grad_y = np.abs(np.gradient(np.mean(rgb, axis=2), axis=0))

    return np.stack(
        [prior, brightness_inverse, saturation, texture, dark_channel, grad_y],
        axis=-1,
    ).astype(np.float32)


def _depth_from_linear_features(features: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """Project feature tensor to a normalized depth map.

    Parameters
    ----------
    features : numpy.ndarray
        Feature tensor of shape ``(H, W, F)``.
    weights : numpy.ndarray
        Linear weights of shape ``(F,)``.
    bias : float
        Linear bias term.

    Returns
    -------
    numpy.ndarray
        Depth map in approximate range ``[1, 80]``.
    """
    raw = np.tensordot(features, weights, axes=([2], [0])) + np.float32(bias)
    raw = gaussian_filter(raw.astype(np.float32), sigma=1.2, mode="nearest")

    mn = float(np.min(raw))
    mx = float(np.max(raw))
    if mx - mn < 1e-6:
        return np.full(raw.shape, 40.0, dtype=np.float32)

    depth = (raw - mn) / (mx - mn)
    return (1.0 + 79.0 * depth).astype(np.float32)


def ssiDepthDetect(image: np.ndarray, model: Mapping[str, Any]) -> np.ndarray:
    """Estimate a dense depth map from a single image.

    Parameters
    ----------
    image : numpy.ndarray
        RGB or grayscale image.
    model : Mapping[str, Any]
        Model dictionary produced by ``ssiDepthTrain``.

    Returns
    -------
    numpy.ndarray
        Estimated depth map with shape ``(H, W)``.
    """
    from ssiDepthOptions import SSIDepthTrainOptions

    opts = SSIDepthTrainOptions.from_dict(model["opts"])

    image_f = img_as_float32(image)
    if image_f.ndim == 2:
        image_f = np.stack([image_f, image_f, image_f], axis=-1)

    orig_h, orig_w = image_f.shape[:2]
    res_h, res_w = opts.imResize
    resized = _resize_image(image_f, (res_h, res_w))

    features = compute_depth_features(resized, opts)

    detector = model.get("detector", {})
    weights = np.asarray(detector.get("weights", DEFAULT_WEIGHTS), dtype=np.float32)
    bias = float(detector.get("bias", DEFAULT_BIAS))

    if weights.shape[0] != features.shape[2]:
        weights = DEFAULT_WEIGHTS
        bias = DEFAULT_BIAS

    depth_small = _depth_from_linear_features(features, weights, bias)
    depth = _resize_image(depth_small, (orig_h, orig_w))

    return gaussian_filter(depth, sigma=2.0, mode="nearest").astype(np.float32)
