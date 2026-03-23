"""Evaluation metrics for the Python SSI depth pipeline."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.io import loadmat
from skimage import transform


def _safe_load_depth(path: Path) -> np.ndarray:
    """Load ``depth`` variable from MATLAB file."""
    data = loadmat(path, simplify_cells=True)
    if "depth" not in data:
        raise KeyError(f"Missing 'depth' variable in {path}.")
    return np.asarray(data["depth"], dtype=np.float32)


def _resize_like(src: np.ndarray, ref_shape: tuple[int, int]) -> np.ndarray:
    """Resize depth map to a reference shape."""
    return transform.resize(
        src,
        ref_shape,
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)


def _extract_make3d_id(stem: str) -> str:
    """Extract Make3D numeric image identifier from a file stem."""
    m = re.search(r"(\d+)$", stem)
    return m.group(1) if m else stem


def ssiDepthEval(depth_path: str | Path, gt_path: str | Path, model: Mapping[str, Any]) -> tuple[float, float, float]:
    """Evaluate depth estimation with relative, log10, and RMSE metrics.

    Parameters
    ----------
    depth_path : str or pathlib.Path
        Directory with predicted ``.mat`` depth files.
    gt_path : str or pathlib.Path
        Directory with ground-truth ``.mat`` files.
    model : Mapping[str, Any]
        Model dictionary from ``ssiDepthTrain``.

    Returns
    -------
    tuple of float
        ``(rel, lg10, rmse)`` averaged over all files.
    """
    depth_path = Path(depth_path)
    gt_path = Path(gt_path)

    dataset = str(model["opts"].get("dataSet", "make3d")).lower()

    pred_files = sorted(depth_path.glob("*.mat"))
    if not pred_files:
        raise FileNotFoundError(f"No prediction .mat files found in {depth_path}.")

    rel_all: list[np.ndarray] = []
    lg10_all: list[np.ndarray] = []
    rmse_all: list[np.ndarray] = []

    for pred_file in pred_files:
        pred = _safe_load_depth(pred_file)

        if dataset == "make3d":
            idx = _extract_make3d_id(pred_file.stem)
            gt_file = gt_path / f"depth_sph_corr-{idx}.mat"
            gt = loadmat(gt_file, simplify_cells=True)
            laser_depth = np.asarray(gt["Position3DGrid"], dtype=np.float32)
            if laser_depth.ndim == 3 and laser_depth.shape[2] >= 4:
                laser_depth = laser_depth[:, :, 3]
            laser_depth = np.clip(laser_depth, 1e-3, 80.0)
        else:
            gt_file = gt_path / f"{pred_file.stem}.mat"
            gt = loadmat(gt_file, simplify_cells=True)
            laser_depth = np.asarray(gt.get("depth"), dtype=np.float32)
            laser_depth = np.clip(laser_depth, 1e-3, None)

        pred = _resize_like(pred, laser_depth.shape)
        pred = np.clip(pred, 1e-3, None)

        rel_all.append(np.abs((pred - laser_depth) / laser_depth).ravel())
        lg10_all.append(np.abs(np.log10(pred) - np.log10(laser_depth)).ravel())
        rmse_all.append((pred - laser_depth).ravel())

    rel = float(np.mean(np.concatenate(rel_all)))
    lg10 = float(np.mean(np.concatenate(lg10_all)))
    rmse = float(np.sqrt(np.mean(np.concatenate(rmse_all) ** 2)))

    return rel, lg10, rmse
