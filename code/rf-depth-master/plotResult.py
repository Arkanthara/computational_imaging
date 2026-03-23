"""Plotting helper for comparing depth predictions with ground truth."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from skimage import io, transform


def _resize_like(src: np.ndarray, ref_shape: tuple[int, int]) -> np.ndarray:
    """Resize an array to match a reference shape."""
    return transform.resize(
        src,
        ref_shape,
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)


def plot_result(
    image_path: str | Path,
    predicted_depth_mat: str | Path,
    gt_depth_mat: str | Path,
    save_path: str | Path | None = None,
) -> None:
    """Plot RGB input, predicted depth, and ground truth depth.

    Parameters
    ----------
    image_path : str or pathlib.Path
        Path to RGB image.
    predicted_depth_mat : str or pathlib.Path
        Path to ``.mat`` file containing variable ``depth``.
    gt_depth_mat : str or pathlib.Path
        Path to ``.mat`` ground-truth file.
    save_path : str or pathlib.Path, optional
        If provided, write a figure to this path.
    """
    image = io.imread(Path(image_path))
    pred = loadmat(Path(predicted_depth_mat), simplify_cells=True)["depth"].astype(np.float32)

    gt_raw = loadmat(Path(gt_depth_mat), simplify_cells=True)
    if "Position3DGrid" in gt_raw:
        gt = np.asarray(gt_raw["Position3DGrid"], dtype=np.float32)
        if gt.ndim == 3 and gt.shape[2] >= 4:
            gt = gt[:, :, 3]
    else:
        gt = np.asarray(gt_raw["depth"], dtype=np.float32)

    pred = _resize_like(pred, gt.shape)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[1].imshow(pred, cmap="viridis")
    axes[1].set_title("Predicted")
    axes[2].imshow(gt, cmap="viridis")
    axes[2].set_title("Ground Truth")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(Path(save_path), dpi=150, bbox_inches="tight")
    plt.show()
