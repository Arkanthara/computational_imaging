from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter
from skimage import color, img_as_float32, transform

from features.calculateFilterBanks_old import calculate_filter_banks_old
from features.rgb2hsi import rgb2hsi

if TYPE_CHECKING:
    from ssiDepthOptions import SSIDepthTrainOptions


def _as_float_rgb(image: np.ndarray) -> np.ndarray:
    """Convert image to float32 RGB in ``[0, 1]``."""
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.ndim != 3:
        raise ValueError("Expected an image with shape (H, W) or (H, W, C).")
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.shape[2] != 3:
        raise ValueError("Expected RGB image with 3 channels.")
    return img_as_float32(image)


def _resize_image(image: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Resize image with anti-aliasing and preserved range."""
    return transform.resize(
        image,
        out_hw,
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)


def _dark_channel(rgb_image: np.ndarray, radius: int = 7) -> np.ndarray:
    """Compute a dark-channel prior map.

    Parameters
    ----------
    rgb_image : numpy.ndarray
        Float RGB image in ``[0, 1]`` with shape ``(H, W, 3)``.
    radius : int, default=7
        Radius of the square structuring element.

    Returns
    -------
    numpy.ndarray
        Dark-channel map with shape ``(H, W)``.
    """
    local_min = np.min(rgb_image, axis=2)
    win = 2 * int(radius) + 1
    return minimum_filter(local_min, size=(win, win), mode="nearest").astype(np.float32)


def _smooth_channels(chns: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian smoothing to channel tensors."""
    if sigma <= 0:
        return chns
    return gaussian_filter(chns, sigma=(sigma, sigma, 0.0), mode="nearest").astype(np.float32)


def ssiDepthChns(
    image: np.ndarray,
    opts: "SSIDepthTrainOptions",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int, int, int]:
    """Compute channels used by the SSI depth detector.

    Parameters
    ----------
    image : numpy.ndarray
        Input image of shape ``(H, W, 3)`` or ``(H, W)``.
    opts : SSIDepthTrainOptions
        Model options object.

    Returns
    -------
    tuple
        ``(chnsReg, colsReg, chnsSim, colsSim, nChnFtrs, nColChnFtrs,
        nSimFtrs, nColSimFtrs, nChns)``.
    """
    image = _as_float_rgb(image)

    res_h, res_w = opts.imResize
    im_width = opts.imWidth
    shrink = opts.shrink
    shrink_col = opts.shrinkCol

    chn_smooth = opts.chnSmooth
    sim_smooth = opts.simSmooth

    resized = _resize_image(image, (res_h, res_w))

    sh_h = max(1, res_h // shrink)
    sh_w = max(1, res_w // shrink)

    rgb_shrink = _resize_image(resized, (sh_h, sh_w))
    luv_shrink = _resize_image(color.rgb2luv(resized).astype(float), (sh_h, sh_w))
    hsi_shrink = _resize_image(rgb2hsi(resized), (sh_h, sh_w))

    prior = np.linspace(0.0, 1.0, sh_h, dtype=np.float32)[:, None]
    prior = np.repeat(prior, sh_w, axis=1)[:, :, None]

    channel_blocks: list[np.ndarray] = [prior, rgb_shrink, hsi_shrink, luv_shrink]

    for s in (1, 2):
        scale_h = max(1, int(round(res_h / s)))
        scale_w = max(1, int(round(res_w / s)))
        scaled = _resize_image(resized, (scale_h, scale_w))

        laws = calculate_filter_banks_old(scaled)
        dark = _dark_channel(scaled)

        laws_resampled = _resize_image(laws, (sh_h, sh_w))
        dark_resampled = _resize_image(dark, (sh_h, sh_w))[:, :, None]

        channel_blocks.append(laws_resampled)
        channel_blocks.append(dark_resampled)

    chns = np.concatenate(channel_blocks, axis=2).astype(np.float32)

    chn_sigma = chn_smooth / float(shrink)
    sim_sigma = sim_smooth / float(shrink)
    chns_reg = _smooth_channels(chns, chn_sigma)
    chns_sim = _smooth_channels(chns, sim_sigma)

    col_h = max(1, res_h // shrink_col)
    col_w = max(1, res_w // shrink_col)
    cols_reg = _resize_image(chns_reg, (col_h, col_w))
    cols_sim = _resize_image(chns_sim, (col_h, col_w))

    n_cells = opts.nCells
    n_cells_col = opts.nCellsCol
    n_chns = int(chns_reg.shape[2])

    sig_ftr_size = (im_width // shrink) ** 2
    col_ftr_size = (res_h // shrink_col) * (im_width // shrink_col)
    n_chn_ftrs = sig_ftr_size * n_chns
    n_col_chn_ftrs = col_ftr_size * n_chns

    n_patch_prop = max(1, int(round(res_h / im_width)))
    n_vertical = n_cells_col * n_patch_prop
    n_col_sim_ftrs = ((n_vertical * n_cells_col) * (n_vertical * n_cells_col - 1) // 2) * n_chns
    n_sim_ftrs = ((n_cells * n_cells) * (n_cells * n_cells - 1) // 2) * n_chns

    return (
        chns_reg,
        cols_reg,
        chns_sim,
        cols_sim,
        int(n_chn_ftrs),
        int(n_col_chn_ftrs),
        int(n_sim_ftrs),
        int(n_col_sim_ftrs),
        n_chns,
    )
