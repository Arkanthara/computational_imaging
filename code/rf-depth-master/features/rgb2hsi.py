import numpy as np
from skimage import img_as_float


def rgb2hsi(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to HSI.

    Parameters
    ----------
    rgb : numpy.ndarray
        Input RGB image of shape ``(H, W, 3)``.

    Returns
    -------
    numpy.ndarray
        HSI image of shape ``(H, W, 3)`` with channels in ``[0, 1]``.
    """
    rgb = img_as_float(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3).")

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    theta = np.arccos(np.clip(num / (den + np.finfo(np.float32).eps), -1.0, 1.0))

    h = theta.copy()
    mask = b > g
    h[mask] = 2.0 * np.pi - h[mask]
    h /= 2.0 * np.pi

    min_rgb = np.minimum(np.minimum(r, g), b)
    den = r + g + b
    den = np.where(den == 0.0, np.finfo(np.float32).eps, den)
    s = 1.0 - 3.0 * min_rgb / den
    h[s == 0.0] = 0.0

    i = (r + g + b) / 3.0
    return np.stack((h, s, i), axis=-1)
