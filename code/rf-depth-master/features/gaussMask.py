import numpy as np


def gauss_mask(w: int, sigma: float) -> np.ndarray:
    """Create a normalized 2D Gaussian mask.

    Parameters
    ----------
    w : int
        Radius of the kernel. The output size is ``(2*w + 1, 2*w + 1)``.
    sigma : float
        Standard deviation of the Gaussian.

    Returns
    -------
    numpy.ndarray
        Normalized Gaussian kernel as ``float32``.
    """
    coords = np.arange(-w, w + 1)
    x, y = np.meshgrid(coords, coords, indexing="xy")
    kernel = np.exp(-(x**2 + y**2) / (2.0 * sigma**2)) / (2.0 * np.pi)
    kernel /= np.sum(kernel)
    return kernel
