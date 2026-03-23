import numpy as np
from scipy.signal import convolve2d
from skimage import color, img_as_float


# def _pad_to_same(conv_valid: np.ndarray, kernel_shape: tuple[int, int]) -> np.ndarray:
#     """Pad a valid convolution result to preserve the input spatial shape."""
#     ky, kx = kernel_shape
#     top = int(np.round((ky - 1) / 2.0))
#     bottom = ky - 1 - top
#     left = int(np.round((kx - 1) / 2.0))
#     right = kx - 1 - left
#     return np.pad(conv_valid, ((top, bottom), (left, right)), mode="edge")


# def _conv_same_like_matlab(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
#     """Apply 2D convolution with MATLAB-like valid+edge padding behavior."""
#     conv_valid = convolve2d(img, kernel, mode="valid")
#     return _pad_to_same(conv_valid, kernel.shape)


def calculate_filter_banks_old(img: np.ndarray) -> np.ndarray:
    """Compute 17-channel texture response tensor.

    This function ports ``calculateFilterBanks_old.m`` from the original
    Structured Depth Estimation toolbox.

    Parameters
    ----------
    img : numpy.ndarray
        Input RGB image of shape ``(H, W, 3)``.

    Returns
    -------
    numpy.ndarray
        Absolute filter responses with shape ``(H, W, 17)`` as ``float32``.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3).")

    rgb = img_as_float(img)
    ycbcr = color.rgb2ycbcr(rgb)

    matrix_conditioner = 0.2
    l3 = np.array([1.0, 2.0, 1.0]) / (128.0 * matrix_conditioner)
    e3 = np.array([-1.0, 0.0, 1.0])
    s3 = np.array([-1.0, 2.0, -1.0])

    nb1 = np.array(
        [
            [-100, -100, 0, 100, 100],
            [-100, -100, 0, 100, 100],
            [-100, -100, 0, 100, 100],
            [-100, -100, 0, 100, 100],
            [-100, -100, 0, 100, 100],
        ],
        dtype=float,
    ) / 2000.0
    nb2 = np.array(
        [
            [-100, 32, 100, 100, 100],
            [-100, -78, 92, 100, 100],
            [-100, -100, 0, 100, 100],
            [-100, -100, -92, 78, 100],
            [-100, -100, -100, -32, 100],
        ],
        dtype=float,
    ) / 2000.0
    nb3 = -nb2.T
    nb4 = -nb1.T
    nb5 = -np.flipud(nb3)
    nb6 = nb5.T

    img_y = ycbcr[:, :, 0]
    cb = ycbcr[:, :, 1]
    cr = ycbcr[:, :, 2]

    kernels = [
        np.outer(l3, l3),
        np.outer(l3, e3),
        np.outer(l3, s3),
        np.outer(e3, l3),
        np.outer(e3, e3),
        np.outer(e3, s3),
        np.outer(s3, l3),
        np.outer(s3, e3),
        np.outer(s3, s3),
    ]

    H = np.zeros((img.shape[0], img.shape[1], 17), dtype=float)
    for i, ker in enumerate(kernels):
        H[:, :, i] = convolve2d(img_y, ker, mode="valid")
    
    H[:, :, 9] = convolve2d(cb, np.outer(l3, l3), mode="valid")
    H[:, :, 10] = convolve2d(cr, np.outer(l3, l3), mode="valid")

    nb_kernels = [nb1, nb2, nb3, nb4, nb5, nb6]
    for i, ker in enumerate(nb_kernels, start=11):
        H[:, :, i] = convolve2d(img_y, ker, mode="valid")

    return np.abs(H, dtype=float)
