import numpy as np
import matplotlib.pyplot as plt
from pypher.pypher import psf2otf
from scipy.signal import convolve2d
import argparse
import skimage as sk
import time


def read_image(name: str = "img/cameraman.jpg") -> np.ndarray:
    img = sk.io.imread(name, as_gray=True)
    return img


def print_range(img: np.ndarray):
    print("========================")
    print("     IMAGE RANGE")
    print("========================")
    print(f"dtype: {img.dtype}")
    print(f"Min: {img.min()}")
    print(f"Max: {img.max()}")
    print(f"Range: {img.max() - img.min()}")


def print_image(
    img: np.ndarray,
    title: str = "Image",
    magnitude: bool = False,
    phase: bool = False,
    log: bool = False,
    figsize: tuple = (5, 5),
    axis: bool = False,
) -> np.ndarray:

    plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()

    F_img = np.fft.fft2(img)

    if magnitude:
        if axis:
            F_img_M = np.fft.fftshift(np.abs(F_img))

            plt.figure(figsize=figsize)
            plt.title("Magnitude")
            if log:
                tmp = np.log(F_img_M + 1)
                plt.imshow(
                    tmp,
                    cmap="gray",
                    extent=[
                        -tmp.shape[1] / 2.0,
                        tmp.shape[1] / 2.0,
                        tmp.shape[0] / 2.0,
                        -tmp.shape[0] / 2.0,
                    ],
                )
            else:
                plt.imshow(F_img_M, cmap="gray")
            plt.show()

        else:
            F_img_M = np.fft.fftshift(np.abs(F_img))

            plt.figure(figsize=figsize)
            plt.title("Magnitude")
            if log:
                plt.imshow(np.log(F_img_M + 1), cmap="gray")
            else:
                plt.imshow(F_img_M, cmap="gray")
            plt.axis("off")
            plt.show()

    if phase:
        F_img_P = np.fft.fftshift(np.arctan2(F_img.imag, F_img.real))

        plt.figure(figsize=figsize)
        plt.title("Phase")
        plt.imshow(F_img_P, cmap="gray")
        plt.axis("off")
        plt.show()


def normalize(img: np.ndarray, target: float = 1.0) -> np.ndarray:
    return (img - img.min()) * target / (img.max() - img.min())


def add_noise(img: np.ndarray, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    noise = np.random.normal(loc=mean, scale=std, size=img.shape)
    img_noised = img + noise
    return np.clip(img_noised, a_min=0, a_max=255)


def MSE(img_1: np.ndarray, img_2: np.ndarray) -> float:
    return np.mean((img_1 - img_2) ** 2)


# Fix the PSNR function first
def PSNR(img_1: np.ndarray, img_2: np.ndarray) -> float:
    mse = MSE(img_1, img_2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    return 10 * np.log10((max_pixel**2) / mse)


# Fix the filterFT function
def filterFT(
    img: np.ndarray,
    h: np.ndarray,
    inv_filter: bool = False,
    wiener: bool = False,
    K: float = 0.01,
    debug: bool = False,
) -> np.ndarray:
    F_img = np.fft.fft2(img)
    # H = np.fft.fft2(h, s=img.shape)
    H = psf2otf(h, shape=img.shape)
    if debug:
        print_image(np.log(np.fft.fftshift(np.abs(H)) + 1))
        print_image(np.log(np.fft.fftshift(np.abs(F_img)) + 1))
        print_image(np.clip(np.fft.ifft2(F_img).real, 0, 255))
    if inv_filter:
        # H[H == 0] = 1e-10
        # H_inv = np.linalg.inv(H)
        H_inv = 1 / (H + 1e-10)
        F_img_filtered = F_img * H_inv
        if debug:
            print_image(np.log(np.fft.fftshift(np.abs(F_img_filtered) + 1)))
    elif wiener:
        # H_wiener = (1 / (H + 1e-10)) * (np.abs(H) ** 2 / (np.abs(H) ** 2 + K + 1e-10))
        H_wiener = np.conj(H) / (np.abs(H) ** 2 + K + 1e-10)
        F_img_filtered = F_img * H_wiener
    else:
        F_img_filtered = F_img * H
    img_filtered = np.fft.ifft2(F_img_filtered).real
    return img_filtered


def filter(img: np.ndarray, h: np.ndarray) -> np.ndarray:
    # Here, we ask for symmetric padding to apply filter to avoid vignette artifact
    return convolve2d(img, h, mode="same")


def gaussianKernel(std: float, size: int = 101) -> np.ndarray:
    x = np.arange(-(size // 2), size // 2 + 1)
    kernel = np.exp(-(x**2) / (2 * std**2))
    kernel = np.outer(kernel, kernel)
    kernel /= kernel.sum()
    return kernel


def grad_l2(A, x, b):
    return A.T @ A @ x - A.T @ b


def residual_l2(A, x, b):
    return 0.5 * np.linalg.norm(A @ x - b) ** 2


def run_gd(
    A,
    b,
    step_size: float = 1e-4,
    num_iters: int = 1500,
    grad_fn=grad_l2,
    residual=residual_l2,
):
    # Create random x
    x = np.random.rand(A.shape[1], 1)
    losses = []
    time_list = []
    for i in range(num_iters):
        grad = grad_fn(A, x, b)
        x = x - step_size * grad
        losses.append(residual(A, x, b))
        time_list.append(time.time())
    return x, losses, time_list


def run_sgd(
    A,
    b,
    step_size: float = 1e-4,
    num_iters: int = 1500,
    batch_size: int = 32,
    grad_fn=grad_l2,
    residual=residual_l2,
):
    x = np.random.rand(A.shape[1], 1)
    losses = []
    time_list = []
    for i in range(num_iters):
        idx = np.random.choice(A.shape[0], batch_size, replace=False)
        A_batch = A[idx]
        b_batch = b[idx]
        grad = grad_fn(A_batch, x, b_batch)
        x = x - step_size * grad
        losses.append(residual(A, x, b))
        time_list.append(time.time())
    return x, losses, time_list


def tasks(task: int = 1, subtask: int = 1, figsize: tuple[int, int] = (8, 8), original: bool = False):
    cameraman = sk.io.imread(
        "img/tangled_small.jpg",
        as_gray=True,
    )
    cameraman = sk.util.img_as_float(cameraman)
    print_range(cameraman)
    # TASK 1
    # 1.1 Low-pass filtering in frequency domain (using np.convolve2d !)
    if task == 1 and subtask == 1:
        index = 1
        plt.figure(figsize=figsize)
        for i in [0.1, 1, 10]:
            # Create filter
            plt.subplot(3, 3, index)
            h = gaussianKernel(i)
            plt.imshow(h, cmap="gray")
            plt.title(f"Gaussian kernel, $\\sigma = {i}$")
            plt.axis("off")
            index += 1

            # Apply filter in frequency domain -> 2D convolution
            plt.subplot(3, 3, index)
            cameraman_gaussian_blur = filter(cameraman, h)
            plt.imshow(cameraman_gaussian_blur, cmap="gray")
            plt.title(f"$\\sigma = {i}$, frequency domain")
            plt.axis("off")
            index += 1

            # Apply filter in fourier domain -> multiplication
            plt.subplot(3, 3, index)
            cameraman_gaussian_blur_ft = filterFT(cameraman, h)
            plt.imshow(cameraman_gaussian_blur_ft, cmap="gray")
            plt.title(f"$\\sigma = {i}$, Fourier domain")
            plt.axis("off")
            index += 1
        return plt.gcf()

    elif task == 1 and subtask == 2:
        # 1.2 High-pass filtering (HighPass = Img - LowPass(Img))
        index = 1
        plt.figure(figsize=figsize)
        for i in [0.1, 1, 10]:
            # Create filter
            plt.subplot(3, 3, index)
            h = gaussianKernel(i)
            plt.imshow(h, cmap="gray")
            plt.title(f"Gaussian kernel, $\\sigma = {i}$")
            plt.axis("off")
            index += 1

            # Apply filter in frequency domain -> 2D convolution
            plt.subplot(3, 3, index)
            cameraman_gaussian_blur = filter(cameraman, h)
            print_range(cameraman_gaussian_blur)
            plt.imshow(cameraman - cameraman_gaussian_blur, cmap="gray")
            plt.title(f"$\\sigma = {i}$, frequency domain")
            plt.axis("off")
            index += 1

            # Apply filter in fourier domain -> multiplication
            plt.subplot(3, 3, index)
            cameraman_gaussian_blur_ft = filterFT(cameraman, h)
            plt.imshow(cameraman - cameraman_gaussian_blur_ft, cmap="gray")
            plt.title(f"$\\sigma = {i}$, Fourier domain")
            plt.axis("off")
            index += 1
        return plt.gcf()

    # TASK 2
    # Blur image
    elif task == 2 and subtask == 1:
        cameraman_blured = filterFT(cameraman, gaussianKernel(5))
        if original:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(cameraman, cmap="gray")
            plt.axis("off")
            plt.title("Original image")
            plt.subplot(1, 2, 2)
            plt.imshow(cameraman_blured, cmap="gray")
            plt.axis("off")
            plt.title("Blured image")
            return plt.gcf()
        index = 1
        plt.figure(figsize=figsize)
        for i in [0, 0.001, 0.01, 0.1]:
            plt.subplot(2, 2, index)
            # Add noise
            cameraman_noised = add_noise(cameraman_blured, std=i)

            # Inverse filter in Fourrier domain
            h = gaussianKernel(5)
            cameraman_inv_filter = filterFT(cameraman_noised, h, inv_filter=True)
            plt.imshow(cameraman_inv_filter, cmap="gray")
            plt.title(f"Inverse filtering with noise $\\sigma = {i}$")
            plt.axis("off")
            index += 1
        return plt.gcf()

    elif task == 2 and subtask == 2:
        cameraman_blured = filterFT(cameraman, gaussianKernel(5))
        if original:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(cameraman, cmap="gray")
            plt.axis("off")
            plt.title("Original image")
            plt.subplot(1, 2, 2)
            plt.imshow(cameraman_blured, cmap="gray")
            plt.axis("off")
            plt.title("Blured image")
            return plt.gcf()
        index = 1
        plt.figure(figsize=figsize)
        for i in [0, 0.001, 0.01, 0.1]:
            plt.subplot(2, 2, index)

            # Add noise
            cameraman_noised = add_noise(cameraman_blured, std=i)
            # Wiener filter in Fourrier domain
            h = gaussianKernel(5)
            cameraman_inv_filter = filterFT(
                cameraman_noised, h, wiener=True, K=i / np.mean(cameraman_noised)
            )
            plt.imshow(cameraman_inv_filter, cmap="gray")
            plt.title(f"Wiener filtering with noise $\\sigma = {i}$")
            plt.axis("off")
            index += 1

        return plt.gcf()

    # TASK 3: Gradient descent
    elif task == 3:
        # Classical gradient descent
        A = np.random.rand(500, 100)
        b = np.random.rand(500, 1)
        x, losses, time_list = run_gd(A, b)
        plt.figure()
        plt.plot(losses)
        plt.title("Gradient Descent Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()
        plt.figure()
        plt.plot(time_list)
        plt.title("Time taken for gradient descent step")
        plt.xlabel("step")
        plt.ylabel("time")
        plt.show()

        # Stochastic gradient descent
        A = np.random.rand(500, 100)
        b = np.random.rand(500, 1)
        x, losses, time_list = run_sgd(A, b)
        plt.figure()
        plt.plot(losses)
        plt.title("Stochastic Gradient Descent Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()
        plt.figure()
        plt.plot(time_list)
        plt.title("Time taken for stochastic gradient descent step")
        plt.xlabel("step")
        plt.ylabel("time")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TP1 of computational imaging")
    parser.add_argument(
        "-t",
        "--task",
        type=int,
        default=3,
        help="Enter the number of the task to execute",
    )

    args = parser.parse_args()

    tasks(args.task)
