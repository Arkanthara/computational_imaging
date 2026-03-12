import numpy as np
import matplotlib.pyplot as plt
from pypher.pypher import psf2otf
from scipy.signal import convolve2d
import argparse
import scipy
import skimage as sk
import time


def read_image(name: str = "img/img.jpg") -> np.ndarray:
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
    return A.T @ (A @ x - b)

def residual_l2(A, x, b):
    return 1/2 * np.linalg.norm(A @ x - b) ** 2

def grad_l2_fourier(H, F, B):
    return 2 * H * (H * F - B)

def residual_l2_fourier(H, F, B):
    return np.linalg.norm(H * F - B) ** 2

def run_gd_fourier(
    H,
    B,
    step_size: float = 1e-1,
    num_iters: int = 3000,
    grad_fn=grad_l2_fourier,
    residual=residual_l2_fourier,
):
    F = B.copy()
    losses = []
    time_list = []
    for i in range(num_iters):
        init_time = time.time()
        grad = grad_fn(H, F, B)
        F = F - step_size * grad
        losses.append(residual(H, F, B))
        time_list.append(time.time() - init_time)
    return F, losses, time_list

def run_sgd_fourier(
    H,
    B,
    step_size: float = 1e-1,
    num_iters: int = 3000,
    batch_size: int = 32,
    grad_fn=grad_l2_fourier,
    residual=residual_l2_fourier,
):
    F = B.copy()
    losses = []
    time_list = []
    for i in range(num_iters):
        init_time = time.time()
        index = np.random.choice(B.size, batch_size, replace=False)
        idx, idy = np.unravel_index(index, B.shape)
        H_batch = H[idx, idy]
        F_batch = F[idx, idy]
        B_batch = B[idx, idy]
        grad = grad_fn(H_batch, F_batch, B_batch)
        F[idx, idy] = F[idx, idy] - step_size * grad
        losses.append(residual(H, F, B))
        time_list.append(time.time() - init_time)
    return F, losses, time_list


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
    init_time = time.time()
    for i in range(num_iters):
        idx = np.random.choice(A.shape[0], batch_size, replace=False)
        A_batch = A[idx]
        b_batch = b[idx]
        grad = grad_fn(A_batch, x, b_batch)
        x = x - step_size * grad
        losses.append(residual(A, x, b))
        time_list.append(time.time() - init_time)
    return x, losses, time_list


def tasks(task: int = 1, subtask: int = 1, img_path: str = "img/tangled_small.jpg", figsize: tuple[int, int] = (10, 10), original: bool = False, test: bool = False):
    img = sk.io.imread(
        img_path,
        as_gray=True,
    )
    img = sk.util.img_as_float(img)
    # TASK 1
    # 1.1 Low-pass filtering in Spatial domain (using np.convolve2d !)
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

            # Apply filter in Spatial domain -> 2D convolution
            plt.subplot(3, 3, index)
            img_gaussian_blur = filter(img, h)
            plt.imshow(img_gaussian_blur, cmap="gray")
            plt.title(f"Spatial domain#linebreak()PSNR = {sk.metrics.peak_signal_noise_ratio(img, img_gaussian_blur):.2f} dB")
            plt.axis("off")
            index += 1

            # Apply filter in fourier domain -> multiplication
            plt.subplot(3, 3, index)
            img_gaussian_blur_ft = filterFT(img, h)
            plt.imshow(img_gaussian_blur_ft, cmap="gray")
            plt.title(f"Fourier domain#linebreak()PSNR = {sk.metrics.peak_signal_noise_ratio(img, img_gaussian_blur_ft):.2f} dB")
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

            # Apply filter in Spatial domain -> 2D convolution
            plt.subplot(3, 3, index)
            img_gaussian_blur = filter(img, h)
            plt.imshow(img - img_gaussian_blur, cmap="gray")
            plt.title(f"Spatial domain#linebreak()PSNR = {sk.metrics.peak_signal_noise_ratio(img, img - img_gaussian_blur):.2f} dB")
            plt.axis("off")
            index += 1

            # Apply filter in fourier domain -> multiplication
            plt.subplot(3, 3, index)
            img_gaussian_blur_ft = filterFT(img, h)
            plt.imshow(img - img_gaussian_blur_ft, cmap="gray")
            plt.title(f"Fourier domain#linebreak()PSNR = {sk.metrics.peak_signal_noise_ratio(img, img - img_gaussian_blur_ft):.2f} dB")
            plt.axis("off")
            index += 1
        return plt.gcf()

    # TASK 2: Inverse filtering and Wiener filtering
    elif task == 2:
        img_blured = filterFT(img, gaussianKernel(5))
        if original:
            plt.figure(figsize=figsize)
            plt.subplot(1, 2, 1)
            plt.imshow(img, cmap="gray")
            plt.axis("off")
            plt.title("Original image")
            plt.subplot(1, 2, 2)
            plt.imshow(img_blured, cmap="gray")
            plt.axis("off")
            plt.title("Blured image")
            return plt.gcf()
        if subtask == 1:
            index = 1
            plt.figure(figsize=figsize)
            for i in [0, 0.001, 0.01, 0.1]:
                plt.subplot(2, 2, index)
                # Add noise
                img_noised = add_noise(img_blured, std=i)
                # Inverse filter in Fourrier domain
                h = gaussianKernel(5)
                img_inv_filter = filterFT(img_noised, h, inv_filter=True)
                plt.imshow(img_inv_filter, cmap="gray")
                plt.title(f"Inverse filtering#linebreak()noise $\\sigma = {i}$#linebreak()PSNR = {sk.metrics.peak_signal_noise_ratio(img, img_inv_filter):.2f} dB")
                plt.axis("off")
                index += 1
            return plt.gcf()
        if subtask == 2:
            index = 1
            plt.figure(figsize=figsize)
            for i in [0, 0.001, 0.01, 0.1]:
                plt.subplot(2, 2, index)
                # Add noise
                img_noised = add_noise(img_blured, std=i)
                # Wiener filter in Fourrier domain
                h = gaussianKernel(5)
                img_inv_filter = filterFT(
                    img_noised, h, wiener=True, K=i / np.mean(img_noised)
                )
                plt.imshow(img_inv_filter, cmap="gray")
                plt.title(f"Wiener filtering#linebreak()noise $\\sigma = {i}$#linebreak()PSNR = {sk.metrics.peak_signal_noise_ratio(img, img_inv_filter):.2f} dB")
                plt.axis("off")
                index += 1
            return plt.gcf()

    # TASK 3: Gradient descent
    elif task == 3:
        # Classical gradient descent
        downsampled_img = img[::2, ::2]
        h = gaussianKernel(3, size=31)
        b = filterFT(downsampled_img, h)
        # b = add_noise(b, std=0.1)
        B = np.fft.fft2(b)
        H = psf2otf(h, shape=b.shape)
        step_size = 1e-2
        num_iters = 500
        batch_size = 1000
        if original:
            plt.figure(figsize=figsize)
            plt.subplot(1, 2, 1)
            plt.imshow(downsampled_img, cmap="gray")
            plt.axis("off")
            plt.title("Original image")
            plt.subplot(1, 2, 2)
            plt.imshow(b, cmap="gray")
            plt.axis("off")
            plt.title("Damaged image")
            return plt.gcf()
        if subtask == 1:
            F, losses, time_list = run_gd_fourier(H, B, step_size=step_size, num_iters=num_iters)
            f = np.fft.ifft2(F).real
            plt.figure(figsize=figsize)
            plt.subplot(2, 2, 1)
            plt.imshow(b, cmap="gray")
            plt.title("Damaged image")
            plt.axis("off")
            plt.subplot(2, 2, 2)
            plt.imshow(f, cmap="gray")
            plt.title("Reconstructed image #linebreak()PSNR = {:.2f} dB".format(sk.metrics.peak_signal_noise_ratio(downsampled_img, f)))
            plt.axis("off")
            plt.subplot(2, 2, 3)
            plt.plot(losses)
            plt.title("Gradient Descent Loss")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.subplot(2, 2, 4)
            plt.plot(time_list)
            plt.title("Time taken for gradient descent step")
            plt.xlabel("step")
            plt.ylabel("time")
            return plt.gcf()
        if subtask == 2:
            F, losses, time_list = run_sgd_fourier(H, B, step_size=step_size, num_iters=num_iters * 20, batch_size=batch_size)
            f = np.fft.ifft2(F).real
            plt.figure(figsize=figsize)
            plt.subplot(2, 2, 1)
            plt.imshow(b, cmap="gray")
            plt.title("Damaged image")
            plt.axis("off")
            plt.subplot(2, 2, 2)
            plt.imshow(f, cmap="gray")
            plt.title("Reconstructed image#linebreak()PSNR = {:.2f} dB".format(sk.metrics.peak_signal_noise_ratio(downsampled_img, f)))
            plt.axis("off")
            plt.subplot(2, 2, 3)
            plt.plot(losses)
            plt.title("Stochastic Gradient Descent Loss")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.subplot(2, 2, 4)
            plt.plot(time_list)
            plt.title("Time taken for a step")
            plt.xlabel("step")
            plt.ylabel("time")
            return plt.gcf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TP1 of computational imaging")
    parser.add_argument(
        "-t",
        "--task",
        type=int,
        default=3,
        help="Enter the number of the task to execute",
    )
    parser.add_argument("-i", "--image", type=str, default="img/tangled_small.jpg", help="Path to the input image")

    args = parser.parse_args()

    figsize = (8, 8)

    img = sk.io.imread(
        args.image,
        as_gray=True,
    )
    img = sk.util.img_as_float(img)
    print_range(img)
    # TASK 1
    # 1.1 Low-pass filtering in Spatial domain (using np.convolve2d !)
    if args.task == 1:
        tasks(1, 1, figsize).show()
        tasks(1, 2, figsize).show()
    elif args.task == 2:
        tasks(2, original=True, figsize=figsize).show()
        tasks(2, 1, figsize).show()
        tasks(2, 2, figsize).show()
    elif args.task == 3:
        tasks(3, original=True, figsize=figsize).show()
        tasks(3, 1, figsize=figsize).show()
        # tasks(3, test=True, figsize=figsize).show()
        # tasks(3, 2, figsize).show()
        input("Press Enter to continue...")
