import numpy as np
import matplotlib.pyplot as plt
from pypher.pypher import psf2otf
from scipy.signal import convolve2d
import argparse
import skimage as sk
import time


def add_noise(img: np.ndarray, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    noise = np.random.normal(loc=mean, scale=std, size=img.shape)
    img_noised = img + noise
    return np.clip(img_noised, a_min=0, a_max=255)


def filterFT(
    img: np.ndarray,
    h: np.ndarray,
    inv_filter: bool = False,
    wiener: bool = False,
    K: float = 0.01,
) -> np.ndarray:
    F_img = np.fft.fft2(img)
    H = psf2otf(h, shape=img.shape)
    if inv_filter:
        H_inv = 1 / (H + 1e-10)
        F_img_filtered = F_img * H_inv
    elif wiener:
        H_wiener = 1/(H + 1e-10) * (np.abs(H)**2) / (np.abs(H) ** 2 + K + 1e-10)
        F_img_filtered = F_img * H_wiener
    else:
        F_img_filtered = F_img * H
    img_filtered = np.fft.ifft2(F_img_filtered).real
    return img_filtered


def filter(img: np.ndarray, h: np.ndarray) -> np.ndarray:
    return convolve2d(img, h, mode="same")


def gaussianKernel(std: float, size: int = 101) -> np.ndarray:
    x = np.arange(-(size // 2), size // 2 + 1)
    kernel = np.exp(-(x**2) / (2 * std**2))
    kernel = np.outer(kernel, kernel)
    kernel /= kernel.sum()
    return kernel

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


def tasks(task: int = 1, subtask: int = 1, img_path: str = "img/tangled_small.jpg", figsize: tuple[int, int] = (10, 10), original: bool = False, test: int = 1):
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
            plt.title(f"Spatial domain\nPSNR = {sk.metrics.peak_signal_noise_ratio(img, img_gaussian_blur):.2f} dB")
            plt.axis("off")
            index += 1

            # Apply filter in fourier domain -> multiplication
            plt.subplot(3, 3, index)
            img_gaussian_blur_ft = filterFT(img, h)
            plt.imshow(img_gaussian_blur_ft, cmap="gray")
            plt.title(f"Fourier domain\nPSNR = {sk.metrics.peak_signal_noise_ratio(img, img_gaussian_blur_ft):.2f} dB")
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
            plt.title(f"Spatial domain\nPSNR = {sk.metrics.peak_signal_noise_ratio(img, img - img_gaussian_blur):.2f} dB")
            plt.axis("off")
            index += 1

            # Apply filter in fourier domain -> multiplication
            plt.subplot(3, 3, index)
            img_gaussian_blur_ft = filterFT(img, h)
            plt.imshow(img - img_gaussian_blur_ft, cmap="gray")
            plt.title(f"Fourier domain\nPSNR = {sk.metrics.peak_signal_noise_ratio(img, img - img_gaussian_blur_ft):.2f} dB")
            plt.axis("off")
            index += 1
        return plt.gcf()

    # TASK 2: Inverse filtering and Wiener filtering
    elif task == 2:
        img_blurred = filterFT(img, gaussianKernel(5))
        if original:
            plt.figure(figsize=figsize)
            plt.subplot(1, 2, 1)
            plt.imshow(img, cmap="gray")
            plt.axis("off")
            plt.title("Original image")
            plt.subplot(1, 2, 2)
            plt.imshow(img_blurred, cmap="gray")
            plt.axis("off")
            plt.title(f"blurred image\nPSNR = {sk.metrics.peak_signal_noise_ratio(img, img_blurred)}")
            return plt.gcf()
        if subtask == 1:
            index = 1
            plt.figure(figsize=figsize)
            for i in [0, 0.001, 0.01, 0.1]:
                plt.subplot(2, 2, index)
                # Add noise
                img_noised = add_noise(img_blurred, std=i)
                # Inverse filter in Fourrier domain
                h = gaussianKernel(5)
                img_inv_filter = filterFT(img_noised, h, inv_filter=True)
                plt.imshow(img_inv_filter, cmap="gray")
                plt.title(f"Inverse filtering\nnoise $\\sigma = {i}$\nPSNR = {sk.metrics.peak_signal_noise_ratio(img, img_inv_filter):.2f} dB")
                plt.axis("off")
                index += 1
            return plt.gcf()
        if subtask == 2:
            index = 1
            plt.figure(figsize=figsize)
            for i in [0, 0.001, 0.01, 0.1]:
                plt.subplot(2, 2, index)
                # Add noise
                img_noised = add_noise(img_blurred, std=i)
                # Wiener filter in Fourrier domain
                h = gaussianKernel(5)
                img_inv_filter = filterFT(
                    img_noised, h, wiener=True, K=i / np.mean(img_noised)
                )
                plt.imshow(img_inv_filter, cmap="gray")
                plt.title(f"Wiener filtering\nnoise $\\sigma = {i}$\nPSNR = {sk.metrics.peak_signal_noise_ratio(img, img_inv_filter):.2f} dB")
                plt.axis("off")
                index += 1
            return plt.gcf()

    # TASK 3: Gradient descent
    elif task == 3:
        # Classical gradient descent
        downsampled_img = img[::2, ::2]
        h = gaussianKernel(3, size=31)
        b = filterFT(downsampled_img, h)
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
            plt.suptitle(f"Gradient Descent with step size = {step_size}")
            plt.subplot(2, 2, 1)
            plt.imshow(b, cmap="gray")
            plt.title("Damaged image\nPSNR = {:.2f} dB".format(sk.metrics.peak_signal_noise_ratio(downsampled_img, b)))
            plt.axis("off")
            plt.subplot(2, 2, 2)
            plt.imshow(f, cmap="gray")
            plt.title("Reconstructed image \nPSNR = {:.2f} dB".format(sk.metrics.peak_signal_noise_ratio(downsampled_img, f)))
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
            if test == 1:
                batch_size = 10
                F, losses, time_list = run_sgd_fourier(H, B, step_size=step_size, num_iters=num_iters * 20, batch_size=batch_size)
                f = np.fft.ifft2(F).real
                plt.figure(figsize=figsize)
                plt.suptitle(f"Stochastic Gradient Descent with batch size = {batch_size} and step size = {step_size}")
                plt.subplot(2, 2, 1)
                plt.imshow(b, cmap="gray")
                plt.title("Damaged image\nPSNR = {:.2f} dB".format(sk.metrics.peak_signal_noise_ratio(downsampled_img, b)))
                plt.axis("off")
                plt.subplot(2, 2, 2)
                plt.imshow(f, cmap="gray")
                plt.title("Reconstructed image\nPSNR = {:.2f} dB".format(sk.metrics.peak_signal_noise_ratio(downsampled_img, f)))
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
            if test == 2:
                batch_size = 100
                F, losses, time_list = run_sgd_fourier(H, B, step_size=step_size, num_iters=num_iters * 20, batch_size=batch_size)
                f = np.fft.ifft2(F).real
                plt.figure(figsize=figsize)
                plt.suptitle(f"Stochastic Gradient Descent with batch size = {batch_size} and step size = {step_size}")
                plt.subplot(2, 2, 1)
                plt.imshow(b, cmap="gray")
                plt.title("Damaged image\nPSNR = {:.2f} dB".format(sk.metrics.peak_signal_noise_ratio(downsampled_img, b)))
                plt.axis("off")
                plt.subplot(2, 2, 2)
                plt.imshow(f, cmap="gray")
                plt.title("Reconstructed image\nPSNR = {:.2f} dB".format(sk.metrics.peak_signal_noise_ratio(downsampled_img, f)))
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
            if test == 3:
                batch_size = 1000
                F, losses, time_list = run_sgd_fourier(H, B, step_size=step_size, num_iters=num_iters * 20, batch_size=batch_size)
                f = np.fft.ifft2(F).real
                plt.figure(figsize=figsize)
                plt.suptitle(f"Stochastic Gradient Descent with batch size = {batch_size} and step size = {step_size}")
                plt.subplot(2, 2, 1)
                plt.imshow(b, cmap="gray")
                plt.title("Damaged image\nPSNR = {:.2f} dB".format(sk.metrics.peak_signal_noise_ratio(downsampled_img, b)))
                plt.axis("off")
                plt.subplot(2, 2, 2)
                plt.imshow(f, cmap="gray")
                plt.title("Reconstructed image\nPSNR = {:.2f} dB".format(sk.metrics.peak_signal_noise_ratio(downsampled_img, f)))
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
    if args.task == 1:
        tasks(1, 1, figsize).show()
        tasks(1, 2, figsize).show()
        input("Press Enter to exit...")
    elif args.task == 2:
        tasks(2, original=True, figsize=figsize).show()
        tasks(2, 1, figsize).show()
        tasks(2, 2, figsize).show()
        input("Press Enter to exit...")
    elif args.task == 3:
        tasks(3, original=True, figsize=figsize).show()
        tasks(3, 1, figsize=figsize).show()
        input("Press Enter to exit...")
