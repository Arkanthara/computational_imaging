import numpy as np
import matplotlib.pyplot as plt
from pypher.pypher import psf2otf
from scipy.signal import convolve2d
import argparse
import skimage as sk


def read_image(name: str = "img/tangled.jpg") -> np.ndarray:
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


def MSE(img_1: np.ndarray, img_2: np.ndarray) -> float:
    return np.mean((img_1 - img_2) ** 2)


def normalize(img: np.ndarray, target: float = 1.0) -> np.ndarray:
    return (img - img.min()) * target / (img.max() - img.min())


def add_noise(img: np.ndarray, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    noise = np.random.normal(loc=mean, scale=std, size=img.shape)
    img_noised = img + noise
    return np.clip(img_noised, a_min=0, a_max=255)


def filterFT(img: np.ndarray, h: np.ndarray) -> np.ndarray:
    F_img = np.fft.fft2(img)
    h = psf2otf(h, shape=(img.shape))
    h = normalize(h)
    return normalize(np.abs(np.fft.ifft2(F_img * h)), target=1.0)


def filter(img: np.ndarray, h: np.ndarray) -> np.ndarray:
    # Here, we ask for symmetric padding to apply filter to avoid vignette artifact
    return normalize(convolve2d(img, h, mode="same"), target=1.0)


def gaussianKernel(std: float) -> np.ndarray:
    h = np.arange(102).astype(np.float64)
    h = np.exp(-1 / 2 * (h - np.mean(h)) ** 2 / std**2)
    h = normalize(h, target=1.0)
    h = np.outer(h, h)
    return h


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TP1 of computational imaging")
    parser.add_argument(
        "-t",
        "--task",
        type=int,
        default=1,
        help="Enter the number of the task to execute",
    )

    args = parser.parse_args()

    tangled = sk.io.imread("img/tangled_small.jpg", as_gray=True)
    tangled = normalize(tangled, target=1.0)
    print_range(tangled)

    # TASK 1
    # 1.1 Low-pass filtering in frequency domain (using np.convolve2d !)
    if args.task == 1:
        index = 1
        plt.figure()
        plt.suptitle("Gaussian blur")
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
            tangled_gaussian_blur = filter(tangled, h)
            plt.imshow(tangled_gaussian_blur, cmap="gray")
            plt.title(f"$\\sigma = {i}$, frequency domain")
            plt.axis("off")
            index += 1

            # Apply filter in fourier domain -> multiplication
            plt.subplot(3, 3, index)
            tangled_gaussian_blur_ft = filterFT(tangled, h)
            plt.imshow(tangled_gaussian_blur_ft, cmap="gray")
            plt.title(f"$\\sigma = {i}$, Fourier domain")
            plt.axis("off")
            index += 1
        plt.show()

        # 1.2 High-pass filtering (HighPass = Img - LowPass(Img))
        index = 1
        plt.figure()
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
            tangled_gaussian_blur = filter(tangled, h)
            print_range(tangled_gaussian_blur)
            plt.imshow(tangled - tangled_gaussian_blur, cmap="gray")
            plt.title(f"$\\sigma = {i}$, frequency domain")
            plt.axis("off")
            index += 1

            # Apply filter in fourier domain -> multiplication
            plt.subplot(3, 3, index)
            tangled_gaussian_blur_ft = filterFT(tangled, h)
            plt.imshow(tangled - tangled_gaussian_blur_ft, cmap="gray")
            plt.title(f"$\\sigma = {i}$, Fourier domain")
            plt.axis("off")
            index += 1
        plt.show()

    # TASK 2
    # Blur image
    elif args.task == 2:
        tangled_blured = cv2.GaussianBlur(tangled, (101, 101), 5)

        # 2.1
        index = 1
        plt.figure()
        for i in [0, 0.001, 0.01, 0.1]:
            plt.subplot(2, 2, index)

            # Add noise
            noise = cv2.randn(np.zeros_like(tangled), mean=0, stddev=i)
            tangled_noised = cv2.add(tangled_blured, noise)

            # Inverse filter in Fourrier domain
            F_img = np.fft.fft2(tangled_noised)
            h = cv2.getGaussianKernel(101, 5)
            h = h @ h.T
            h = psf2otf(h, shape=(tangled.shape))
            h = np.fft.fftshift(h)
            tangled_inv_filter = np.abs(np.fft.ifft2(F_img / (h + 1e-16)))
            plt.imshow(tangled_inv_filter, cmap="gray")
            plt.title(f"Inverse filtering with noise $\\sigma = {i}$")
            plt.axis("off")
            index += 1

            # Inverse filter in Fourrier domain
            F_img = np.fft.fft2(tangled_noised)
            h = cv2.getGaussianKernel(101, 5)
            h = h @ h.T
            h = psf2otf(h, shape=(tangled.shape))
            h = np.fft.fftshift(h)
            tangled_inv_filter = np.abs(np.fft.ifft2(F_img / (h + 1e-16)))
            plt.imshow(tangled_inv_filter, cmap="gray")
            plt.title(f"Inverse filtering with noise $\\sigma = {i}$")
            plt.axis("off")
            index += 1
        plt.show()
