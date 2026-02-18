import cv2
import numpy as np
import matplotlib.pyplot as plt
from pypher.pypher import psf2otf


def read_image(name: str = "img/tangled.jpg") -> np.ndarray:
    img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
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


def normalize(img: np.ndarray, target: int = 255) -> np.ndarray:
    return (img - img.min()) * target / (img.max() - img.min())


def add_noise(img: np.ndarray, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    noise = np.random.normal(loc=mean, scale=std, size=img.shape)
    img_noised = img + noise
    return np.clip(img_noised, a_min=0, a_max=255)


def filterFT(img: np.ndarray, h: np.ndarray) -> np.ndarray:
    F_img = np.fft.fft2(img)
    h = np.fft.fftshift(h)
    # print_image(np.log(np.abs(F_img * h) + 1))
    return np.abs(np.fft.ifft2(F_img * h))


def gaussianKernel(std: float) -> np.ndarray:
    h = cv2.getGaussianKernel(101, std)
    h = h @ h.T
    return h


if __name__ == "__main__":
    tangled = cv2.imread("img/tangled.jpg", cv2.IMREAD_GRAYSCALE)
    print_range(tangled)

    # TASK 1
    # 1.1 Low-pass filtering in frequency domain (using np.convolve2d !)
    index = 1
    plt.figure()
    for i in [0.1, 1, 10]:
        plt.subplot(3, 1, index)
        tangled_gaussian_blur = cv2.GaussianBlur(tangled, (101, 101), i)
        plt.imshow(tangled_gaussian_blur, cmap="gray")
        plt.title(f"Gaussian blur with $\\sigma = {i}$")
        plt.axis("off")
        index += 1
    plt.show()

    # 1.2 High-pass filtering (HighPass = Img - LowPass(Img))
    index = 1
    plt.figure()
    for i in [0.1, 1, 10]:
        plt.subplot(3, 1, index)
        tangled_gaussian_highpass = tangled - cv2.GaussianBlur(tangled, (101, 101), i)
        plt.imshow(tangled_gaussian_highpass, cmap="gray")
        plt.title(f"Gaussian sharpening with $\\sigma = {i}$")
        plt.axis("off")
        index += 1
    plt.show()

    # TASK 2
    # Blur image
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
    plt.show()
