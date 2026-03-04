import numpy as np
import matplotlib.pyplot as plt
import argparse
import skimage as sk

class Mixture:
    def __init__(self, img_1: str, img_2: str):
        self.img_1 = self.readImage(img_1)
        self.img_2 = self.readImage(img_2)

    def readImage(self, img: str) -> np.ndarray:
        img = sk.io.imread(img, as_gray=True)
        img = sk.util.img_as_float(img)
        return img
    
    def getPhase(self, img: np.ndarray) -> np.ndarray:
        F_img = np.fft.fft2(img)
        F_img_P = np.arctan2(F_img.imag, F_img.real)
        return F_img_P
    
    def getMagnitude(self, img: np.ndarray) -> np.ndarray:
        F_img = np.fft.fft2(img)
        F_img_M = np.abs(F_img)
        return F_img_M
    
    def switchPhase(self, alpha: float = 0.5) -> np.ndarray:
        F_img_1_M = self.getMagnitude(self.img_1)
        F_img_1_P = self.getPhase(self.img_1)
        F_img_2_M = self.getMagnitude(self.img_2)
        F_img_2_P = self.getPhase(self.img_2)

        self.alpha = alpha

        self.mixture_1 = F_img_1_M * np.exp(1j * ((1 - alpha) * F_img_1_P + alpha * F_img_2_P))
        self.mixture_2 = F_img_2_M * np.exp(1j * ((1 - alpha) * F_img_1_P + alpha * F_img_2_P))

        self.mixture_1 = np.fft.ifft2(self.mixture_1)
        self.mixture_2 = np.fft.ifft2(self.mixture_2)

    def printMixture(self, figsize: tuple = (8, 8)):
        plt.figure(figsize=figsize)
        plt.subplot(2, 2, 1)
        plt.title("Image 1")
        plt.imshow(self.img_1, cmap="gray")
        plt.axis("off")
        plt.subplot(2, 2, 2)
        plt.title("Image 2")
        plt.imshow(self.img_2, cmap="gray")
        plt.axis("off")
        plt.subplot(2, 2, 3)
        plt.title(f"$|F_1| exp(j({1-self.alpha} \\phi_1 + {self.alpha} \\phi_2))$")
        plt.imshow(np.real(self.mixture_1), cmap="gray")
        plt.axis("off")
        plt.subplot(2, 2, 4)
        plt.title(f"$|F_2| exp(j({1-self.alpha} \\phi_1 + {self.alpha} \\phi_2))$")
        plt.imshow(np.real(self.mixture_2), cmap="gray")
        plt.axis("off")
        return plt.gcf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TP2 of computational imaging")
    parser.add_argument("-i1", "--image1", type=str, default="img/tangled.jpg", help="Path to the first input image")
    parser.add_argument("-i2", "--image2", type=str, default="img/tangled_2.jpg", help="Path to the second input image")
    parser.add_argument("-a", "--alpha", type=float, default=0, help="Alpha value for phase mixing")
    args = parser.parse_args()

    mixture = Mixture(args.image1, args.image2)
    mixture.switchPhase(args.alpha)
    mixture.printMixture().savefig('result.jpg')