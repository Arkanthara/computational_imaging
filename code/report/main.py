import skimage as sk
import numpy as np
import matplotlib.pyplot as plt

def add_noise(image, noise_level):
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)

def save_noised_images(image_path: str, noise_levels: list):
    image = sk.io.imread(image_path)
    image = sk.img_as_float(image)
    for noise_level in noise_levels:
        noisy_image = sk.util.random_noise(image, mode='gaussian', var=noise_level**2, clip=True)
        sk.io.imsave(f"noisy_image_{noise_level}.png", sk.img_as_ubyte(noisy_image))
    sk.io.imsave("noise.png", sk.img_as_ubyte(np.clip(np.random.normal(0, 0.5, image.shape), 0, 1)))

if __name__ == "__main__":
    image_path = "img/tangled.png"  # Path to your input image
    noise_levels = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 10.0]  # Different noise levels to apply
    save_noised_images(image_path, noise_levels)