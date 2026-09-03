import skimage as sk
import numpy as np
import matplotlib.pyplot as plt
import os

def add_noise(image, noise_level):
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)

def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def save_noised_images(image_path: str, noise_levels: list):
    image = sk.io.imread(image_path)
    image = sk.img_as_float(image)
    image = normalize_image(image)
    for noise_level in noise_levels:
        noisy_image = sk.util.random_noise(image, mode='gaussian', var=noise_level**2, clip=True)
        sk.io.imsave(f"noisy_image_{noise_level}.png", sk.img_as_ubyte(noisy_image))
        sk.io.imsave(f"noise_{noise_level}.png", sk.img_as_ubyte(np.clip(np.random.normal(0, noise_level, image.shape), 0, 1)))
    sk.io.imsave("noise.png", sk.img_as_ubyte(np.clip(np.random.normal(0, 0.5, image.shape), 0, 1)))

def save_bit_planes(image_path: str, output_dir: str = "bit_planes"):
    # Charger en niveaux de gris
    image = sk.io.imread(image_path, as_gray=True)

    # Conversion uint8 [0,255]
    image = sk.img_as_ubyte(image)

    print(
        f"dtype={image.dtype}, "
        f"range=[{image.min()}, {image.max()}]"
    )

    os.makedirs(output_dir, exist_ok=True)

    sk.io.imsave(
        os.path.join(output_dir, "original.png"),
        image)

    for i in range(8):
        bit_plane = ((image >> i) & 1).astype(np.uint8)
        bit_plane_img = bit_plane * 255

        print(f"bit {i}: {np.unique(bit_plane)}")

        sk.io.imsave(
            os.path.join(output_dir, f"bit_plane_{i}.png"),
            bit_plane_img,
            check_contrast=False
        )

if __name__ == "__main__":
    image_path = "img/tangled_2.png"  # Path to your input image
    noise_levels = [0.1, 0.3, 0.5, 1.0, 2.0, 10.0]  # Different noise levels to apply
    image = sk.io.imread(image_path)
    save_noised_images(image_path, noise_levels)
    save_bit_planes("img/tangled_2.png")