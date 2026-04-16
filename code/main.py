"""
Reconstruction from Sarepy sinogram + visualization.

This script:
- Loads a sinogram (.tif)
- Displays it
- Removes stripe artifacts
- Finds center of rotation
- Reconstructs image
- Displays result
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

import algotom.prep.calculation as calc
import algotom.prep.removal as remo
import algotom.rec.reconstruction as rec


# =============================================================================
# VISUALIZE ORIGINAL SINOGRAM
# =============================================================================

def visualize_sinogram(sinogram: np.ndarray, title: str = "Original sinogram") -> None:
    """
    Visualize the original sinogram.

    Parameters:
    -----------
    sinogram: np.ndarray
        Input sinogram to visualize.
    title: str
        Title for the plot.

    Returns:
    --------
    None
        Displays the original sinogram.

    """
    plt.figure()
    plt.title("Original sinogram")
    plt.imshow(sinogram, cmap="gray")
    plt.xlabel("Detector pixels")
    plt.ylabel("Projection angle")
    plt.colorbar()
    plt.show()

def get_reconstruction(sinogram: np.ndarray, title: str = "Reconstructed image") -> np.ndarray:
    """
    Reconstruct an image from a sinogram.

    Parameters:
    -----------
    sinogram: np.ndarray
        Input sinogram for reconstruction.
    title: str
        Title for the plot.

    Returns:
    --------
    np.ndarray
        The reconstructed image.

    """
    n_angles = sinogram.shape[0]
    angles = np.linspace(0, np.pi, n_angles)

    center = calc.find_center_vo(sinogram)
    rec_img = rec.fbp_reconstruction(
        sinogram,
        center,
        angles=angles,
        apply_log=True,
        gpu=False
    )
    return rec_img

# =============================================================================
# REMOVE STRIPES (IMPORTANT FOR SAREPY DATA)
# =============================================================================

def remove_stripe_plot(sinogram: np.ndarray, remove_stripe: callable, args: dict = {}, shape: tuple = (1, 2), title: str = "Wavelet FFT based method") -> np.ndarray:
    """
    Remove stripe artifacts from a sinogram using different methods and visualize results.

    Parameters:
    -----------
    sinogram: np.ndarray
        Input sinogram with stripe artifacts.
    remove_stripe: callable
        Function to remove stripes from the sinogram.
    args: dict
        Arguments for the remove_stripe function.
    shape: tuple
        Shape of the subplot grid (rows, cols).
    title: str
        Title for the plot.

    Returns:
    --------
    None
         Displays the cleaned sinograms and their reconstructions.

    """


    idx = 1
    plt.figure()
    plt.suptitle(title)
    # Apply Fourier-Wavelet method
    sino = remove_stripe(sinogram, **args)
    plt.subplot(shape[0], shape[1], idx)
    plt.imshow(sino, cmap="gray")
    plt.title(f"Cleaned sinogram")
    plt.axis("off")
    idx += 1

    rec_img = get_reconstruction(sino)
    # Visualize result
    plt.subplot(shape[0], shape[1], idx)
    plt.imshow(rec_img, cmap="gray")
    plt.title(f"Reconstruction")
    plt.axis("off")
    idx += 1

    plt.tight_layout()
    plt.show()



# sino_fw = remo.remove_stripe_fourier_wavelet(sinogram, sigmas=[3, 3])

# # Fourier method
# sino_fourier = remo.remove_stripe_fourier(sinogram, sigma=3)

# # Normalization based method
# sino_norm = remo.remove_stripe_based_normalization(sinogram, 15)

# # Regularization based method
# sino_reg = remo.remove_stripe_based_regularization(sinogram, 0.1)

# # Sorting based method
# sino_sort = remo.remove_stripe_based_sorting(sinogram, 15)

# Option : enlever pixels aberrants
# sino_clean = remo.remove_zinger(sino_clean, 0.08)


# =============================================================================
# VISUALIZE CLEANED SINOGRAM
# =============================================================================

# plt.figure()
# plt.title("Cleaned sinogram")
# plt.imshow(sino_clean, cmap="gray", aspect="auto")
# plt.colorbar()
# plt.show()


# =============================================================================
# FIND CENTER OF ROTATION
# =============================================================================

# center = calc.find_center_vo(sino_clean)
# print(f"Estimated center: {center}")


# =============================================================================
# RECONSTRUCTION
# =============================================================================

# rec_img = rec.fbp_reconstruction(
#     sino_clean,
#     center,
#     angles=angles,
#     apply_log=True,
#     gpu=False
# )


# =============================================================================
# VISUALIZE RESULT
# =============================================================================

# plt.figure()
# plt.title("Reconstructed slice")
# plt.imshow(rec_img, cmap="gray")
# plt.colorbar()
# plt.axis("off")
# plt.show()

if __name__ == "__main__":

    # =============================================================================
    # LOAD SINOGRAM
    # =============================================================================

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "sarepy", "data", "sinogram_normal.tif")

    sinogram = io.imread(file_path).astype(np.float32)

    print(f"Sinogram shape: {sinogram.shape}")

    remove_stripe_plot(sinogram, remove_stripe=remo.remove_stripe_based_wavelet_fft, title="Wavelet FFT based method")
    remove_stripe_plot(sinogram, remove_stripe=remo.remove_stripe_based_fft, title="Fourier based method")
    remove_stripe_plot(sinogram, remove_stripe=remo.remove_stripe_based_normalization, title="Normalization based method")
    remove_stripe_plot(sinogram, remove_stripe=remo.remove_stripe_based_regularization, title="Regularization based method")
    remove_stripe_plot(sinogram, remove_stripe=remo.remove_stripe_based_sorting, title="Sorting based method")