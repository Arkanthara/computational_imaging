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
# LOAD SINOGRAM
# =============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "sarepy", "data", "sinogram_normal.tif")

sinogram = io.imread(file_path).astype(np.float32)

print(f"Sinogram shape: {sinogram.shape}")


# =============================================================================
# VISUALIZE ORIGINAL SINOGRAM
# =============================================================================

plt.figure()
plt.title("Original sinogram")
plt.imshow(sinogram, cmap="gray", aspect="auto")
plt.xlabel("Detector pixels")
plt.ylabel("Projection angle")
plt.colorbar()
plt.show()


# =============================================================================
# GENERATE ANGLES
# =============================================================================

n_angles = sinogram.shape[0]
angles = np.linspace(0, np.pi, n_angles)


# =============================================================================
# REMOVE STRIPES (IMPORTANT FOR SAREPY DATA)
# =============================================================================

# Méthode rapide (Algotom)
sino_clean = remo.remove_stripe_based_normalization(sinogram, 15)

# Option : enlever pixels aberrants
sino_clean = remo.remove_zinger(sino_clean, 0.08)


# =============================================================================
# VISUALIZE CLEANED SINOGRAM
# =============================================================================

plt.figure()
plt.title("Cleaned sinogram")
plt.imshow(sino_clean, cmap="gray", aspect="auto")
plt.colorbar()
plt.show()


# =============================================================================
# FIND CENTER OF ROTATION
# =============================================================================

center = calc.find_center_vo(sino_clean)
print(f"Estimated center: {center}")


# =============================================================================
# RECONSTRUCTION
# =============================================================================

rec_img = rec.fbp_reconstruction(
    sino_clean,
    center,
    angles=angles,
    apply_log=True,
    gpu=False
)


# =============================================================================
# VISUALIZE RESULT
# =============================================================================

plt.figure()
plt.title("Reconstructed slice")
plt.imshow(rec_img, cmap="gray")
plt.colorbar()
plt.axis("off")
plt.show()