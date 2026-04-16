// Main report file
#import "template.typ": make-report, report-footnote
#import "metadata.typ": my-report
#import "@preview/theofig:0.1.0": definition
#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node
#import "@preview/cetz:0.4.2": canvas, draw
// #import "graph_utils.typ": *
#import "neural-viz/lib.typ": *
#import emoji: camera

// Main content
#show: make-report.with(my-report)

= Introduction

The computational imaging allows to reconstruct an image from a set of measurements from different kind of devices, such as X-ray devices.
The computed tomography (CT) is a specific type of computational imaging that allows to reconstruct a 3D image from a set of 2D projections, called sinograms.
The reconstruction of the image from the sinogram is a challenging problem, especially when the sinogram is corrupted by noise or artifacts, such as stripe artifacts.
In this report, we will explore different methods to remove stripe artifacts from sinograms and reconstruct the image.

= Methodology

In CT, a X-ray source rotates around the object and collects projections at different angles, which are then used to reconstruct the image.
All the projections are collected in a sinogram, which is a 2D image where the horizontal axis represents the angle of the projection and the vertical axis represents the position of the detector.

The stripe artifacts are caused by the presence of defects in the detector such as dead pixels or hot pixels, which can be due to various reasons, such as dust, scratches, or electronic noise.
These artifacts appear as vertical stripes in the sinogram as shown on @fig1, which can significantly degrade the quality of the reconstructed image.
Indeed, the presence of stripe artifacts can seriously affect the quality of the diagnosis due to the presence of ring artifacts in the reconstructed image, which can be misinterpreted as pathological features or hide real pathologies.

```python
%| echo: False
%| label: fig1
%| grid-align: bottom
%| grid-inset: 6pt

import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import skimage
import numpy as np
import algotom.prep.removal as remo

report_dir = Path.cwd()
repo_root = report_dir.parent
code_dir = os.path.join(repo_root, "code")

sys.path.append(code_dir)

from main import remove_stripe_plot, get_reconstruction

# Load sinogram
sinogram = skimage.io.imread(os.path.join(code_dir, "sarepy", "data", "challenging",  "all_stripe_types_sample1.tif"))

# Visualize sinogram
plt.figure()
plt.suptitle("Original sinogram with stripe artifacts and reconstruction without removal")
plt.subplot(1, 2, 1)
plt.imshow(sinogram, cmap="gray")
plt.title("Sinogram with stripe artifacts")
plt.axis("off")
plt.subplot(1, 2, 2)
rec_img = get_reconstruction(sinogram)
plt.imshow(rec_img, cmap="gray")
plt.title("Reconstruction without stripe removal")
plt.axis("off")
plt.show()
```

To remove the stripe artifacts, we will explore different methods, such as Fourier-based methods, wavelet-based methods, normalization-based methods, regularization-based methods, and sorting-based methods.

== Fourier-based methods

The Fourier-based methods are based on the idea that the stripe artifacts can be represented as high-frequency components in the Fourier domain.
By applying a low-pass filter in the Fourier domain, we can remove the high-frequency components and thus remove the stripe artifacts.

This is done for all Fourier-based methods such as the FFT (Fast Fourier Transform) Wavelet method and the FFT method.

```python
%| echo: False
%| label: fig2
%| img-width: 140%

# Visualize FFT of sinogram with stripe artifacts

fft_sinogram = np.fft.fft2(sinogram)
fft_sinogram_shifted = np.fft.fftshift(fft_sinogram)

plt.figure()
plt.suptitle("Representation of stripe artifacts in magnitude Fourier domain")
plt.subplot(1, 3, 1)
plt.imshow(sinogram, cmap="gray")
plt.title("Original sinogram with stripe artifacts")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(np.log(1 + np.abs(fft_sinogram_shifted)), cmap="gray")
plt.title("Magnitude of Fourier transform")
plt.axis("off")

removed_stripe = remo.remove_stripe_based_fft(sinogram)
fft_removed_stripe = np.fft.fft2(removed_stripe)
fft_removed_stripe_shifted = np.fft.fftshift(fft_removed_stripe)
fft_removed_stripe_shifted = fft_sinogram_shifted - fft_removed_stripe_shifted


plt.subplot(1, 3, 3)
plt.imshow(np.log(1 + np.abs(fft_removed_stripe_shifted)), cmap="gray")
plt.title("Stripe artifacts")
plt.axis("off")
plt.show()
```

== Normalization-based methods

The normalization-based methods are based on the idea that the stripe artifacts can be removed by normalizing the sinogram.
This is done by dividing each column of the sinogram by its mean or median value, which can help to reduce the intensity of the stripe artifacts.

== Regularization-based methods

The regularization-based methods are based on the idea that the stripe artifacts can be removed by adding a regularization term to the reconstruction problem.
This is done by adding a term that penalizes the presence of stripe artifacts in the reconstructed image, such as the total variation (TV) regularization.

== Sorting-based methods

The sorting-based methods are based on the idea that the stripe artifacts can be removed by sorting the columns of the sinogram based on their mean or median value.
This can help to reduce the intensity of the stripe artifacts by rearranging the columns of the sinogram.
