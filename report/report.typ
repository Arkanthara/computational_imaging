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

= Methods

In CT, a X-ray source rotates around the object and collects projections at different angles, which are then used to reconstruct the image.
All the projections are collected in a sinogram, which is a 2D representation of the projections with each row corresponding to a specific angle and each column corresponding to a specific detector.
The reconstruction of the image from the sinogram is generally done by using the Radon transform and its inverse, which allows to reconstruct the image from the projections.

The CT acquisition represent a huge amount of data, which can be affected by different types of artifacts, such as motion artifacts and defective detectors.
The computational imaging then is able to correct these artifacts either directly in the obtained reconstructed image or in the sinogram before reconstruction.

Some artifacts can appear due to a small movement of the object during the acquisition, which can lead to a misalignment of the projections.
Then, the alignment of the projections must be done before the reconstruction to avoid uncorrected mapping of the projections, leading generally to some blurring or some kind of ghost artifacts in the reconstructed image, due to the misalignment of the projections.

We will focus in this report on the stripe artifacts, which are most generally caused by defective detectors during the acquisition and produce ring artifacts in the reconstructed image.
The defective detectors will produce columns in the sinogram with different intensity values compared to the other columns, which can lead to the presence of stripe artifacts in the sinogram and thus to a degradation of the quality of the reconstructed image.
These artifacts generally appear as vertical stripes in the sinogram as shown on @fig1-a, which can significantly degrade the quality of the reconstructed image, like in @fig1-b.
Indeed, if a detector is defective, it will stay defective during the whole acquisition, so for all angle of projection.
This produces some vertical stripes in the sinogram, which produce some ring artifacts in the reconstructed image due to the combination of all the projections from all angles.

The quality of the diagnosis therefore can be significantly degraded by the presence of these artifacts, which can lead to a misdiagnosis or a missed diagnosis.
So the removal of these artifacts is a crucial step in the reconstruction process to improve the quality of the reconstructed image and thus the quality of the diagnosis to allow a better care of the patients.

```python
%| echo: False
%| label: fig1
%| grid-align: bottom
%| grid-inset: 6pt

import os
from pathlib import Path
import matplotlib.pyplot as plt
import skimage
import numpy as np

import algotom.prep.calculation as calc
import algotom.prep.removal as remo
import algotom.rec.reconstruction as rec
import algotom.util.utility as util

report_dir = Path.cwd()
repo_root = report_dir.parent
code_dir = os.path.join(repo_root, "code")

def get_reconstruction(sinogram: np.ndarray, title: str = "Reconstructed image") -> np.ndarray:
    """
    Reconstruct an image from a sinogram.
    Takes as input a sinogram, then compute the center of rotation and the angles of the projections, and finally apply the filtered back projection (FBP) algorithm to reconstruct the image.

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

# --- Load the sinogram with stripe artifacts ---
sinogram = skimage.io.imread(
    os.path.join(code_dir, "sarepy", "data", "challenging", "all_stripe_types_sample1.tif")
).astype(np.float32)

# --- Reconstruction without stripe removal ---
rec_img = get_reconstruction(sinogram)

# --- visualization ---
plt.figure()
plt.suptitle("Original sinogram with stripe artifacts and reconstruction without removal")
plt.subplot(1, 2, 1)
plt.imshow(sinogram, cmap="gray")
plt.title("Sinogram with stripe artifacts")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(rec_img, cmap="gray")
plt.title("Reconstruction without stripe removal")
plt.axis("off")
plt.show()
```

To remove the stripe artifacts, we will explore different methods, such as Fourier-based methods, wavelet-based methods, normalization-based methods, regularization-based methods, and sorting-based methods.

== Fourier-based methods

The Fourier-based methods are based on the idea that the stripe artifacts are specific structures that introduce specific patterns in the Fourier domain by introducing energy at a specific level of frequencies.
Indeed, if we look at the @fig2-f, we can see that the stripe artifacts are represented by a line of high energy at a given position in the Fourier domain.
So removing this line of high energy can help to remove the stripe artifacts from the sinogram and improve the quality of the reconstructed image, as shown in @fig3-b.

The challenge of the Fourier-based methods is to accurately identify the line of high energy corresponding to the stripe artifacts and to remove it without affecting the other components of the sinogram, which can lead to a loss of information and a degradation of the quality of the reconstructed image.

So this method can be very effective to remove a single stripe artifact, but it can be less effective when there are multiple stripe artifacts with different intensities and positions... However an iterative process could be done to remove multiple stripe artifacts by applying the Fourier-based method multiple times focus different stripes, but it can be computationally expensive and can lead to a loss of information if not done carefully.

```python
%| echo: False
%| label: fig2
%| grid-inset: 6pt

# --- Load the sinogram with stripe artifacts ---
sinogram = skimage.io.imread(
    os.path.join(code_dir, "sarepy", "data", "challenging", "all_stripe_types_sample1.tif")
).astype(np.float32)

# --- Fourier transform of the sinogram ---
fft_sinogram = np.fft.fft2(sinogram)
fft_sinogram_shifted = np.fft.fftshift(fft_sinogram)

# -- Remove stripe artifacts using Fourier-based method ---
removed_stripe = remo.remove_stripe_based_fft(sinogram)

# --- Fourier transform of the filtered sinogram ---
fft_removed_stripe = np.fft.fft2(removed_stripe)
fft_removed_stripe_shifted = np.fft.fftshift(fft_removed_stripe)

# -- Visualization of the process in spatial and Fourier domains ---
plt.figure(figsize=(12, 12))

plt.suptitle("Fourier-based stripe removal: visualizing the process in spatial and Fourier domains")

plt.subplot(2, 3, 1)
plt.imshow(sinogram, cmap="gray")
plt.title("Original sinogram with stripe artifacts")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(removed_stripe, cmap="gray")
plt.title("Sinogram after Fourier-based stripe removal")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(sinogram - removed_stripe, cmap="gray")
plt.title("Removed component")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(np.log1p(np.abs(fft_sinogram_shifted)), cmap="gray")
plt.title("Magnitude of FFT")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(np.log1p(np.abs(fft_removed_stripe_shifted)), cmap="gray")
plt.title("Magnitude of FFT (destriped)")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(np.log1p(np.abs(fft_sinogram_shifted - fft_removed_stripe_shifted)), cmap="gray")
plt.title("Removed component in Fourier domain")
plt.axis("off")

plt.tight_layout()
plt.show()
```

```python
%| echo: False
%| label: fig3

# --- Reconstruction without stripe removal ---
rec_img = get_reconstruction(sinogram)

# --- Reconstruction with stripe removal ---
rec_img_destriped = get_reconstruction(removed_stripe)

# --- visualization ---
plt.figure()

plt.suptitle("Effect of Fourier-based stripe removal on the reconstructed image")

plt.subplot(1, 3, 1)
plt.imshow(rec_img, cmap="gray")
plt.title("Reconstructed image (with stripes)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(rec_img_destriped, cmap="gray")
plt.title("Reconstructed image (destriped)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(rec_img - rec_img_destriped, cmap="gray")
plt.title("Removed component")
plt.axis("off")

plt.tight_layout()
plt.show()
```

== Wavelet-based methods

The wavelet-based methods are based on the idea that the stripe artifacts can be removed by applying a wavelet decomposition to the sinogram.
Then, some of the wavelet coefficients capturing the stripe artifacts can be identified, like in @fig4-c.
Applying the Fourier transform to these coefficients allows to identify the line of high energy corresponding to the stripe artifacts, like in @fig4-g.
Removing this line of high energy and applying the inverse Fourier transform allows to remove the stripe artifacts from the wavelet coefficients.
Then, applying the inverse wavelet transform allows to reconstruct the sinogram without stripe artifacts, as shown in @fig4-b.

This method is based on the multi-scale analysis obtained through wavelet decomposition.
Indeed, some stripe artifacts are more visible at a specific scale of the wavelet decomposition, which allows to easily identify the coefficients corresponding to the stripe artifacts and thus to remove them.

Compared to the Fourier-based methods, the wavelet-based methods can be more effective to remove multiple stripe artifacts with different intensities and positions, as they allow to analyze the sinogram at different scales and thus to identify the coefficients corresponding to the different stripe artifacts.
And we can see this result by comparing the @fig2-b and @fig5-b, where the wavelet-based method seems to be very more effective to remove the stripe artifacts compared to the Fourier-based method: the results shows that there is still some ring artifacts in the reconstructed image after applying the Fourier-based method, while the wavelet-based method seems to have removed almost all the ring artifacts in the reconstructed image.
However, by applying parameters correctly and using an iterative process, the Fourier-based method can be as effective as the wavelet-based method to remove multiple stripe artifacts, but it can be computationally expensive and can lead to a loss of information if not done carefully.

The parameters of the wavelet decomposition, such as the type of wavelet, the level of decomposition and the strength of the removal, can affect the performance of the method and need to be carefully chosen to achieve good results without introducing artifacts or losing information in the reconstructed image.

The disadvantage of the wavelet-based methods is that they can be computationally expensive, especially when the level of decomposition is high, as it requires to apply the wavelet decomposition and the inverse wavelet transform multiple times.
And for some usecases, applying another method such as normalization-based methods can be more effective and less computationally expensive to remove the stripe artifacts, especially when the stripe artifacts are not too strong and do not require a multi-scale analysis to be removed effectively.

```python
%| echo: False
%| label: fig4
%| raw: False
%| grid-inset: 6pt

# Visualize wavelet coefficients of sinogram with stripe artifacts

# --- Load the sinogram with stripe artifacts ---
sinogram = skimage.io.imread(
	os.path.join(code_dir, "sarepy", "data", "challenging", "all_stripe_types_sample1.tif")
).astype(np.float32)

# --- wavelet decomposition ---
outputs = util.apply_wavelet_decomposition(sinogram, "db10", level=7)

# outputs structure: [cA_L, (cH_L,cV_L,cD_L), ..., (cH_1,cV_1,cD_1)]
# pick one detail tuple to visualize (outputs[-1] is (cH_1, cV_1, cD_1))
cH, cV, cD = outputs[-1]

# --- FFT of vertical coefficients ---
fft_cV = np.fft.fftshift(np.fft.fft2(cV))

# -- Remove stripe artifacts ---
removed_stripe = remo.remove_stripe_based_wavelet_fft(sinogram, level=7, size=6, wavelet_name="db10")

# --- wavelet decomposition of the filtered sinogram ---
outputs_filtered = util.apply_wavelet_decomposition(removed_stripe, "db10", level=7)
cH_filtered, cV_filtered, cD_filtered = outputs_filtered[-1]

# --- FFT of vertical coefficients of the filtered sinogram ---
fft_cV_filtered = np.fft.fftshift(np.fft.fft2(cV_filtered))

# --- visualization ---
plt.figure(figsize=(12, 12))

plt.suptitle("Wavelet-based stripe removal: visualizing the process in spatial, wavelet, and Fourier domains")

plt.subplot(3, 3, 1)
plt.imshow(sinogram, cmap="gray")
plt.title("Original sinogram with stripe artifacts")
plt.axis("off")

plt.subplot(3, 3, 2)
plt.imshow(removed_stripe, cmap="gray")
plt.title("Sinogram after wavelet-based stripe removal")
plt.axis("off")

plt.subplot(3, 3, 3)
plt.imshow(sinogram - removed_stripe, cmap="gray")
plt.title("Removed component")
plt.axis("off")

plt.subplot(3, 3, 4)
plt.imshow(cV, cmap="gray")
plt.title("Wavelet cV (original)")
plt.axis("off")

plt.subplot(3, 3, 5)
plt.imshow(cV_filtered, cmap="gray")
plt.title("Wavelet cV (destriped)")
plt.axis("off")

plt.subplot(3, 3, 6)
plt.imshow(cV - cV_filtered, cmap="gray")
plt.title("Wavelet cV removed")
plt.axis("off")

plt.subplot(3, 3, 7)
plt.imshow(np.log1p(np.abs(fft_cV)), cmap="gray")
plt.title("log|FFT(cV original)|")
plt.axis("off")

plt.subplot(3, 3, 8)
plt.imshow(np.log1p(np.abs(fft_cV_filtered)), cmap="gray")
plt.title("log|FFT(cV destriped)|")
plt.axis("off")

plt.subplot(3, 3, 9)
plt.imshow(np.log1p(np.abs(fft_cV - fft_cV_filtered)), cmap="gray")
plt.title("log|FFT(cV removed)|")
plt.axis("off")

plt.tight_layout()
plt.show()
```

```python
%| echo: False
%| label: fig5

# --- Reconstruction without stripe removal ---
rec_img = get_reconstruction(sinogram)

# --- Reconstruction with stripe removal ---
rec_img_destriped = get_reconstruction(removed_stripe)

# --- visualization ---
plt.figure()

plt.suptitle("Effect of wavelet-based stripe removal on the reconstructed image")

plt.subplot(1, 3, 1)
plt.imshow(rec_img, cmap="gray")
plt.title("Reconstructed image (with stripes)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(rec_img_destriped, cmap="gray")
plt.title("Reconstructed image (destriped)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(rec_img - rec_img_destriped, cmap="gray")
plt.title("Removed component")
plt.axis("off")

plt.tight_layout()
plt.show()
```

== Normalization-based methods

The normalization-based methods are based on the assumption that each column of the sinogram corresponds to a specific detector.
If the detector is defective, the corresponding column will have a different mean or median value compared to the other columns.
The normalization then consists to retrieve the good calibration of the detector by finding the baseline tendency of columns of the sinogram by analyzing their statistical properties, such as the mean or median.
To do that, a Gaussian filter can be applied to the sinogram to smooth the columns and retrieve the baseline tendency by deleting the high frequency components corresponding to the noise that distort the estimation of the baseline tendency.
The strength of the Gaussian filter impacts the performance of the method, as a too strong filter can lead to a loss of information and a degradation of the quality of the reconstructed image, while a too weak filter can lead to a poor estimation of the baseline tendency and thus a poor performance of the method.

So the normalization-based methods can help to reduce the intensity of the stripe artifacts by correcting the calibration of the detector and thus reducing the intensity of the stripe artifacts in the sinogram, which can improve the quality of the reconstructed image, as shown in @fig6-e.

Due to the fact that the normalization-based methods are based on the analysis of the statistical properties of the columns of the sinogram, they can be less effective when there are multiple stripe artifacts with different intensities and positions, as it can be difficult to accurately estimate the baseline tendency of the columns in the presence of multiple stripe artifacts.
On top of that, the output quality of the method depends on the choice of the parameters, such as the strength of the Gaussian filter, which can be difficult to choose correctly without introducing artifacts or losing information in the reconstructed image.
And the normalization can also be applied to non-defective columns, which can lead to a degradation of the quality of the reconstructed image if not done carefully.

So compare to the other methods, this method is more simple and more focus on the correction of the calibration of the detector. And in our case, it is outperformed by the wavelet-based method which seems to be more efficient, as shown on @fig5-b and @fig6-e, but it can be more effective than the Fourier-based method, as shown on @fig3-b, especially when there are multiple stripe artifacts with different intensities and positions.

```python
%| echo: False
%| label: fig6
%| grid-inset: 6pt

# Visualize normalization-based stripe removal

# --- Loading the sinogram ---
sinogram = skimage.io.imread(
    os.path.join(code_dir, "sarepy", "data", "challenging", "all_stripe_types_sample1.tif")
).astype(np.float32)

# --- Reconstruction without stripe removal ---
rec_img = get_reconstruction(sinogram)

# --- normalization-based stripe removal ---
removed_stripe = remo.remove_stripe_based_normalization(sinogram, sigma=25)

# --- Reconstruction with stripe removal ---
rec_img_destriped = get_reconstruction(removed_stripe)

# --- visualization ---
plt.figure()
plt.suptitle("Normalization-based stripe removal")
plt.subplot(2, 3, 1)
plt.imshow(sinogram, cmap="gray")
plt.title("Original sinogram with stripe artifacts")
plt.axis("off")
plt.subplot(2, 3, 2)
plt.imshow(removed_stripe, cmap="gray")
plt.title("Sinogram after normalization-based stripe removal")
plt.axis("off")
plt.subplot(2, 3, 3)
plt.imshow(sinogram - removed_stripe, cmap="gray")
plt.title("Removed component")
plt.axis("off")
plt.subplot(2, 3, 4)
plt.imshow(rec_img, cmap="gray")
plt.title("Reconstructed image (with stripes)")
plt.axis("off")
plt.subplot(2, 3, 5)
plt.imshow(rec_img_destriped, cmap="gray")
plt.title("Reconstructed image (destriped)")
plt.axis("off")
plt.subplot(2, 3, 6)
plt.imshow(rec_img - rec_img_destriped, cmap="gray")
plt.title("Removed component in reconstructed image")
plt.axis("off")
plt.tight_layout()
plt.show()
```



== Regularization-based methods

The regularization-based methods are based on the idea that the stripe artifacts can be removed by adding a regularization term to the reconstruction problem.

Indeed, we can consider that the acquisition of the sinogram is a linear process that can be represented by a linear operator A, such that the sinogram y can be expressed as y = A x + n, where x is the original image and n is the noise.
The reconstruction of the image from the sinogram can then be formulated as an optimization problem, where we want to find the image x that minimizes the difference between the sinogram y and the projection of the image A x, while also adding a regularization term R(x) that imposes certain constraints on the solution.
Indeed, the regularization term indicates which kind of solution we want to obtain, such as a smooth solution or a sparse solution.

So the main challenge of the regularization-based methods is to choose the right regularization term that can effectively remove the stripe artifacts while preserving the important features of the image.
The method works quite well as shown in @fig7.

The big disadvantage of the regularization-based methods is that they can be computationally expensive, as they require to solve an optimization problem, which can be time-consuming, especially for large images or for complex regularization terms.
And the output is not anymore the "true" image, but the solution of an optimization problem, which can be different from the true image and can be affected by the choice of the regularization term and the parameters of the optimization algorithm.

But if we look at the result in @fig7-e, it is very good and seems very similar to the result obtained with the wavelet-based method, as shown in @fig5-b, which seems to be the best method to remove the stripe artifacts in our case.

```python
%| echo: False
%| label: fig7
%| grid-inset: 6pt

# Visualize regularization-based stripe removal
# --- Loading the sinogram ---
sinogram = skimage.io.imread(
    os.path.join(code_dir, "sarepy", "data", "challenging", "all_stripe_types_sample1.tif")
).astype(np.float32)

# --- Reconstruction without stripe removal ---
rec_img = get_reconstruction(sinogram)

# --- regularization-based stripe removal ---
removed_stripe = remo.remove_stripe_based_regularization(sinogram)

# --- Reconstruction with stripe removal ---
rec_img_destriped = get_reconstruction(removed_stripe)

# --- visualization ---
plt.figure()
plt.suptitle("Regularization-based stripe removal")
plt.subplot(2, 3, 1)
plt.imshow(sinogram, cmap="gray")
plt.title("Original sinogram with stripe artifacts")
plt.axis("off")
plt.subplot(2, 3, 2)
plt.imshow(removed_stripe, cmap="gray")
plt.title("Sinogram after regularization-based stripe removal")
plt.axis("off")
plt.subplot(2, 3, 3)
plt.imshow(sinogram - removed_stripe, cmap="gray")
plt.title("Removed component")
plt.axis("off")
plt.subplot(2, 3, 4)
plt.imshow(rec_img, cmap="gray")
plt.title("Reconstructed image (with stripes)")
plt.axis("off")
plt.subplot(2, 3, 5)
plt.imshow(rec_img_destriped, cmap="gray")
plt.title("Reconstructed image (destriped)")
plt.axis("off")
plt.subplot(2, 3, 6)
plt.imshow(rec_img - rec_img_destriped, cmap="gray")
plt.title("Removed component in reconstructed image")
plt.axis("off")
plt.tight_layout()
plt.show()
```



== Sorting-based methods

The sorting-based methods are based on the idea that the stripe artifacts can be removed by sorting the columns of the sinogram based on their intensity values.
The main idea is that the columns corresponding to the defective detectors will have much different intensity values compared to the other columns.
So the sorting allows to identify these columns by sorting the intensity values inside each column and looking at the distribution of the sorted values across the columns.
Then, the columns corresponding to the defective detectors can be identified by looking at the distribution of the sorted values across the columns.
The distribution of defective columns can be equalized by applying some correction such as enforcing the same distribution of intensity values accross the columns.
To do that, a filter like a median filter can be applied to the sorted sinogram (shown in @fig8-d) to smooth the distribution of intensity values across the columns and thus reduce the intensity of the stripe artifacts in the sinogram, which can improve the quality of the reconstructed image, as shown in @fig8-h.
Then, the inverse sorting can be applied to retrieve the destriped sinogram in the original order of columns.

The advantage of the sorting-based method is its simplicity and its ability to effectively remove stripe artifacts by equalizing the distribution of intensity values across the columns of the sinogram.

The disadvantage of the sorting-based method is that it can be less effective when there are multiple stripe artifacts with different intensities and positions, as it can be difficult to accurately identify the columns corresponding to the defective detectors and to apply the appropriate correction to equalize the distribution of intensity values across the columns.

And if the stripe artifacts are not too strong, the sorting-based method can be less effective than the other methods by not detecting the artifacts and thus not applying the appropriate correction to remove them.

Due to the equalization, generally made with a filter like a median filter, the method can lead to a loss of information and a degradation of the quality of the reconstructed image if not done carefully.
Indeed, the equalization is applied also to non-defective columns.



```python
%| echo: False
%| label: fig8
%| grid-inset: 6pt

# Visualize sorting-based stripe removal
# --- Loading the sinogram ---
sinogram = skimage.io.imread(
    os.path.join(code_dir, "sarepy", "data", "challenging", "all_stripe_types_sample1.tif")
).astype(np.float32)

# --- Reconstruction without stripe removal ---
rec_img = get_reconstruction(sinogram)

# --- sorting-based stripe removal ---
removed_stripe = remo.remove_stripe_based_sorting(sinogram, size=50)

# --- Reconstruction with stripe removal ---
rec_img_destriped = get_reconstruction(removed_stripe)

# --- visualization ---
plt.figure(figsize=(12, 12))
plt.suptitle("Sorting-based stripe removal")
plt.subplot(3, 3, 1)
plt.imshow(sinogram, cmap="gray")
plt.title("Original sinogram with stripe artifacts")
plt.axis("off")
plt.subplot(3, 3, 2)
plt.imshow(removed_stripe, cmap="gray")
plt.title("Sinogram after sorting-based stripe removal")
plt.axis("off")
plt.subplot(3, 3, 3)
plt.imshow(sinogram - removed_stripe, cmap="gray")
plt.title("Removed component")
plt.axis("off")
plt.subplot(3, 3, 4)
plt.imshow(np.sort(sinogram, axis=0), cmap="gray")
plt.title("Sorted sinogram")
plt.axis("off")
plt.subplot(3, 3, 5)
plt.imshow(np.sort(removed_stripe, axis=0), cmap="gray")
plt.title("Sorted destriped sinogram")
plt.axis("off")
plt.subplot(3, 3, 6)
plt.imshow(np.sort(sinogram, axis=0) - np.sort(removed_stripe, axis=0), cmap="gray")
plt.title("Removed component in sorted sinogram")
plt.axis("off")
plt.subplot(3, 3, 7)
plt.imshow(rec_img, cmap="gray")
plt.title("Reconstructed image (with stripes)")
plt.axis("off")
plt.subplot(3, 3, 8)
plt.imshow(rec_img_destriped, cmap="gray")
plt.title("Reconstructed image (destriped)")
plt.axis("off")
plt.subplot(3, 3, 9)
plt.imshow(rec_img - rec_img_destriped, cmap="gray")
plt.title("Removed component in reconstructed image")
plt.axis("off")
plt.tight_layout()
plt.show()
```
#pagebreak()
= Conclusion

In this report, we have explored different methods to remove stripe artifacts from sinograms and reconstruct the image.

The Fourier-based methods are based on the idea that the stripe artifacts are specific structures that introduce specific patterns in the Fourier domain by introducing energy at a specific level of frequencies.
The Fourier-based methods can be effective to remove a single stripe artifact, but they can be less effective when there are multiple stripe artifacts with different intensities and positions, as it can be difficult to accurately identify the line of high energy corresponding to each stripe artifact and to remove it without affecting the other components of the sinogram. However an iterative process could be done.

The wavelet-based methods are based on the idea that the stripe artifacts can be removed by applying a wavelet decomposition to the sinogram that gives a multi-scale representation.
Then, the stripe artifacts can be identified at a specific scale of the wavelet decomposition and removed by applying a Fourier-based method to the corresponding wavelet coefficients.
The wavelet-based methods can be more effective to remove multiple stripe artifacts with different intensities and positions, as they allow to analyze the sinogram at different scales and thus to identify the coefficients corresponding to the different stripe artifacts.
However, they can be computationally expensive.

The normalization-based methods are based on the assumption that each column of the sinogram corresponds to a specific detector and that the defective detectors will have different intensity values compared to the other columns, so the normalization consists to retrieve the good calibration of the detector by finding the baseline tendency of columns of the sinogram by analyzing their statistical properties, such as the mean or median.
The normalization-based methods is more effective when the stripe artifacts are caused by defective detectors to recalibrate.
But in case of multiple stripe artifacts with different intensities and positions, it can be difficult to accurately estimate the baseline tendency of the columns in the presence of multiple stripe artifacts, which can lead to a poor performance of the method.

The regularization-based methods are based on the idea that the stripe artifacts can be removed by adding a regularization term to the reconstruction problem.
Due to the optimization process, the regularization-based methods can be computationally expensive and return an image which is not the original, but they can be very effective to remove the stripe artifacts while preserving the important features of the image if the right regularization term is chosen.

The sorting-based methods are based on the idea that the stripe artifacts can be removed by sorting the columns of the sinogram based on their intensity values and applying some correction to equalize the distribution of intensity values across the columns.
The process is very simple and not computationally expensive, but it can be less effective when there are multiple stripe artifacts with different intensities and positions or when the artifacts have poor intensity profiles.

So the choice of the method to use depends on the type of stripe artifacts and requirements of the application, such as the quantity of information to preserve, the computational resources available, the desired image quality or something else.
Each method has its own advantages and disadvantages, and the performance of each method can be affected by the choice of the parameters involved in the methods, which need to be carefully chosen to achieve good results without introducing artifacts or losing information in the reconstructed image.

This report was done based on the algotom library, which provides different methods to remove stripe artifacts from sinograms and reconstruct the image @Vo:21.

The dataset used in this report was obtained from the github repository of the sarepy library which provides different sinograms with different types of stripe artifacts to test the performance of the different methods @sarepy_github.
#report-footnote("Usage of AI: ChatGPT for a thorough understanding and to summarize papers explaining the different methods for stripe removal, VSCode autocompletion for Python and report writing.")
