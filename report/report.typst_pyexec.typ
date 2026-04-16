
#show figure.where(kind: "subfigure"): set figure(supplement: "Figure")

#show figure.where(kind: image): outer => {
  counter(figure.where(kind: "subfigure")).update(0)
  set figure(numbering: (..nums) => {
    let outer-nums = counter(figure.where(kind: image)).at(outer.location())
    std.numbering("1a", ..outer-nums, ..nums)
  })
  show figure.where(kind: "subfigure"): inner => {
    show figure.caption: it => context {
      std.numbering("(a)", it.counter.at(inner.location()).last())
      [ ]
      it.body
    }
    inner
  }
  outer
}
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

#figure(grid(columns: 2, align: bottom, inset: 6pt, [#figure(image(".typst_pyexec/figures/cell_1_1_1.svg"), kind: "subfigure", caption: [Sinogram with stripe artifacts]) <fig1-a>], [#figure(image(".typst_pyexec/figures/cell_1_1_2.svg"), kind: "subfigure", caption: [Reconstruction without stripe removal]) <fig1-b>]), caption: [Original sinogram with stripe artifacts and reconstruction without removal], kind: image) <fig1>


To remove the stripe artifacts, we will explore different methods, such as Fourier-based methods, wavelet-based methods, normalization-based methods, regularization-based methods, and sorting-based methods.

== Fourier-based methods

The Fourier-based methods are based on the idea that the stripe artifacts can be represented as high-frequency components in the Fourier domain.
By applying a low-pass filter in the Fourier domain, we can remove the high-frequency components and thus remove the stripe artifacts.

This is done for all Fourier-based methods such as the FFT (Fast Fourier Transform) Wavelet method and the FFT method.

#figure(grid(columns: 3, [#figure(image(".typst_pyexec/figures/cell_2_1_1.svg", width: 140%), kind: "subfigure", caption: [Original sinogram with stripe artifacts]) <fig2-a>], [#figure(image(".typst_pyexec/figures/cell_2_1_2.svg", width: 140%), kind: "subfigure", caption: [Magnitude of Fourier transform]) <fig2-b>], [#figure(image(".typst_pyexec/figures/cell_2_1_3.svg", width: 140%), kind: "subfigure", caption: [Stripe artifacts]) <fig2-c>]), caption: [Representation of stripe artifacts in magnitude Fourier domain], kind: image) <fig2>


== Normalization-based methods

The normalization-based methods are based on the idea that the stripe artifacts can be removed by normalizing the sinogram.
This is done by dividing each column of the sinogram by its mean or median value, which can help to reduce the intensity of the stripe artifacts.

== Regularization-based methods

The regularization-based methods are based on the idea that the stripe artifacts can be removed by adding a regularization term to the reconstruction problem.
This is done by adding a term that penalizes the presence of stripe artifacts in the reconstructed image, such as the total variation (TV) regularization.

== Sorting-based methods

The sorting-based methods are based on the idea that the stripe artifacts can be removed by sorting the columns of the sinogram based on their mean or median value.
This can help to reduce the intensity of the stripe artifacts by rearranging the columns of the sinogram.
