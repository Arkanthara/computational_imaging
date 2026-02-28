// Main report file
#import "template.typ": make-report, report-footnote
#import "metadata.typ": my-report
#import "@preview/theofig:0.1.0": definition
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import ".typst_pyimage/pyimage.typ": pyimage, pyinit

// Main content
#show: make-report.with(my-report)

= Introduction

Humans have always tried to capture the present moment, both in painting and in photography.
However, the photos taken are not always of good quality,
that's why some computational methods, such as image filtering, have been developped to increase the quality of a given picture.

In this project, we will examine some basic image filtering methods and some basic way to reconstruct damaged pictures.

= Methodology <methodology>

The image filtering works in both spatial and Fourier domain.

The basics are the following:

- A filter is created to apply some operation on the image.
For instance, a laplacian filter gives the gradient of the image.
- The filter is applied to the image in spatial or Fourier domain.
  - In spatial domain, the convolution between the image and the filter is performed to apply the filter to each pixel of the image.
  - In Fourier domain, a simple multiplication between the image and the filter applies the filter to the image.
    #diagram(
      node-fill: purple.lighten(60%),
      node((0, 1), "Original image"),
      node((0, 3), "filter"),
      node((1, 1), "Fourier Transform"),
      node((1, 3), "Fourier Transform"),
      node((2, 2), "F * H"),

    )



= Implementation <impl>


= Results

#pyinit("
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/../code')
sys.path.append(os.getcwd())
from tp1 import tasks
")

#figure(
pyimage("tasks(1, 1)"), caption: "Low-Pass filter: Gaussian Blur")
#figure(
pyimage("tasks(1, 2)"),
caption: "High-Pass filter: Unsharp Mask")
#figure(grid(rows: 2, pyimage("tasks(2, 1, original=True)", width: 85%), pyimage("tasks(2, 1)")), caption: "Inverse filtering on blured image")
#figure(grid(rows: 2, pyimage("tasks(2, 2, original=True)", width: 85%), pyimage("tasks(2, 2)")), caption: "Wiener filter on blured image")

= Discussion

Blabla
Blabla

= Conclusion
