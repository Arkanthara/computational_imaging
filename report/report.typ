// Main report file
#import "template.typ": make-report, report-footnote
#import "metadata.typ": my-report
#import "@preview/theofig:0.1.0": definition
#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node
#import "@preview/cetz:0.4.2": canvas, draw
// #import "graph_utils.typ": *
#import "neural-viz/lib.typ": *

// Main content
#show: make-report.with(my-report)

= Introduction

The classical digital photography allows to capture images with a good quality.
However, due to the physical properties of the camera, the focus is made on a specific point, and the notion of depth of field appears: objects not focused are blurred.

To respond to this problem, computational imaging allows to refocus the image after it has been captured thanks to analysis of the depth of field.
Indeed, the depth of field gives a notion of the distance between the camera and the objects of the scene, and can be estimated by an analysis of the blur of the image, as we will study in the first part of this work.
Then, thanks to the depth of field information, a refocus can be performed thanks to application of specific filters according to the depth of field, as we will see in the second part of this work.

To have a more precise estimation of the depth of field, some cameras have been designed like the plenoptic camera.
These cameras capture the depth of field by a multi view acquisition system created by a microlens array placed in front of the sensor.
The depth of field can then be estimated by an analysis of the multi view acquisition and the refocus can be performed in this way, as we will see in the third part of this work.

= Task 1: Depth estimation

The code was implemented in matlab.
As some random forest was used in matlab, the code wasn't directly reusable in python.

That's why I try to understand the code with the main functions and principles.
Then, I used the understanding to implement from scratch a similar pipeline in python using ChatGPT and python libraries like scikit-learn.

The base principle of the depth map estimation is described in the @fig1 for the training pipeline, and in @fig2 for the prediction pipeline.

// Two inputs: image dataset and depth map dataset (shown as stacked images with offset)
// Image input → encoder "ssiDepthChns" (trapezoid shape, big side input, small side output)
// Feature extraction → Latent space (vertical rectangle representing H×W×#features matrix)
// Latent space → "Depth Map Estimator" block (ssiDepthDetect function)
// Depth Map Estimator → set of computed depth maps
// Comparison block between input depth maps and computed depth maps
// Two outputs: loss/comparison result

// ─────────────────────────────────────────────────────────────────
//  ssiDepth pipeline diagram — Left-to-Right, CeTZ
// ─────────────────────────────────────────────────────────────────

#let node-gap = 1.25
#let arrow-gap = 0.45
#let arrow-label-gap = 0.30
#let make-arrow = set-arrow-defaults(options: (spacing: arrow-gap, label-gap: arrow-label-gap))
#let make-trapezoid = set-trapezoid-defaults(options: (width: 2.8, big-half: 1.65, small-half: 0.80))

#figure(
graph-canvas({

  // Nodes: auto-positioned from left to right using gap.
  let image-ds = make-image-dataset(
    "Image\nDataset",
    src: "../../../code/depth_master/utils/image.jpg",
    images: 3,
    image-spacing: 0.22,
    pos: (1.0, 0.0),
    color: rgb("#a8c8e8"),
  )

  let depth-ds = make-image-dataset(
    "Depth Map\nDataset",
    src: "../../../code/depth_master/utils/depth.jpg",
    images: 3,
    image-size: (1.9, 2.3),
    image-spacing: 0.22,
    pos: (1.0, -3.8),
    color: rgb("#c5a0f0"),
  )

  let encoder = make-trapezoid(
    "Features\nextraction",
    mode: "encoder",
    after: image-ds,
    gap: node-gap,
    color: rgb("#b0dba0"),
  )

  let latent = make-image-dataset(
    "Features",
    src: "../../../code/depth_master/utils/law_filter.jpg",
    images: 6,
    image-size: (1.15, 1.6),
    image-spacing: 0.11,
    title-position: "below",
    title-truncate: true,
    after: encoder,
    gap: node-gap,
    color: rgb("#fcd97a"),
    title-size: 0.50em,
  )

  let detector = make-box(
    "Depth Map\nEstimation",
    subtitle: "Random Forest",
    after: latent,
    gap: node-gap,
    size: (3.6, 2.2),
    color: rgb("#a0b8f5"),
  )

  let computed = make-image-dataset(
    "Computed\nDepth Maps",
    src: "../../../code/depth_master/utils/predicted_depth.jpg",
    images: 3,
    image-size: (1.9, 2.3),
    image-spacing: 0.22,
    after: detector,
    gap: node-gap,
    color: rgb("#a0e8be"),
    title-size: 0.54em,
  )

  let compare = make-box(
    "Compare",

    after: computed,
    gap: node-gap,
    y: -1.9,
    color: rgb("#f5a0be"),
    title-size: 0.56em,
  )

  let nodes = (image-ds, depth-ds, encoder, latent, detector, computed, compare)
  let arrows = (
    make-arrow(image-ds, encoder, from-outer: true, auto-spacing: true),
    make-arrow(encoder, latent, auto-spacing: false),
    make-arrow(latent, detector, from-outer: true, auto-spacing: false),
    make-arrow(detector, computed, auto-spacing: false),
    make-arrow(computed, compare, in-side: "top", auto-spacing: true, label: [prediction]),
    make-arrow(
      depth-ds,
      compare,
      out-side: "right",
      in-side: "bottom",
      order: 2,
      from-outer: true,
      auto-spacing: true,
      mode: "hv",
      label: [ground truth],
    ),
    make-arrow(
      compare,
      detector,
      out-side: "bottom",
      order: 1,
      in-side: "bottom",
      label: [train parameters],
    ),
  )

  draw-graph(nodes: nodes, arrows: arrows)
  draw-node-emoji(detector, kind: "lock-open", place: "top")
}),
caption: "Depth map estimation training based on the given matlab code") <fig1>

#figure(
graph-canvas({

  // Nodes: auto-positioned from left to right using gap.
  let image-ds = make-image-dataset(
    "Image\nDataset",
    src: "../../../code/depth_master/utils/image.jpg",
    images: 1,
    image-spacing: 0.22,
    pos: (1.0, 0.0),
    color: rgb("#a8c8e8"),
  )

  let encoder = make-trapezoid(
    "Features\nextraction",
    mode: "encoder",
    after: image-ds,
    gap: node-gap,
    color: rgb("#b0dba0"),
  )

  let latent = make-image-dataset(
    "Features",
    src: "../../../code/depth_master/utils/law_filter.jpg",
    images: 6,
    image-size: (1.15, 1.6),
    image-spacing: 0.11,
    title-position: "below",
    title-truncate: true,
    after: encoder,
    gap: node-gap,
    color: rgb("#fcd97a"),
    title-size: 0.50em,
  )

  let detector = make-box(
    "Depth Map\nEstimation",
    subtitle: "Random Forest",
    after: latent,
    gap: node-gap,
    size: (3.6, 2.2),
    color: rgb("#a0b8f5"),
  )

  let computed = make-image-dataset(
    "Computed\nDepth Map",
    src: "../../../code/depth_master/utils/predicted_depth.jpg",
    images: 1,
    image-size: (1.9, 2.3),
    image-spacing: 0.22,
    after: detector,
    gap: node-gap,
    color: rgb("#a0e8be"),
    title-size: 0.54em,
  )

  let nodes = (image-ds, encoder, latent, detector, computed)
  let arrows = (
    make-arrow(image-ds, encoder, from-outer: true, auto-spacing: true),
    make-arrow(encoder, latent, auto-spacing: false),
    make-arrow(latent, detector, from-outer: true, auto-spacing: false),
    make-arrow(detector, computed, auto-spacing: false),

  )

  draw-graph(nodes: nodes, arrows: arrows)
  draw-node-emoji(detector, kind: "lock-closed", place: "top")
}),
caption: "Depth map estimation prediction") <fig2>

== Features

There is a lot of features extracted used to compute the depth map, such as color, edge and texture features.

=== Color features

As shown on @fig3, color features are extracted from different color spaces:
  - RGB (Red, Green, Blue)
  - LUV (Lightness, U and V chrominance channels)
  - HSI (Hue, Saturation, Intensity)

They allow to capture different representations of the image that are used in depth estimation.
Indeed, an object with a specific color is more likely to be at a specific depth.

=== Texture features

Texture features are computed by applying laws filters to the image, as shown on @fig4.

Laws filters are a set of filters designed to capture different levels of texture in an image.

Indeed, some of the filters make a smoothing of the image while others capture fine details.



=== Edge features

Edge features are computed by applying Navatia-Babu filters to the image, as shown on @fig5.

Navatia-Babu filters are a set of filters designed to capture edges in an image at different orientations and scales.
The current implementation uses 6 filters to capture edges at orientations of 0°, 30°, 60°, 90°, 120° and 150°.

```python
%| echo: false
%| keep-subplots: false
%| raw: false
%| label: fig3
%| grid-align: top
%| subfigure-caption-position: top

import sys
import os

os.chdir(os.path.normpath(os.path.join(os.getcwd(), "../code/depth_master")))

sys.path.append(os.getcwd())

from depth_estimator import *

dataset_path = os.path.join(os.getcwd(), "make3d")
test_path = os.path.join(os.getcwd(), "Dataset1")
model_path = os.path.join(os.getcwd(), "ssi_rf_model.pkl")

opts = {
    "imResize": (340, 256),
    "shrink": 1,
    "shrinkCol": 4,
    "chnSmooth": 2,
    "simSmooth": 4,
    "nCells": 4
}

model = get_or_train_model(opts, model_path, dataset_path)

sample_img_path = os.path.join(test_path, "img-2.jpg")
sample_img = io.imread(sample_img_path)

fig = plot_depth_filters_structured(sample_img, opts, include=("colors",))["colors"]
fig.suptitle("Color features")
fig.show()
```

```python
%| echo: false
%| label: fig4
%| subfigure-caption-position: top
fig = plot_depth_filters_structured(sample_img, opts, include=("laws",))["laws"]
fig.suptitle("Laws texture features")
fig.show()
```

```python
%| echo: false
%| label: fig5
%| subfigure-caption-position: top

fig = plot_depth_filters_structured(sample_img, opts, include=("babu",))["babu"]
fig.suptitle("Navatia-Babu edge features")
fig.show()
```

== Results

The results obtained with the implemented pipeline are shown on @fig6.

We can see that the predicted depth map captures the general depth of the scene, even if it is not perfect and noisy.
Indeed, the depth of the sky is correctly estimated to be far while the depth of the ground is correctly estimated to be close.

However, as we can see, the depth is quite noisy and not very accurate, which is due to the single capture of the image and the complexity of the depth estimation problem that requires a lot of information to be accurate, like for instance a focal stack or a focus-aperture stack or a multi view acquisition as in plenoptic cameras.

```python
%| echo: false
%| label: fig6
predicted_depth = model.predict(sample_img)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(sample_img)
axes[0].set_title("Input image")
im = axes[1].imshow(predicted_depth, cmap="inferno")
axes[1].set_title("Predicted depth")
axes[1].figure.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
fig.suptitle("Depth estimation results")
axes[0].axis("off")
axes[1].axis("off")
fig.show()
```



= Task 2: Image refocusing

In this task, we will use the depth map of an image or some sub-aperture images to perform some refocusing of the image, based on an existing implementation in matlab.
To convert the matlab code to python, I used another approach than in task 1: this time, I converted the matlab code to python line by line.

Then, I used Github Copilot to update the documentation to numpy style and to debug the code to make it properly working in python.

== Refocusing using depth map

For the refocusing using depth map, the principle is to select the pixels of the image that are at a specific depth an apply a blur filter that increases with the distance to the selected depth on other pixels, as shown on @fig7.

The obtained result is a refocused image at the selected depth, as shown on @fig8.
Thanks to the depth map, the refocusing is accurate and the result is quite good.
Indeed, the information of the depth is crucial to perform a good refocusing, as it allows to apply the right amount of blur on each pixel according to its depth to have a realistic refocus that follows the physical properties of the camera.

So thanks to the depth map, a good refocusing can be performed after the capture of the image, which can be only optained thanks to computational imaging techniques.



#figure(
graph-canvas({

  let img = make-image-node(
    "Input image",
    src: "../../../../code/LightFieldRefocus/Sample_Data/Sample_SelectiveBlurring/painter.png",
    image-width: 3.0,
    image-height: 3.0,
    image-pad: 0,
    image-shift-x: -2,
    pos: (1.0, 0.0),
    color: rgb("#a8c8e8"),
  )

  let select = make-box(
    "Select pixels\nat specific depth",
    after: img,
    gap: 2.0,
    size: (3, 3),
    color: rgb("#a0b8f5"),
  )

  let blur = make-box(
    "Apply blur\nfilter",
    subtitle: "to other pixels",
    legend: "The strength of the blur increases\nwith the distance to the selected depth",
    legend-position: "top",
    after: select,
    gap: 1.79,
    color: rgb("#f5a0be"),
  )


  let depth = make-image-node(
    "Depth map",
    src: "../../../../code/LightFieldRefocus/Sample_Data/Sample_SelectiveBlurring/painter-depth.png",
    image-width: 3.0,
    image-height: 3.0,
    image-shift-x: -2,
    image-pad: 0,
    pos: (1, -4),
    color: rgb("#a0e8be"),
  )

  let depth_in_levels = make-box(
    "Divide depth\ninto levels",
    legend: "Analyze the most frequent depth\nfor background detection",
    after: depth,
    gap: 1.0,
    color: rgb("#f5a0be"),
  )

  let result = make-image-node(
    "Refocused image",
    src: "../../../../code/LightFieldRefocus/outputs/books.jpg",
    image-width: 3.0,
    image-height: 3.0,
    image-shift-x: -2,
    image-pad: 0,
    after: depth_in_levels)

  
  // let result = make-image-node(
  //   "Refocused image",
  //   src: "../../../../code/LightFieldRefocus/output/books.jpg",
  //   after: depth_in_levels
  // )

  let nodes = (img, depth, depth_in_levels, select, blur, result)

  let arrows = (
    make-arrow(depth, depth_in_levels, from-outer: true, auto-spacing: true),
    make-arrow(img, select, mode: "hvh", mode-shift: (0, 1)),
    make-arrow(depth_in_levels, select, out-side: "top", in-side: "left", mode: "vhvh", mode-shifts: ((1, 0.4), (-1, -0.6))),
    make-arrow(select, blur),
    make-arrow(blur, result, out-side: "bottom", in-side: "top")
  )
  draw-graph(nodes: nodes, arrows: arrows)
}), caption: "Refocusing using depth map") <fig7>


```python
%| echo: false
%| label: fig8
%| img-width: 100%

os.chdir(os.path.join(os.getcwd(), "../code/LightFieldRefocus"))
sys.path.append(os.getcwd())

from selectiveBlurring import selectiveBlurring
from skimage import io, util

img = io.imread(os.path.join(os.getcwd(), "Sample_Data", "Sample_SelectiveBlurring", "painter.png"))
depthMap = io.imread(os.path.join(os.getcwd(), "Sample_Data", "Sample_SelectiveBlurring", "painter-depth.png"))

resultBooks = selectiveBlurring(img,depthMap,20,[1, 6],15)

plt.figure(figsize=(20, 10))
plt.suptitle("Refocusing on the books using depth map")
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Input image")
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(depthMap, cmap="gray")
plt.title("Depth map")
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(resultBooks)
plt.title("Refocused image")
plt.axis('off')
plt.show()
```

== Refocusing using sub-aperture images



= Task 3: Image refocusing and depth estimation



= Implementation <impl>

= Results

= Discussion

= Conclusion

