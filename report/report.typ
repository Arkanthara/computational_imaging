// Main report file
#import "template.typ": make-report, report-footnote
#import "metadata.typ": my-report
#import "@preview/theofig:0.1.0": definition
#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node
#import "@preview/cetz:0.4.2": canvas, draw
#import "graph_utils.typ": *

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

The code for this part was given in matlab.
I converted it to python using github copilot and reviewed it to understand how it works.
I have also tried to optimize it by using FFT-based convolution instead of spatial convolution.

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

#align(center)[
#scale(88%)[
#canvas(length: 0.72cm, {
  import draw: *

  // Nodes: auto-positioned from left to right using gap.
  let image-ds = make-dataset(
    "Image\nDataset",
    images: 3,
    image-size: (1.9, 2.3),
    image-spacing: 0.22,
    pos: (1.0, 0.0),
    color: rgb("#a8c8e8"),
  )

  let depth-ds = make-dataset(
    "Depth Map\nDataset",
    images: 3,
    image-size: (1.9, 2.3),
    image-spacing: 0.22,
    pos: (1.0, -3.8),
    color: rgb("#c5a0f0"),
  )

  let encoder = make-trapezoid(
    "ssiDepthChns",
    subtitle: "Encoder",
    mode: "encoder",
    width: 2.8,
    big-half: 1.65,
    small-half: 0.80,
    after: image-ds,
    gap: node-gap,
    color: rgb("#b0dba0"),
  )

  let latent = make-dataset(
    "H × W × #Features",
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
    "ssiDepthDetect",
    subtitle: "Depth Map\nEstimator",
    after: latent,
    gap: node-gap,
    size: (3.6, 2.2),
    color: rgb("#a0b8f5"),
  )

  let computed = make-dataset(
    "Computed\nDepth Maps",
    images: 3,
    image-size: (1.9, 2.3),
    image-spacing: 0.22,
    after: detector,
    gap: node-gap,
    color: rgb("#a0e8be"),
    title-size: 0.54em,
  )

  let compare = make-box(
    "Comparison Block",
    subtitle: "Loss / Evaluation",
    after: computed,
    gap: node-gap,
    y: -1.85,
    color: rgb("#f5a0be"),
    title-size: 0.56em,
  )

  draw-node(image-ds)
  draw-node(depth-ds)
  draw-node(encoder)
  draw-node(latent)
  draw-node(detector)
  draw-node(computed)
  draw-node(compare)

  // Arrows and labels.
  let a-input = make-arrow(
    image-ds,
    encoder,
    from-outer: true,
    auto-spacing: true,
    spacing: arrow-gap,
    label-gap: arrow-label-gap,
    label: [input],
  )

  let a-features = make-arrow(
    encoder,
    latent,
    auto-spacing: false,
    spacing: arrow-gap,
    label-gap: arrow-label-gap,
    label: [features],
  )

  let a-decode = make-arrow(
    latent,
    detector,
    from-outer: true,
    auto-spacing: false,
    spacing: arrow-gap,
    label-gap: arrow-label-gap,
    label: [decode],
  )

  let a-estimate = make-arrow(
    detector,
    computed,
    auto-spacing: false,
    spacing: arrow-gap,
    label-gap: arrow-label-gap,
  )

  let a-compare = make-arrow(
    computed,
    compare,
    out-side: "bottom",
    in-side: "left",
    auto-spacing: true,
    spacing: arrow-gap,
    label-gap: arrow-label-gap,
    mode: "vh",
    label: [compare],
  )

  let a-ground = make-arrow(
    depth-ds,
    compare,
    out-side: "right",
    in-side: "bottom",
    from-outer: true,
    auto-spacing: true,
    spacing: arrow-gap,
    label-gap: arrow-label-gap,
    mode: "hv",
    label: [ground truth],
  )

  draw-arrow(a-input)
  draw-arrow(a-features)
  draw-arrow(a-decode)
  draw-arrow(a-estimate)
  draw-arrow(a-compare)
  draw-arrow(a-ground)
})
]
]



= Task 2: Image refocusing

= Task 3: Image refocusing and depth estimation



= Implementation <impl>

= Results

= Discussion

= Conclusion

