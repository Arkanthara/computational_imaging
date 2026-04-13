
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
    src: "../../../../code/depth_master/utils/image.jpg",
    images: 3,
    image-spacing: 0.22,
    pos: (1.0, 0.0),
    color: rgb("#a8c8e8"),
  )

  let depth-ds = make-image-dataset(
    "Depth Map\nDataset",
    src: "../../../../code/depth_master/utils/depth.jpg",
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
    src: "../../../../code/depth_master/utils/law_filter.jpg",
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
    src: "../../../../code/depth_master/utils/predicted_depth.jpg",
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
    src: "../../../../code/depth_master/utils/image.jpg",
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
    src: "../../../../code/depth_master/utils/law_filter.jpg",
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
    src: "../../../../code/depth_master/utils/predicted_depth.jpg",
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

#[
  #show figure.where(kind: "subfigure"): set figure.caption(position: top)
  #figure(grid(columns: 3, align: top, [#figure(image(".typst_pyexec/figures/cell_1_1_1.svg", width: 100%), kind: "subfigure", caption: [R]) <fig3-a>], [#figure(image(".typst_pyexec/figures/cell_1_1_2.svg", width: 100%), kind: "subfigure", caption: [G]) <fig3-b>], [#figure(image(".typst_pyexec/figures/cell_1_1_3.svg", width: 100%), kind: "subfigure", caption: [B]) <fig3-c>], [#figure(image(".typst_pyexec/figures/cell_1_1_4.svg", width: 100%), kind: "subfigure", caption: [Hue]) <fig3-d>], [#figure(image(".typst_pyexec/figures/cell_1_1_5.svg", width: 100%), kind: "subfigure", caption: [Saturation]) <fig3-e>], [#figure(image(".typst_pyexec/figures/cell_1_1_6.svg", width: 100%), kind: "subfigure", caption: [Intensity]) <fig3-f>], [#figure(image(".typst_pyexec/figures/cell_1_1_7.svg", width: 100%), kind: "subfigure", caption: [L]) <fig3-g>], [#figure(image(".typst_pyexec/figures/cell_1_1_8.svg", width: 100%), kind: "subfigure", caption: [U]) <fig3-h>], [#figure(image(".typst_pyexec/figures/cell_1_1_9.svg", width: 100%), kind: "subfigure", caption: [V]) <fig3-i>]), caption: [Color features], kind: image) <fig3>
]


#[
  #show figure.where(kind: "subfigure"): set figure.caption(position: top)
  #figure(grid(columns: 4, [#figure(image(".typst_pyexec/figures/cell_2_1_1.svg"), kind: "subfigure", caption: [Original]) <fig4-a>], [#figure(image(".typst_pyexec/figures/cell_2_1_2.svg"), kind: "subfigure", caption: [Laws 1]) <fig4-b>], [#figure(image(".typst_pyexec/figures/cell_2_1_3.svg"), kind: "subfigure", caption: [Laws 2]) <fig4-c>], [#figure(image(".typst_pyexec/figures/cell_2_1_4.svg"), kind: "subfigure", caption: [Laws 3]) <fig4-d>], [#figure(image(".typst_pyexec/figures/cell_2_1_5.svg"), kind: "subfigure", caption: [Laws 4]) <fig4-e>], [#figure(image(".typst_pyexec/figures/cell_2_1_6.svg"), kind: "subfigure", caption: [Laws 5]) <fig4-f>], [#figure(image(".typst_pyexec/figures/cell_2_1_7.svg"), kind: "subfigure", caption: [Laws 6]) <fig4-g>], [#figure(image(".typst_pyexec/figures/cell_2_1_8.svg"), kind: "subfigure", caption: [Laws 7]) <fig4-h>], [#figure(image(".typst_pyexec/figures/cell_2_1_9.svg"), kind: "subfigure", caption: [Laws 8]) <fig4-i>], [#figure(image(".typst_pyexec/figures/cell_2_1_10.svg"), kind: "subfigure", caption: [Laws 9]) <fig4-j>], [#figure(image(".typst_pyexec/figures/cell_2_1_11.svg"), kind: "subfigure", caption: [Cb L3L3]) <fig4-k>], [#figure(image(".typst_pyexec/figures/cell_2_1_12.svg"), kind: "subfigure", caption: [Cr L3L3]) <fig4-l>]), caption: [Laws texture features], kind: image) <fig4>
]


#[
  #show figure.where(kind: "subfigure"): set figure.caption(position: top)
  #figure(grid(columns: 3, [#figure(image(".typst_pyexec/figures/cell_3_1_1.svg"), kind: "subfigure", caption: [NB 1]) <fig5-a>], [#figure(image(".typst_pyexec/figures/cell_3_1_2.svg"), kind: "subfigure", caption: [NB 2]) <fig5-b>], [#figure(image(".typst_pyexec/figures/cell_3_1_3.svg"), kind: "subfigure", caption: [NB 3]) <fig5-c>], [#figure(image(".typst_pyexec/figures/cell_3_1_4.svg"), kind: "subfigure", caption: [NB 4]) <fig5-d>], [#figure(image(".typst_pyexec/figures/cell_3_1_5.svg"), kind: "subfigure", caption: [NB 5]) <fig5-e>], [#figure(image(".typst_pyexec/figures/cell_3_1_6.svg"), kind: "subfigure", caption: [NB 6]) <fig5-f>]), caption: [Navatia-Babu edge features], kind: image) <fig5>
]


== Results

The results obtained with the implemented pipeline are shown on @fig6.

We can see that the predicted depth map captures the general depth of the scene, even if it is not perfect and noisy.
Indeed, the depth of the sky is correctly estimated to be far while the depth of the ground is correctly estimated to be close.

However, as we can see, the depth is quite noisy and not very accurate, which is due to the single capture of the image and the complexity of the depth estimation problem that requires a lot of information to be accurate, like for instance a focal stack or a focus-aperture stack or a multi view acquisition as in plenoptic cameras.
On top of that, the database used for training has very small and pixelized depth maps, which makes the training not very good and the results not very accurate.
And the computational power of the model is also quite limited, which makes the results not very good.

#figure(grid(columns: 2, [#figure(image(".typst_pyexec/figures/cell_4_1_1.svg"), kind: "subfigure", caption: [Input image]) <fig6-a>], [#figure(image(".typst_pyexec/figures/cell_4_1_2.svg"), kind: "subfigure", caption: [Predicted depth]) <fig6-b>]), caption: [Depth estimation results], kind: image) <fig6>




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
    src: "../../../.typst_pyexec/figures/cell_5_1_1.svg",
    image-width: 3.0,
    image-height: 3.0,
    image-pad: -0.05,
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
    src: "../../../.typst_pyexec/figures/cell_5_1_2.svg",
    image-width: 3.0,
    image-height: 3.0,
    image-shift-x: -2,
    image-pad: -0.05,
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
    src: "../../../.typst_pyexec/figures/cell_5_1_3.svg",
    image-width: 3.0,
    image-height: 3.0,
    image-shift-x: -2,
    image-pad: -0.05,
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


#figure(grid(columns: 3, [#figure(image(".typst_pyexec/figures/cell_5_1_1.svg", width: 100%), kind: "subfigure", caption: [Input image]) <fig8-a>], [#figure(image(".typst_pyexec/figures/cell_5_1_2.svg", width: 100%), kind: "subfigure", caption: [Depth map]) <fig8-b>], [#figure(image(".typst_pyexec/figures/cell_5_1_3.svg", width: 100%), kind: "subfigure", caption: [Refocused image]) <fig8-c>]), caption: [Refocusing on the books using depth map], kind: image) <fig8>


== Refocusing using sub-aperture images

The principle of the refocusing using sub-aperture images is to use the multi view acquisition.
The multi view acquisition can be obtained by multiple cameras that takes the same scene from different angles.
It can also be obtained by a camera on a moving platform that takes multiple captures of the same scene from different angles.
Or the multi view acquisition can be obtained by a camera that has a microlens array placed in front of the sensor allowing to capture the scene from different angles in a single capture, as in plenoptic cameras.

Then, the shift and sum refocusing technique can be obtained by a combination of all views of the multi view acquisition.
Some views are shown on @fig10. By showing the circle on the left of the images, we can see that the circle is at different positions in each view, which confirms that the views are taken from different angles.
The views must then be aligned according to the desired refocus depth, and then summed together to obtain the refocused image.
So there is no need of a depth map to perform the refocusing.

If we look at the results in @fig11, we can see that the refocusing gives a good result at the refocused depth.
However, the blur is not very realistic due to the multi view acquisition: indeed, the blur is like a squared blur since all acquisitions are aligned and summed together with different angles, making the blur not very natural.
This is probably due to the fact that the multi view acquisition is made from different positions, which makes the alignment of the views not perfect and therefore the blur not very realistic.

So this refocusing technique works quite well even if the result is not very realistic: in fact, this method can be improved to have a more realistic blur.
For instance, thanks to the multi view acquisition, a depth map can be easily estimated, allowing to perform therefore a better refocusing by applying different blur filters according to the depth map, as seen in the previous part of this work.

#figure(
graph-canvas({

  let cam = camera
  let row = cam + " " + cam + " " + cam
  let title = row + "\n" + row + "\n" + row

  let cameras = make-box(
    title,
    size: (2, 2),
    legend: "Multi view acquisition",
    color: rgb("#f0f0f0"),
  )

  let images = make-image-dataset(
    "Captures",
    src: "../../../.typst_pyexec/figures/cell_6_1_1.svg",
    images: 8,
    image-size: (2, 2),
    image-spacing: 0.06,
    image-shift-x: -2,
    image-pad: -0.05,
    after: cameras,
  )

  let img1 = make-image-node(
    "",
    src: "../../../.typst_pyexec/figures/cell_6_1_1.svg",
    image-width: 1,
    image-height: 1,
    image-shift-x: -2,
    image-pad: -0.05,
    after: images,
    y: 1.5,
  )

  let img2 = make-image-node(
    "",
    src: "../../../.typst_pyexec/figures/cell_6_1_1.svg",
    image-width: 1,
    image-height: 1,
    image-shift-x: -2,
    image-pad: -0.05,
    after: images,

  )

  let img3 = make-image-node(
    "",
    src: "../../../.typst_pyexec/figures/cell_6_1_1.svg",
    image-width: 1,
    image-height: 1,
    image-shift-x: -2,
    image-pad: -0.05,
    after: images,
    y: -1.5,
  )

  let shift1 = make-box(
    "Shift",
    after: img1,
    size: (1, 1)
  )

  let shift2 = make-box(
    "Shift",
    after: img2,
    size: (1, 1)
  )

  let shift3 = make-box(
    "Shift",
    after: img3,
    size: (1, 1)
  )

  
  let sum = make-trapezoid(
    "Sum",
    after: shift2,
    width: 1.3,
    big-half: 1.5,
    small-half: 1,
  )

  let refocus = make-image-node(
    "Refocused image",
    src: "../../../.typst_pyexec/figures/cell_7_1_1.svg",
    image-size: (2, 2),
    image-pad: -0.05,
    image-shift-x: -1.5,
    after: sum,
  )

  let nodes = (cameras, images, sum, img1, img2, img3, shift1, shift2, shift3, refocus)

  let arrows = (
    make-arrow(cameras, images, from-outer: true, auto-spacing: true),
    make-arrow(images, img1, from-outer: true),
    make-arrow(images, img2, from-outer: true),
    make-arrow(images, img3, from-outer: true),
    make-arrow(img1, shift1),
    make-arrow(img2, shift2),
    make-arrow(img3, shift3),
    make-arrow(shift1, sum),
    make-arrow(shift2, sum),
    make-arrow(shift3, sum),
    make-arrow(sum, refocus)
  )

  draw-graph(nodes: nodes, arrows: arrows)

}), caption: "Refocusing using sub-aperture images") <fig9>

#figure(grid(columns: 3, inset: 6pt, [#figure(image(".typst_pyexec/figures/cell_6_1_1.svg"), kind: "subfigure") <fig10-a>], [#figure(image(".typst_pyexec/figures/cell_6_1_2.svg"), kind: "subfigure") <fig10-b>], [#figure(image(".typst_pyexec/figures/cell_6_1_3.svg"), kind: "subfigure") <fig10-c>], [#figure(image(".typst_pyexec/figures/cell_6_1_4.svg"), kind: "subfigure") <fig10-d>], [#figure(image(".typst_pyexec/figures/cell_6_1_5.svg"), kind: "subfigure") <fig10-e>], [#figure(image(".typst_pyexec/figures/cell_6_1_6.svg"), kind: "subfigure") <fig10-f>], [#figure(image(".typst_pyexec/figures/cell_6_1_7.svg"), kind: "subfigure") <fig10-g>], [#figure(image(".typst_pyexec/figures/cell_6_1_8.svg"), kind: "subfigure") <fig10-h>], [#figure(image(".typst_pyexec/figures/cell_6_1_9.svg"), kind: "subfigure") <fig10-i>]), caption: [Sub-aperture images], kind: image) <fig10>


#figure(grid(columns: 3, inset: 6pt, [#figure(image(".typst_pyexec/figures/cell_7_1_1.svg"), kind: "subfigure", caption: [Z = 2 meters]) <fig11-a>], [#figure(image(".typst_pyexec/figures/cell_7_1_2.svg"), kind: "subfigure", caption: [Z = 2.5 meters]) <fig11-b>], [#figure(image(".typst_pyexec/figures/cell_7_1_3.svg"), kind: "subfigure", caption: [Z = 3.0 meters]) <fig11-c>], [#figure(image(".typst_pyexec/figures/cell_7_1_4.svg"), kind: "subfigure", caption: [Z = 3.5 meters]) <fig11-d>], [#figure(image(".typst_pyexec/figures/cell_7_1_5.svg"), kind: "subfigure", caption: [Z = 4.0 meters]) <fig11-e>], [#figure(image(".typst_pyexec/figures/cell_7_1_6.svg"), kind: "subfigure", caption: [Z = 4.5 meters]) <fig11-f>]), caption: [Refocused at Different Depths], kind: image) <fig11>


= Task 3: Image refocusing and depth estimation

In this task, we will study some multi view acquisition from a plenoptic camera taken from the github repository https://github.com/hahnec/plenopticam.

As the code was already delivered in the repository as a jupyter notebook, I only played with the code and the parameters to understand how it works and how the refocusing and depth map estimation can be performed with the multi view acquisition of a plenoptic camera.
However, by trying by myself to run the code, I had some issues with libraries and dependencies.
Indeed, the code was implemented with older libraries and some functions were deprecated, making the code not directly runnable.

The main principle of the plenoptic camera is to capture the scene from different angles thanks to a microlens array placed between the lens and the sensor, allowing to capture multiple views of the scene in a single capture.
The result is a multi view acquisition that can be exprimed as a 4D matrix (x, y, u, v) where (x, y) are the spatial coordinates and (u, v) are the angular coordinates of the light field.

The obtained multi view acquisition must then be processed to have a good representation of the scene.
Indeed, without processing, the output looks like an image with a lot of small squares, as shown on @fig12-a.

To process the image, the disposition and properties of the microlens array must be known, so a calibration step is necessary to perform good processing of the image.
Indeed, if some zoom was applied, the centroid of the microlenses is not at the same position than estimated, so the results can look messy.
To calibrate, some white images of calibration are available, as shown on @fig12-b.
Then, the calibration image taken must have same parameters than the image to process.





#figure(grid(columns: 2, [#figure(image(".typst_pyexec/figures/cell_8_1_1.svg"), kind: "subfigure", caption: [Raw Illum image]) <fig12-a>], [#figure(image(".typst_pyexec/figures/cell_8_1_2.svg"), kind: "subfigure", caption: [Raw white calibration image]) <fig12-b>]), caption: [Raw Illum image], kind: image) <fig12>


When the calibration is done, each sub-aperture image can be extracted from the multi view acquisition, as shown on @fig13.










#figure(image(".typst_pyexec/figures/cell_14_1.svg"), caption: [All sub-aperture images view]) <fig13>


Then, thanks to the multi view acquisition, we can take the view of the scene from a specific angle as shown on @fig14.
In fact, it corresponds to take all pixels of the multi view acquisition that corresponds to a specific angle, which is like taking all pixels of a pinhole camera with a very small aperture that captures only the light coming from that specific angle, allowing to have a large depth of field.
However, the price to pay is that the image is with a low resolution since only a portion of the pixels is used to create the image.
For instance, in our case, as we have around 100 views, the resolution of each view is around 1/100 of the overall resolution of the sensor.

As we can see on @fig14, the difference between views is very small (we can see that at the borders of the images).
Indeed, the acquisition is made with a microlens array, which means that the views are taken from close angles, making the difference between views quite small.
That's why to have good results, many views must be taken.

#figure(grid(columns: 3, [#figure(image(".typst_pyexec/figures/cell_15_1_1.svg"), kind: "subfigure", caption: [Left view]) <fig14-a>], [#figure(image(".typst_pyexec/figures/cell_15_1_2.svg"), kind: "subfigure", caption: [Central view]) <fig14-b>], [#figure(image(".typst_pyexec/figures/cell_15_1_3.svg"), kind: "subfigure", caption: [Right view]) <fig14-c>]), caption: [Left view], kind: image) <fig14>


Thanks to this multi view acquisition with a large depth of field, a depth map can be estimated by an analysis of the disparity between views, as shown on @fig15-a.

Indeed, an object that is close to the camera will move more between views compared to an object far from the camera, allowing to estimate distances of objects in the scene and therefore to estimate a depth map of the scene.

Thanks to these properties, a refocus can also be performed by a shift and sum of the views, as shown on @fig15-b.
Compared to the shift and sum refocus performed in task 2, the refocus performed with a plenoptic camera is more accurate with a better blur.
This is probably due to the fact that the multi view acquisition is made from a single capture at a single position with a microlens array, which allows to have a better alignment of the views and therefore a better refocus.
It is also possible to perform a refocus like in task 2 by applying different blur filters according to the depth map estimaded, or to perform a refocus using different method such as Scheimpflug refocusing that allows a refocus of an inclined plane in the scene, such as the ground for instance, which is not possible with the shift and sum refocusing technique, or with classic camera focusing.
The Scheimpflug refocusing corresponds to a shift and sum refocus with an affine shift that depends on the position of the pixels.

#figure(grid(columns: 2, [#figure(image(".typst_pyexec/figures/cell_16_1_1.svg"), kind: "subfigure", caption: [Depth map]) <fig15-a>], [#figure(image(".typst_pyexec/figures/cell_16_1_2.svg"), kind: "subfigure", caption: [Shift-and-Sum]) <fig15-b>]), caption: [Depth map], kind: image) <fig15>


So the pipeline of the processing of a multi view acquisition from a plenoptic camera can be summarized like in the @fig16.

#figure(
graph-canvas({
  let acquisition = make-image-node(
    "Multi view acquisition",
    src: "../../../.typst_pyexec/figures/cell_8_1_1.svg",
    image-size: (2.5, 2.5),
    image-pad: -0.05,
  )
  let calibration_database = make-image-dataset(
    "Database of calibration images",
    src: "../../../.typst_pyexec/figures/cell_8_1_2.svg",
    image-size: (2.5, 2.5),
    image-pad: -0.05,
    y: -4
  )
  let calibration_image = make-image-node(
    "Calibration image",
    src: "../../../.typst_pyexec/figures/cell_8_1_2.svg",
    image-size: (2.5, 2.5),
    image-pad: -0.05,
    after: calibration_database,
    gap: 2
  )
  let calibration = make-box(
    "Calibration",
    subtitle: "Alignment of microlenses",
    after: acquisition
  )
  let sub_aperture = make-image-dataset(
    "Sub-aperture images",
    src: "../../../.typst_pyexec/figures/cell_15_1_1.svg",
    image-size: (2.5, 2.5),
    image-pad: -0.05,
    after: calibration,
    y: -2
  )
  let depth = make-image-node(
    "Depth map estimation",
    src: "../../../.typst_pyexec/figures/cell_16_1_1.svg",
    image-size: (2.5, 2.5),
    image-pad: -0.05,
    after: sub_aperture,
    y: -3.5
  )
  let refocus = make-image-node(
    "Refocusing",
    src: "../../../.typst_pyexec/figures/cell_16_1_2.svg",
    image-size: (2.5, 2.5),
    image-pad: -0.05,
    after: sub_aperture,
    y: 0.0
  )

  let nodes = (acquisition, calibration_database, calibration_image, calibration, sub_aperture, depth, refocus)
  let arrows = (
    make-arrow(acquisition, calibration),
    make-arrow(calibration_database, calibration_image, label: "select", from-outer: true),
    make-arrow(calibration_image, calibration, out-side: "top", in-side: "bottom"),
    make-arrow(calibration, sub_aperture),
    make-arrow(sub_aperture, depth, from-outer: true),
    make-arrow(sub_aperture, refocus, from-outer: true)
  )
  draw-graph(nodes: nodes, arrows: arrows)
  }), caption: "Pipeline of the processing of a multi view acquisition from a plenoptic camera") <fig16>

= Task 4 (optional): Refocusing and depth estimation with neural networks

The depth map estimation can be made with a neural network instead of a random forest regressor.
Indeed, the depth estimation is complex and requires a lot of information to be accurate, so a neural network is more suitable to capture relationships between features and depth.
The general pipeline of the depth estimation with a neural network could be represented like in @fig17.
Note that the extraction of features is not necessary with a neural network: it can be handcrafted with fixed filters.
This can allow to speed up the training or to have a smaller dataset, but it can limit performances if handcrafted features are not relevant enough.

So depending on the architecture of the neural network used, the training must be applied on the encoder and decoder together, or only on the decoder if the encoder is handcrafted with fixed filters.

The big advantage of using a neural network is that when the network is trained, the depth estimation can be very fast and accurate.
However, the training of the network can be long and requires a lot of data and computational power, which can be problematic.

#figure(
graph-canvas({
  let image = make-image-dataset(
    "Input images",
    src: "../../../.typst_pyexec/figures/cell_15_1_2.svg",
    image-size: (2.5, 2.5),
    image-pad: -0.05,
    color: rgb("#a8c8e8"),
  )

  let encoder = make-trapezoid(
    "Encoder",
    subtitle: "Feature extraction",
    legend: "Can be trained or handcrafted\nwith fixed filters",
    after: image,
    gap: 2.0,
    color: rgb("#a0b8f5"),
  )

  let latent = make-latent-space(
    "Latent space",
     subtitle: "Capture depth information",
     after: encoder,
     gap: 1.79,
     color: rgb("#f5a0be"),
   )
  let detector = make-trapezoid(
    "Decoder",
    mode: "decoder",
    subtitle: "Depth map\nestimation",
    after: latent,
    gap: 1.79,
    color: rgb("#a0e8be"),
  )
  let computed = make-image-dataset(
    "Predicted depth map",
    src: "../../../.typst_pyexec/figures/cell_16_1_1.svg",
    image-size: (2.5, 2.5),
    image-pad: -0.05,
    after: detector,
    gap: 2.0,
    color: rgb("#f0f0f0"),
  )
  let nodes = (image, encoder, latent, detector, computed)
  let arrows = (
    make-arrow(image, encoder),
    make-arrow(encoder, latent),
    make-arrow(latent, detector),
    make-arrow(detector, computed)
  )
  draw-graph(nodes: nodes, arrows: arrows)
}), caption: "Pipeline of the depth estimation with a neural network") <fig17>

In case of refocusing, a neural network can also be built.
The training can be made on single acquisition with a depth map, or on single acquisition without depth map, or on multi view acquisition, etc.
In all cases, the principle stays the same: learning the relationship between the non-refocused input and the refocused image at a specific depth.
Due to the wide variety of possible inputs and outputs, the architecture of the neural network can be very different from one case to another.
For instance, if we dispose of a multi view acquisition, the relationship between the input and output will not be the same than if we only have a single acquisition with no depth map.
Indeed, if we dispose already of a multi view acquisition, we have already a lot of features so only a decoder is necessary whereas with single acquisition with no depth map, an encoder is necessary to extract features. Note that it can also be handcrafted depending on the case:
for instance, the features set can be some unfocused images computed with different filters or something else.

So the neural network can be very interesting for refocusing and depth estimation as they are able to learn complex relationships between inputs and outputs.
However, the training can be long and requires a lot of data and computational power, which can be problematic.

#pagebreak()

= Conclusion

In this work, we have implemented first a depth map estimation based on a single image by extracting colors, textures and edge features and processing them together with a random forest regressor to estimate the depth map of the scene.
The results are not very accurate and quite noisy, which is due to the complexity of the depth estimation problem that requires a lot of informations and also due to the database wich contains only small depth maps for images and computational power.

Then, a study of refocusing techniques has been done, based first on depth map information and then on multi view acquisition.
With the depth map, the refocusing is quite good and accurate, and the blur is realistic thanks to the depth information allowing to apply different blur filters according to the depth of the pixels.
With the multi view acquisition, the refocusing don't need a depth map.
However, due to the quality of the sub-aperture images, the blur is not very realistic even if the refocus is good at the refocused depth.

Finally, a study of the refocusing and depth estimation with a plenoptic camera has been done.
The processing and calibration takes some time and the output image has smaller resolution than the sensor resolution, but the results are quite good and allows to change angle of view, refocus using different techniques, compute a depth map, etc.
So the plenoptic camera is very interesting for computational imaging and offers a lot of possibilities in terms of post-processing.

The depth estimation and refocusing can also be performed using neural networks thanks to the ability of neural networks to learn complex relationships between inputs and outputs.
But the training can be long and requires a lot of data that are not always available, and some computational power to obtain good results.
That's why plenoptic cameras and simple algorithms stay very interesting in term of computational imaging, even if the results are not perfect.

We may wonder if the video allows good depth estimation and refocusing thanks to the multi frames capture, which provides much more information than a single capture.