// Main report file
#import "template.typ": make-report, report-footnote
#import "metadata.typ": my-report
#import "@preview/theofig:0.1.0": definition
#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node
#import "@preview/cetz:0.3.1": canvas, draw

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

#canvas(length: 1.1cm, {
  import draw: *

  // ── Palette ──────────────────────────────────────────────────────
  let ci  = rgb("#a8c8e8")   // image input
  let ce  = rgb("#b0dba0")   // encoder (green)
  let cl  = rgb("#fcd97a")   // latent space (amber)
  let cd  = rgb("#a0b8f5")   // depth estimator (blue)
  let co  = rgb("#a0e8be")   // computed output maps
  let ccp = rgb("#f5a0be")   // comparison block (pink)
  let cdm = rgb("#c5a0f0")   // depth-map input (violet)
  let sk  = 0.22             // inter-layer offset for stacked images

  // ── Helper: n stacked rectangles, back→front, centred at (px, py) ──
  let stk(px, py, w, h, n, col) = {
    for i in range(n) {
      let j   = n - 1 - i      // j: n-1 … 0  (back → front on top)
      let off = j * sk
      rect(
        (px - w/2 + off, py - h/2 + off),
        (px + w/2 + off, py + h/2 + off),
        fill: col,
        stroke: (paint: black, thickness: 0.5pt),
      )
    }
  }

  // ── Arrow style ──────────────────────────────────────────────────
  let arst = (paint: black, thickness: 0.75pt)
  let armk = (end: ">")


  // ════════════════════════════════════════════════════════════════
  //  NODES
  // ════════════════════════════════════════════════════════════════

  // 1 · Image Dataset  (x=1.0, y=0.0) ─────────────────────────────
  stk(1.0, 0.0, 1.7, 2.1, 3, ci)
  content((1.0, 0.0), align(center)[#text(size: 0.68em)[Image \ Dataset]])

  // 2 · Depth Map Dataset  (x=1.0, y=−3.8) ────────────────────────
  stk(1.0, -3.8, 1.7, 2.1, 3, cdm)
  content((1.0, -3.8), align(center)[#text(size: 0.68em)[Depth Map \ Dataset]])

  // 3 · Encoder — trapezoid ssiDepthChns ───────────────────────────
  //    Left (input) tall: y ±1.65 ;  Right (output) narrow: y ±0.80
  line(
    (3.3, -1.65), (6.1, -0.80),
    (6.1,  0.80), (3.3,  1.65),
    close: true, fill: ce,
    stroke: (paint: black, thickness: 0.65pt),
  )
  content((4.70,  0.30), align(center)[#text(size: 0.72em)[*ssiDepthChns*]])
  content((4.70, -0.35), align(center)[#text(size: 0.58em, fill: rgb("#333333"))[Encoder]])

  // 4 · Latent Space — tall narrow rectangle ───────────────────────
  //    x = [7.1, 7.95], y = [−2.0, 2.0]
  rect((7.1, -2.0), (7.95, 2.0),
    fill: cl, stroke: (paint: black, thickness: 0.65pt))
  content((7.525, 0.0),
    rotate(-90deg)[#text(size: 0.56em)[H × W × \#Features]])
  content((7.525, -2.6),
    align(center)[#text(size: 0.62em)[Latent Space]])

  // 5 · Depth Map Estimator ─────────────────────────────────────────
  //    x = [9.0, 12.4], y = [−1.1, 1.1]
  rect((9.0, -1.1), (12.4, 1.1),
    fill: cd, stroke: (paint: black, thickness: 0.65pt))
  content((10.7,  0.30), align(center)[#text(size: 0.70em)[*ssiDepthDetect*]])
  content((10.7, -0.35), align(center)[#text(size: 0.58em)[Depth Map Estimator]])

  // 6 · Computed Depth Maps  (x=13.9, y=0.0) ───────────────────────
  stk(13.9, 0.0, 1.7, 2.1, 3, co)
  content((13.9, 0.0), align(center)[#text(size: 0.68em)[Computed \ Depth Maps]])

  // 7 · Comparison Block ────────────────────────────────────────────
  //    x = [15.9, 18.5], y = [−2.95, −0.75]
  rect((15.9, -2.95), (18.5, -0.75),
    fill: ccp, stroke: (paint: black, thickness: 0.65pt))
  content((17.2, -1.60), align(center)[#text(size: 0.70em)[*Comparison Block*]])
  content((17.2, -2.20), align(center)[#text(size: 0.58em, fill: rgb("#444444"))[Loss / Evaluation]])

  // ── Small ⊗ icon inside comparison to hint at difference op ─────
  circle((16.15, -1.85), radius: 0.22,
    stroke: (paint: rgb("#880044"), thickness: 0.6pt), fill: white)
  content((16.15, -1.85), text(size: 0.55em, fill: rgb("#880044"))[⊗])


  // ════════════════════════════════════════════════════════════════
  //  ARROWS
  // ════════════════════════════════════════════════════════════════

  // 1 → 3  Image Dataset → Encoder
  //   exit: back-image right ≈ (1+0.85+0.44, 0+0.22) = (2.29, 0.22)
  line((2.3, 0.22), (3.3, 0.0), mark: armk, stroke: arst)

  // 3 → 4  Encoder → Latent Space
  line((6.1, 0.0), (7.1, 0.0), mark: armk, stroke: arst)

  // 4 → 5  Latent Space → Depth Map Estimator
  line((7.95, 0.0), (9.0, 0.0), mark: armk, stroke: arst)

  // 5 → 6  Depth Map Estimator → Computed Depth Maps
  //   enter front-image left ≈ x=13.9−0.85=13.05
  line((12.4, 0.0), (13.05, 0.22), mark: armk, stroke: arst)

  // 6 → 7  Computed Depth Maps → Comparison Block
  //   exit front-image bottom (13.9, −1.05) → bend down → enter comparison left
  line(
    (13.9, -1.05),
    (13.9, -1.85),
    (15.9, -1.85),
    mark: armk, stroke: arst,
  )

  // 2 → 7  Depth Map Dataset → Comparison Block
  //   exit right (2.3, −3.6) → long horizontal → enter comparison bottom-centre
  line(
    (2.3,  -3.6),
    (17.2, -3.6),
    (17.2, -2.95),
    mark: armk, stroke: arst,
  )


  // ════════════════════════════════════════════════════════════════
  //  EDGE LABELS
  // ════════════════════════════════════════════════════════════════

  content((2.85, 0.45), text(size: 0.52em, fill: rgb("#555555"))[input])
  content((6.55, 0.30), text(size: 0.52em, fill: rgb("#555555"))[features])
  content((8.55, 0.30), text(size: 0.52em, fill: rgb("#555555"))[decode])
  content((14.9, -1.60), text(size: 0.52em, fill: rgb("#555555"))[↓ compare])
  content((9.0, -3.35),  text(size: 0.52em, fill: rgb("#555555"))[ground truth →])
})



= Task 2: Image refocusing

= Task 3: Image refocusing and depth estimation



= Implementation <impl>

= Results

= Discussion

= Conclusion

