// Main report file
#import "template.typ": make-report, report-footnote
#import "metadata.typ": my-report
#import "@preview/theofig:0.1.0": definition
#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node
#import "@preview/cetz:0.4.2": canvas, draw

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

#align(center)[
#scale(88%)[
#canvas(length: 0.72cm, {
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
  let stk(px, py, w, h, n, col, shift: sk) = {
    for i in range(n) {
      let j   = n - 1 - i      // j: n-1 … 0  (back → front on top)
      let off = j * shift
      rect(
        (px - w/2 + off, py - h/2 + off),
        (px + w/2 + off, py + h/2 + off),
        fill: col,
        stroke: (paint: black, thickness: 0.5pt),
      )
    }
  }

  // ── Auto label for straight arrow segments (with manual nudges) ──
  let edge-label(a, b, txt, side: "above", gap: 0.22, dx: 0.0, dy: 0.0) = {
    let ax = a.at(0)
    let ay = a.at(1)
    let bx = b.at(0)
    let by = b.at(1)
    let mx = (ax + bx) / 2 + dx
    let my = (ay + by) / 2 + dy
    let horiz = calc.abs(ay - by) < 0.001
    let ox = if horiz {
      0
    } else if side == "right" {
      gap
    } else {
      -gap
    }
    let oy = if horiz {
      if side == "below" { -gap } else { gap }
    } else {
      0
    }
    content((mx + ox, my + oy), text(size: 0.48em, fill: rgb("#555555"))[#txt])
  }

  // ── Arrow style ──────────────────────────────────────────────────
  let arst = (paint: black, thickness: 0.75pt)
  let armk = (end: ">")


  // ════════════════════════════════════════════════════════════════
  //  NODES
  // ════════════════════════════════════════════════════════════════

  // 1 · Image Dataset  (x=1.0, y=0.0) ─────────────────────────────
  stk(1.0, 0.0, 1.9, 2.3, 3, ci)
  content((1.0, 0.0), align(center)[#text(size: 0.58em)[Image \ Dataset]])

  // 2 · Depth Map Dataset  (x=1.0, y=−3.8) ────────────────────────
  stk(1.0, -3.8, 1.9, 2.3, 3, cdm)
  content((1.0, -3.8), align(center)[#text(size: 0.58em)[Depth Map \ Dataset]])

  // 3 · Encoder — trapezoid ssiDepthChns ───────────────────────────
  //    Left (input) tall: y ±1.65 ;  Right (output) narrow: y ±0.80
  line(
    (3.3, -1.65), (6.1, -0.80),
    (6.1,  0.80), (3.3,  1.65),
    close: true, fill: ce,
    stroke: (paint: black, thickness: 0.65pt),
  )
  content((4.70,  0.30), align(center)[#text(size: 0.66em)[*ssiDepthChns*]])
  content((4.70, -0.35), align(center)[#text(size: 0.54em, fill: rgb("#333333"))[Encoder]])

  // 4 · Latent Space — stacked feature maps (2x more layers) ──────
  //    6 layers, each with a height close to the trapezoid small side (≈1.6)
  stk(8.1, 0.0, 1.15, 1.6, 6, cl, shift: sk / 2)
  content((8.25, -1.55), align(center)[#text(size: 0.50em)[H × W × \#Features]])

  // 5 · Depth Map Estimator ─────────────────────────────────────────
  //    x = [10.225, 13.85], y = [−1.1, 1.1]
  rect((10.225, -1.1), (13.85, 1.1),
    fill: cd, stroke: (paint: black, thickness: 0.65pt))
  content((12.0375,  0.30), align(center)[#text(size: 0.64em)[*ssiDepthDetect*]])
  content((12.0375, -0.35), align(center)[#text(size: 0.54em)[Depth Map Estimator]])

  // 6 · Computed Depth Maps  (x=15.8, y=0.0) ───────────────────────
  stk(15.8, 0.0, 1.9, 2.3, 3, co)
  content((15.8, 0.0), align(center)[#text(size: 0.54em)[Computed \ Depth Maps]])

  // 7 · Comparison Block ────────────────────────────────────────────
  //    x = [18.7, 21.3], y = [−2.95, −0.75]
  rect((18.7, -2.95), (21.3, -0.75),
    fill: ccp, stroke: (paint: black, thickness: 0.65pt))
  content((20.0, -1.60), align(center)[#text(size: 0.56em)[*Comparison \ Block*]])
  content((20.0, -2.20), align(center)[#text(size: 0.52em, fill: rgb("#444444"))[Loss / Evaluation]])

  // ════════════════════════════════════════════════════════════════
  //  ARROWS
  // ════════════════════════════════════════════════════════════════

  // 1 → 3  Image Dataset → Encoder
  //   exit: stack outer-right at mid-height (avoids crossing inner layers)
  line(
    (2.39, 0.0),
    (2.6, 0.0),
    (3.3, 0.0),
    mark: armk, stroke: arst,
  )

  // 3 → 4  Encoder → Latent Space
  line((6.1, 0.0), (7.525, 0.0), mark: armk, stroke: arst)

  // 4 → 5  Latent Space → Depth Map Estimator (stepped like dataset arrow)
  line(
    (9.225, 0.0),
    (9.55, 0.0),
    (10.225, 0.0),
    mark: armk, stroke: arst,
  )

  // 5 → 6  Depth Map Estimator → Computed Depth Maps
  //   enter front-image left-middle (x=15.8−0.95=14.85, y=0)
  line(
    (13.85, 0.0),
    (14.85, 0.0),
    mark: armk, stroke: arst,
  )

  // 6 → 7  Computed Depth Maps → Comparison Block
  //   exit front-image bottom (13.9, −1.05) → bend down → enter comparison left
  line(
    (15.8, -1.15),
    (15.8, -1.85),
    (18.7, -1.85),
    mark: armk, stroke: arst,
  )

  // 2 → 7  Depth Map Dataset → Comparison Block
  //   exit stack outer-right at mid-height (avoids crossing inner layers)
  line(
    (2.39, -3.8),
    (20.0, -3.8),
    (20.0, -2.95),
    mark: armk, stroke: arst,
  )


  // ════════════════════════════════════════════════════════════════
  //  EDGE LABELS
  // ════════════════════════════════════════════════════════════════

  edge-label((2.6, 0.0), (3.3, 0.0), [input], side: "above", dy: 0.02)
  edge-label((6.1, 0.0), (7.525, 0.0), [features], side: "above")
  edge-label((9.55, 0.0), (10.225, 0.0), [decode], side: "below", dx: -0.08, dy: -0.02)
  edge-label((15.8, -1.85), (18.7, -1.85), [compare], side: "above")
  edge-label((2.39, -3.8), (20.0, -3.8), [ground truth], side: "above", dy: 0.02)
})
]
]



= Task 2: Image refocusing

= Task 3: Image refocusing and depth estimation



= Implementation <impl>

= Results

= Discussion

= Conclusion

