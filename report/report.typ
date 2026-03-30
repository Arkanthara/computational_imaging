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

  // ── Global controls: edit these first ───────────────────────────
  let arrow-gap = 1.0            // fixed center-line gap between linked nodes
  let stack-shift = 0.22         // dataset/computed stack offset
  let latent-shift = stack-shift / 2

  // ── Palette ──────────────────────────────────────────────────────
  let ci  = rgb("#a8c8e8")
  let ce  = rgb("#b0dba0")
  let cl  = rgb("#fcd97a")
  let cd  = rgb("#a0b8f5")
  let co  = rgb("#a0e8be")
  let ccp = rgb("#f5a0be")
  let cdm = rgb("#c5a0f0")

  // ── Node geometry ────────────────────────────────────────────────
  let y-main = 0.0
  let y-dset = -3.8

  let img-ds = (kind: "stack", cx: 1.0, cy: y-main, w: 1.9, h: 2.3, n: 3, shift: stack-shift)
  let dep-ds = (kind: "stack", cx: 1.0, cy: y-dset, w: 1.9, h: 2.3, n: 3, shift: stack-shift)

  // Encoder placed automatically from image dataset + arrow-gap
  let enc-left = (img-ds.cx + img-ds.w/2 + (img-ds.n - 1) * img-ds.shift) + arrow-gap
  let enc = (lx: enc-left, rx: enc-left + 2.8, cy: y-main, h-left: 1.65, h-right: 0.80)

  // Latent stack placed automatically from encoder + arrow-gap
  let lat-left = enc.rx + arrow-gap
  let lat = (kind: "stack", cx: lat-left + 1.15/2, cy: y-main, w: 1.15, h: 1.6, n: 6, shift: latent-shift)

  // Estimator placed automatically from latent outer-right + arrow-gap
  let lat-right-outer = lat.cx + lat.w/2 + (lat.n - 1) * lat.shift
  let est-left = lat-right-outer + arrow-gap
  let est = (kind: "box", cx: est-left + 3.6/2, cy: y-main, w: 3.6, h: 2.2)

  // Computed maps placed automatically from estimator + arrow-gap
  let cmp-front-left = (est.cx + est.w/2) + arrow-gap
  let cmp = (kind: "stack", cx: cmp-front-left + 1.9/2, cy: y-main, w: 1.9, h: 2.3, n: 3, shift: stack-shift)

  // Comparison block placed automatically from computed outer-right + arrow-gap
  let cmp-right-outer = cmp.cx + cmp.w/2 + (cmp.n - 1) * cmp.shift
  let eval = (kind: "box", cx: cmp-right-outer + arrow-gap + 2.6/2, cy: -1.85, w: 2.6, h: 2.2)

  // ── Helpers ──────────────────────────────────────────────────────
  let stk(node, col) = {
    for i in range(node.n) {
      let j = node.n - 1 - i
      let off = j * node.shift
      rect(
        (node.cx - node.w/2 + off, node.cy - node.h/2 + off),
        (node.cx + node.w/2 + off, node.cy + node.h/2 + off),
        fill: col,
        stroke: (paint: black, thickness: 0.5pt),
      )
    }
  }

  // side ∈ {left,right,top,bottom,center}, outer=true uses full stack footprint
  let node-anchor(node, side, outer: false) = {
    let ox = if node.kind == "stack" and outer { (node.n - 1) * node.shift } else { 0 }
    let oy = if node.kind == "stack" and outer { (node.n - 1) * node.shift } else { 0 }

    if side == "left" {
      (node.cx - node.w/2, node.cy)
    } else if side == "right" {
      (node.cx + node.w/2 + ox, node.cy)
    } else if side == "top" {
      (node.cx, node.cy + node.h/2 + oy)
    } else if side == "bottom" {
      (node.cx, node.cy - node.h/2)
    } else {
      (node.cx, node.cy)
    }
  }

  let edge-label(a, b, txt, side: "above", gap: 0.22, dx: 0.0, dy: 0.0) = {
    let mx = (a.at(0) + b.at(0)) / 2 + dx
    let my = (a.at(1) + b.at(1)) / 2 + dy
    let horiz = calc.abs(a.at(1) - b.at(1)) < 0.001
    let ox = if horiz { 0 } else if side == "right" { gap } else { -gap }
    let oy = if horiz { if side == "below" { -gap } else { gap } } else { 0 }
    content((mx + ox, my + oy), text(size: 0.48em, fill: rgb("#555555"))[#txt])
  }

  let arst = (paint: black, thickness: 0.75pt)
  let armk = (end: ">")

  // ════════════════════════════════════════════════════════════════
  //  NODES
  // ════════════════════════════════════════════════════════════════

  stk(img-ds, ci)
  content((img-ds.cx, img-ds.cy), align(center)[#text(size: 0.58em)[Image \ Dataset]])

  stk(dep-ds, cdm)
  content((dep-ds.cx, dep-ds.cy), align(center)[#text(size: 0.58em)[Depth Map \ Dataset]])

  // Encoder trapezoid
  line(
    (enc.lx, enc.cy - enc.h-left), (enc.rx, enc.cy - enc.h-right),
    (enc.rx, enc.cy + enc.h-right), (enc.lx, enc.cy + enc.h-left),
    close: true, fill: ce,
    stroke: (paint: black, thickness: 0.65pt),
  )
  content(((enc.lx + enc.rx)/2,  0.30), align(center)[#text(size: 0.66em)[*ssiDepthChns*]])
  content(((enc.lx + enc.rx)/2, -0.35), align(center)[#text(size: 0.54em, fill: rgb("#333333"))[Encoder]])

  stk(lat, cl)
  content((lat.cx + 0.15, -1.55), align(center)[#text(size: 0.50em)[H × W × \#Features]])

  rect(
    (est.cx - est.w/2, est.cy - est.h/2),
    (est.cx + est.w/2, est.cy + est.h/2),
    fill: cd,
    stroke: (paint: black, thickness: 0.65pt),
  )
  content((est.cx,  0.30), align(center)[#text(size: 0.64em)[*ssiDepthDetect*]])
  content((est.cx, -0.35), align(center)[#text(size: 0.54em)[Depth Map Estimator]])

  stk(cmp, co)
  content((cmp.cx, cmp.cy), align(center)[#text(size: 0.54em)[Computed \ Depth Maps]])

  rect(
    (eval.cx - eval.w/2, eval.cy - eval.h/2),
    (eval.cx + eval.w/2, eval.cy + eval.h/2),
    fill: ccp,
    stroke: (paint: black, thickness: 0.65pt),
  )
  content((eval.cx, -1.60), align(center)[#text(size: 0.56em)[*Comparison \ Block*]])
  content((eval.cx, -2.20), align(center)[#text(size: 0.52em, fill: rgb("#444444"))[Loss / Evaluation]])

  // ════════════════════════════════════════════════════════════════
  //  ARROWS (orthogonal only)
  // ════════════════════════════════════════════════════════════════

  // Datasets use front-image center as node center, but outer-right as anchor to avoid overlap
  let a-in-a = node-anchor(img-ds, "right", outer: true)
  let a-in-b = (enc.lx, y-main)
  line(a-in-a, a-in-b, mark: armk, stroke: arst)

  let a-feat-a = (enc.rx, y-main)
  let a-feat-b = node-anchor(lat, "left")
  line(a-feat-a, a-feat-b, mark: armk, stroke: arst)

  let a-dec-a = node-anchor(lat, "right", outer: true)
  let a-dec-b = node-anchor(est, "left")
  line(a-dec-a, a-dec-b, mark: armk, stroke: arst)

  let a-est-a = node-anchor(est, "right")
  let a-est-b = node-anchor(cmp, "left") // to first-image middle
  line(a-est-a, a-est-b, mark: armk, stroke: arst)

  let a-cmp-a = node-anchor(cmp, "bottom")
  let a-cmp-b = node-anchor(eval, "left")
  let a-cmp-k = (a-cmp-a.at(0), a-cmp-b.at(1))
  line(a-cmp-a, a-cmp-k, a-cmp-b, mark: armk, stroke: arst)

  let a-gt-a = node-anchor(dep-ds, "right", outer: true)
  let a-gt-b = node-anchor(eval, "bottom")
  let a-gt-k = (a-gt-b.at(0), a-gt-a.at(1))
  line(a-gt-a, a-gt-k, a-gt-b, mark: armk, stroke: arst)

  // ════════════════════════════════════════════════════════════════
  //  EDGE LABELS
  // ════════════════════════════════════════════════════════════════
  edge-label(a-in-a, a-in-b, [input], side: "above")
  edge-label(a-feat-a, a-feat-b, [features], side: "above")
  edge-label(a-dec-a, a-dec-b, [decode], side: "below", dx: -0.06)
  edge-label(a-cmp-k, a-cmp-b, [compare], side: "above")
  edge-label(a-gt-a, a-gt-k, [ground truth], side: "above")
})
]
]



= Task 2: Image refocusing

= Task 3: Image refocusing and depth estimation



= Implementation <impl>

= Results

= Discussion

= Conclusion

