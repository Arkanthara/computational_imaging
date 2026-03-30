= Graph Utils Documentation

This document describes all public utilities in `graph_utils.typ`.

#import "@preview/cetz:0.4.2": canvas, draw
#import "graph_utils.typ": *

#set heading(numbering: "1.")
#outline(title: [Table of Contents], depth: 3)

== Compact Cheat Sheet

- `make-dataset(...)`: stacked image dataset node.
- `make-trapezoid(...)`: encoder/decoder trapezoid node.
- `make-box(...)`: rectangular processing node.
- `draw-node(node)`: render any node.
- `make-arrow(from, to, ...)`: build routed arrow with labels.
- `draw-arrow(arr)`: render arrow route.
- `draw-node-emoji(node, ...)`: place lock/key/custom emoji near node.
- `set-arrow-defaults(...)`, `set-dataset-defaults(...)`, `set-trapezoid-defaults(...)`, `set-box-defaults(...)`: preconfigured constructors.

Most used one-liners:

```typ
#let ds = make-dataset("Image\nDataset", pos: (1, 0))
#let enc = make-trapezoid("FeatureEncoder", subtitle: "Encoder", after: ds, gap: 1.2)
#let arr = make-arrow(ds, enc, from-outer: true, auto-spacing: true, label: [input])
#draw-node(ds)
#draw-node(enc)
#draw-arrow(arr)
```

== Quick Start

```typ
#import "@preview/cetz:0.4.2": canvas, draw
#import "graph_utils.typ": *

#let make-arrow = set-arrow-defaults(options: (spacing: 0.45, label-gap: 0.30))
#let make-trapezoid = set-trapezoid-defaults(options: (width: 2.8, big-half: 1.65, small-half: 0.80))

#canvas(length: 0.72cm, {
  import draw: *

  let ds = make-dataset("Image\nDataset", pos: (1.0, 0.0), images: 3)
  let enc = make-trapezoid("FeatureEncoder", subtitle: "Encoder", after: ds, gap: 1.2)
  let box = make-box("DepthRegressor", subtitle: "Depth Map\nEstimator", after: enc, gap: 1.2)

  draw-node(ds)
  draw-node(enc)
  draw-node(box)

  let a = make-arrow(ds, enc, from-outer: true, auto-spacing: true, label: [input])
  let b = make-arrow(enc, box, auto-spacing: true, label: [features])
  draw-arrow(a)
  draw-arrow(b)
})
```

Rendered output:

#canvas(length: 0.72cm, {
  import draw: *

  let ds = make-dataset("Image\nDataset", pos: (1.0, 0.0), images: 3)
  let enc = make-trapezoid("FeatureEncoder", subtitle: "Encoder", after: ds, gap: 1.2)
  let box = make-box("DepthRegressor", subtitle: "Depth Map\nEstimator", after: enc, gap: 1.2)

  draw-node(ds)
  draw-node(enc)
  draw-node(box)

  let a = make-arrow(ds, enc, from-outer: true, auto-spacing: true, label: [input])
  let b = make-arrow(enc, box, auto-spacing: true, label: [features])
  draw-arrow(a)
  draw-arrow(b)
})

== String Utilities

=== `clip-str(txt, n)`
Clips a string to the first `n` grapheme clusters.

```typ
#clip-str("Computed Depth Maps", 8) // "Computed"
#clip-str("😀😀😀😀", 2)             // keeps emoji boundaries
```

Rendered output: #clip-str("Computed Depth Maps", 8), #h(1em) #clip-str("😀😀😀😀", 2)

=== `truncate-title(txt, enabled: true, max-chars: 18)`
Truncates with `...` when enabled.

```typ
#truncate-title("Very long title", enabled: true, max-chars: 8)
#truncate-title("Very long title", enabled: false)
```

Rendered output: #truncate-title("Very long title", enabled: true, max-chars: 8), #h(1em) #truncate-title("Very long title", enabled: false)

=== `fit-lines(txt, max-chars: 16, max-lines: 2)`
Current behavior: manual only (no auto wrap). If text contains `\n`, line breaks are preserved.

```typ
#fit-lines("Line 1\nLine 2")
#fit-lines("No auto wrapping applied")
```

Rendered output: #fit-lines("Line 1\nLine 2") / #fit-lines("No auto wrapping applied")

=== `chars-cap(width, min: 8, scale: 4.2)`
Computes estimated character capacity from width.

```typ
#chars-cap(2.0)
#chars-cap(4.0, min: 10, scale: 5.0)
```

Rendered output: #chars-cap(2.0), #h(1em) #chars-cap(4.0, min: 10, scale: 5.0)

== Geometry Utilities

=== `node-size(node, outer: false)`
Returns `(width, height)` of a node; stack nodes include offset when `outer: true`.

```typ
#let ds = make-dataset("A", images: 3, image-size: (1.9, 2.3), image-spacing: 0.22)
#node-size(ds)
#node-size(ds, outer: true)
```

Rendered output: #node-size(make-dataset("A", images: 3, image-size: (1.9, 2.3), image-spacing: 0.22)), #h(1em) #node-size(make-dataset("A", images: 3, image-size: (1.9, 2.3), image-spacing: 0.22), outer: true)

=== `node-edge(node, side: "right", outer: false, outer-value: none)`
Returns one edge coordinate (`x` for left/right, `y` for top/bottom).

```typ
#let ds = make-dataset("A", images: 3, image-size: (1.9, 2.3), image-spacing: 0.22)
#node-edge(ds, side: "right")
#node-edge(ds, side: "right", outer: true)
#node-edge(ds, side: "right", outer: true, outer-value: 0.8)
```

Rendered output: #node-edge(make-dataset("A", images: 3, image-size: (1.9, 2.3), image-spacing: 0.22), side: "right"), #h(1em) #node-edge(make-dataset("A", images: 3, image-size: (1.9, 2.3), image-spacing: 0.22), side: "right", outer: true), #h(1em) #node-edge(make-dataset("A", images: 3, image-size: (1.9, 2.3), image-spacing: 0.22), side: "right", outer: true, outer-value: 0.8)

=== `node-anchor(node, side: "right", outer: false, outer-value: none)`
Returns `(x, y)` anchor for a node side.

```typ
#let box = make-box("B", size: (2.4, 1.8), pos: (5, 0))
#node-anchor(box, side: "left")
#node-anchor(box, side: "top")
```

Rendered output: #node-anchor(make-box("B", size: (2.4, 1.8), pos: (5, 0)), side: "left"), #h(1em) #node-anchor(make-box("B", size: (2.4, 1.8), pos: (5, 0)), side: "top")

=== `auto-pos-right(after, width, gap: 1.0, y: none, outer-after: true)`
Computes a position to place a node to the right of another one.

```typ
#let left = make-box("Left", size: (2.2, 1.6), pos: (2, 0))
#auto-pos-right(left, 3.0, gap: 1.25)
#auto-pos-right(left, 3.0, gap: 1.25, y: -1.0)
```

Rendered output: #auto-pos-right(make-box("Left", size: (2.2, 1.6), pos: (2, 0)), 3.0, gap: 1.25), #h(1em) #auto-pos-right(make-box("Left", size: (2.2, 1.6), pos: (2, 0)), 3.0, gap: 1.25, y: -1.0)

=== `side-dir(side)`
Direction unit vector for side names.

```typ
#side-dir("left")
#side-dir("right")
#side-dir("top")
#side-dir("bottom")
```

Rendered output: #side-dir("left"), #h(0.8em) #side-dir("right"), #h(0.8em) #side-dir("top"), #h(0.8em) #side-dir("bottom")

== Node Constructors

=== `make-dataset(...)`
Creates a stacked-image node.

Important options:
- `title`, `subtitle`, `legend`, `legend-position`
- `images`, `image-size`, `image-spacing`
- `title-position`: `inside` or `below`
- `title-truncate`, `max-title-chars`
- position: `pos` or `after` + `gap` + optional `y`

```typ
#let ds1 = make-dataset(
  "Image\nDataset",
  images: 3,
  image-size: (1.9, 2.3),
  image-spacing: 0.22,
  pos: (1.0, 0.0),
)

#let ds2 = make-dataset(
  "Features",
  subtitle: "H × W × #Features",
  legend: "latent tensor",
  legend-position: "top",
  images: 6,
  image-size: (1.15, 1.6),
  image-spacing: 0.11,
  title-position: "below",
  after: ds1,
  gap: 1.2,
)
```

Rendered output:

#canvas(length: 0.72cm, {
  import draw: *
  let ds1 = make-dataset(
    "Image\nDataset",
    images: 3,
    image-size: (1.9, 2.3),
    image-spacing: 0.22,
    pos: (1.0, 0.0),
  )
  let ds2 = make-dataset(
    "Features",
    subtitle: "H × W × #Features",
    legend: "latent tensor",
    legend-position: "top",
    images: 6,
    image-size: (1.15, 1.6),
    image-spacing: 0.11,
    title-position: "below",
    after: ds1,
    gap: 1.2,
  )
  draw-node(ds1)
  draw-node(ds2)
})

=== `make-trapezoid(...)`
Creates encoder/decoder trapezoids.

Important options:
- `mode`: `encoder` or `decoder`
- `width`, `big-half`, `small-half`
- labels: `title`, `subtitle`, `legend`
- position: `pos` or `after` + `gap`

```typ
#let enc = make-trapezoid(
  "FeatureEncoder",
  subtitle: "Encoder",
  mode: "encoder",
  width: 2.8,
  big-half: 1.65,
  small-half: 0.80,
  pos: (5.0, 0.0),
)

#let dec = make-trapezoid(
  "Decoder",
  subtitle: "Inverse",
  mode: "decoder",
  width: 2.8,
  big-half: 1.65,
  small-half: 0.80,
  after: enc,
  gap: 1.2,
)
```

Rendered output:

#canvas(length: 0.72cm, {
  import draw: *
  let enc = make-trapezoid(
    "FeatureEncoder",
    subtitle: "Encoder",
    mode: "encoder",
    width: 2.8,
    big-half: 1.65,
    small-half: 0.80,
    pos: (5.0, 0.0),
  )
  let dec = make-trapezoid(
    "FeatureDecoder",
    subtitle: "Inverse",
    mode: "decoder",
    width: 2.8,
    big-half: 1.65,
    small-half: 0.80,
    after: enc,
    gap: 1.2,
  )
  draw-node(enc)
  draw-node(dec)
})

=== `make-box(...)`
Creates rectangular blocks with optional auto size.

Important options:
- `size`: manual `(w, h)` or `none` for auto
- auto size controls: `pad-x`, `min-w`, `min-h`
- labels: `title`, `subtitle`, `legend`
- position: `pos` or `after` + `gap` + optional `y`

```typ
#let b1 = make-box("DepthRegressor", subtitle: "Depth Map\nEstimator", size: none, pos: (10.0, 0.0))
#let b2 = make-box("Comparison Block", subtitle: "Loss / Evaluation", after: b1, gap: 1.4, y: -1.6)
```

Rendered output:

#canvas(length: 0.72cm, {
  import draw: *
  let b1 = make-box("DepthRegressor", subtitle: "Depth Map\nEstimator", size: none, pos: (10.0, 0.0))
  let b2 = make-box("Comparison Block", subtitle: "Loss / Evaluation", after: b1, gap: 1.4, y: -1.6)
  draw-node(b1)
  draw-node(b2)
})

== Drawing Helpers

=== `draw-node(node)`
Draws a node created by `make-dataset`, `make-trapezoid`, or `make-box`.

```typ
#canvas(length: 0.72cm, {
  import draw: *
  let ds = make-dataset("Image\nDataset", pos: (1.0, 0.0))
  let enc = make-trapezoid("FeatureEncoder", subtitle: "Encoder", after: ds, gap: 1.2)
  let box = make-box("DepthRegressor", subtitle: "Estimator", after: enc, gap: 1.2)
  draw-node(ds)
  draw-node(enc)
  draw-node(box)
})
```

Rendered output:

#canvas(length: 0.72cm, {
  import draw: *
  let ds = make-dataset("Image\nDataset", pos: (1.0, 0.0))
  let enc = make-trapezoid("FeatureEncoder", subtitle: "Encoder", after: ds, gap: 1.2)
  let box = make-box("DepthRegressor", subtitle: "Estimator", after: enc, gap: 1.2)
  draw-node(ds)
  draw-node(enc)
  draw-node(box)
})

=== `make-arrow(...)`
Creates orthogonal routes between nodes. Supports multi-segment routing.

Important options:
- side control: `out-side`, `in-side`
- outer anchors: `from-outer`, `to-outer`, `from-outer-value`, `to-outer-value`
- spacing: `auto-spacing`, `spacing`
- route mode: `mode` with `h` and `v` pattern (e.g., `hv`, `vh`, `hvhvh`)
- segment controls: `mode-shift` (legacy), `mode-shifts` (per-segment)
- labels: `label`, `label-side`, `label-gap`, `label-dx`, `label-dy`

```typ
#let a = make-arrow(node-a, node-b, mode: "hv", label: [features])
#let b = make-arrow(node-a, node-b, mode: "hvhvh", mode-shifts: (2.5, none, 0.8, none, 1.2))
#let c = make-arrow(node-a, node-b, from-outer: true, in-side: "bottom", mode: "vh")
```

Rendered output:

#canvas(length: 0.72cm, {
  import draw: *
  let node-a = make-box("A", pos: (1.5, 0.0))
  let node-b = make-box("B", pos: (10.0, -1.5))
  draw-node(node-a)
  draw-node(node-b)
  draw-arrow(make-arrow(node-a, node-b, mode: "hv", label: [features]))
  draw-arrow(make-arrow(node-a, node-b, mode: "hvhvh", mode-shifts: (2.5, none, 0.8, none, 1.2), label: [multi]))
  draw-arrow(make-arrow(node-a, node-b, from-outer: true, in-side: "bottom", mode: "vh", label: [outer]))
})

=== `draw-arrow(arr)`
Draws an arrow route created by `make-arrow`.

```typ
#canvas(length: 0.72cm, {
  import draw: *
  let left = make-box("A", pos: (2, 0))
  let right = make-box("B", pos: (9, -1))
  draw-node(left)
  draw-node(right)
  let arr = make-arrow(left, right, mode: "hvh", label: [decode])
  draw-arrow(arr)
})
```

Rendered output:

#canvas(length: 0.72cm, {
  import draw: *
  let left = make-box("A", pos: (2, 0))
  let right = make-box("B", pos: (9, -1))
  draw-node(left)
  draw-node(right)
  let arr = make-arrow(left, right, mode: "hvh", label: [decode])
  draw-arrow(arr)
})

=== `edge-label(a, b, txt, ...)`
Draws a label along an explicit segment.

```typ
#canvas(length: 0.72cm, {
  import draw: *
  line((1,0), (6,0), mark: (end: ">"), stroke: (paint: black, thickness: 0.75pt))
  edge-label((1,0), (6,0), [input], side: "above", gap: 0.30)
})
```

Rendered output:

#canvas(length: 0.72cm, {
  import draw: *
  line((1,0), (6,0), mark: (end: ">"), stroke: (paint: black, thickness: 0.75pt))
  edge-label((1,0), (6,0), [input], side: "above", gap: 0.30)
})

== Security Emoji Utility

=== `draw-node-emoji(...)`
Draws lock/key/custom emoji around a node with optional status text.

Options:
- `kind`: `lock-closed`, `lock-open`, `key`, `custom`
- `emoji`: custom emoji content (overrides `kind`)
- `place`: `top`, `bottom`, `left`, `right`, `over`, `under`
- `gap`, `shift`, `size`, `color`
- `use-outer`: use full stacked footprint for placement
- `key-state`: `encrypted` or `decrypted`
- `state-text`, `state-size`, `state-gap`, `state-position`

```typ
#canvas(length: 0.72cm, {
  import draw: *
  let enc = make-trapezoid("FeatureEncoder", subtitle: "Encoder", pos: (5, 0))
  draw-node(enc)

  // open lock above
  draw-node-emoji(enc, kind: "lock-open", place: "top", shift: (0.2, 0.0))

  // key on right with state
  draw-node-emoji(enc, kind: "key", place: "right", key-state: "decrypted")

  // custom emoji below
  draw-node-emoji(enc, kind: "custom", emoji: [🛡️], place: "under")
})
```

Rendered output:

#canvas(length: 0.72cm, {
  import draw: *
  let enc = make-trapezoid("FeatureEncoder", subtitle: "Encoder", pos: (5, 0))
  draw-node(enc)

  draw-node-emoji(enc, kind: "lock-open", place: "top", shift: (0.2, 0.0))
  draw-node-emoji(enc, kind: "key", place: "right", key-state: "decrypted")
  draw-node-emoji(enc, kind: "custom", emoji: [🛡️], place: "under")
})

== Default Configurators

=== `set-arrow-defaults(options: ())`
Returns a preconfigured `make-arrow` function.

```typ
#let make-arrow = set-arrow-defaults(options: (spacing: 0.5, label-gap: 0.35))
```

Rendered output:

#canvas(length: 0.72cm, {
  import draw: *
  let make-arrow = set-arrow-defaults(options: (spacing: 0.5, label-gap: 0.35))
  let a = make-box("A", pos: (2.0, 0.0))
  let b = make-box("B", pos: (7.0, 0.0))
  draw-node(a)
  draw-node(b)
  draw-arrow(make-arrow(a, b, label: [default route]))
})

=== `set-dataset-defaults(options: ())`
Returns a preconfigured `make-dataset` function.

```typ
#let make-dataset = set-dataset-defaults(options: (images: 4, image-spacing: 0.18))
```

Rendered output:

#canvas(length: 0.72cm, {
  import draw: *
  let make-dataset = set-dataset-defaults(options: (images: 4, image-spacing: 0.18))
  let ds = make-dataset("Dataset", pos: (3.0, 0.0))
  draw-node(ds)
})

=== `set-trapezoid-defaults(options: ())`
Returns a preconfigured `make-trapezoid` function.

```typ
#let make-trapezoid = set-trapezoid-defaults(options: (width: 3.0, big-half: 1.7, small-half: 0.9))
```

Rendered output:

#canvas(length: 0.72cm, {
  import draw: *
  let make-trapezoid = set-trapezoid-defaults(options: (width: 3.0, big-half: 1.7, small-half: 0.9))
  let t = make-trapezoid("FeatureEncoder", subtitle: "Configured", pos: (4.0, 0.0))
  draw-node(t)
})

=== `set-box-defaults(options: ())`
Returns a preconfigured `make-box` function.

```typ
#let make-box = set-box-defaults(options: (min-w: 2.8, min-h: 1.9, pad-x: 0.75))
```

Rendered output:

#canvas(length: 0.72cm, {
  import draw: *
  let make-box = set-box-defaults(options: (min-w: 2.8, min-h: 1.9, pad-x: 0.75))
  let b = make-box("Configured Box", subtitle: "Default sizing", pos: (4.0, 0.0))
  draw-node(b)
})

== Complete Integrated Example

```typ
#import "@preview/cetz:0.4.2": canvas, draw
#import "graph_utils.typ": *

#let node-gap = 1.2
#let make-arrow = set-arrow-defaults(options: (spacing: 0.45, label-gap: 0.30))
#let make-trapezoid = set-trapezoid-defaults(options: (width: 2.8, big-half: 1.65, small-half: 0.80))

#canvas(length: 0.72cm, {
  import draw: *

  let image-ds = make-dataset("Image\nDataset", images: 3, pos: (1.0, 0.0), color: rgb("#a8c8e8"))
  let depth-ds = make-dataset("Depth Map\nDataset", images: 3, pos: (1.0, -3.8), color: rgb("#c5a0f0"))
  let encoder = make-trapezoid("FeatureEncoder", subtitle: "Encoder", after: image-ds, gap: node-gap, color: rgb("#b0dba0"))
  let latent = make-dataset("H × W × #Features", images: 6, image-size: (1.15, 1.6), image-spacing: 0.11, title-position: "below", after: encoder, gap: node-gap, color: rgb("#fcd97a"), title-size: 0.50em)
  let detect = make-box("DepthRegressor", subtitle: "Depth Map\nEstimator", after: latent, gap: node-gap, size: (3.6, 2.2), color: rgb("#a0b8f5"))
  let compds = make-dataset("Computed\nDepth Maps", images: 3, after: detect, gap: node-gap, color: rgb("#a0e8be"))
  let cmp = make-box("Comparison Block", subtitle: "Loss / Evaluation", after: compds, gap: node-gap, y: -1.85, color: rgb("#f5a0be"))

  draw-node(image-ds)
  draw-node(depth-ds)
  draw-node(encoder)
  draw-node(latent)
  draw-node(detect)
  draw-node(compds)
  draw-node(cmp)
  draw-node-emoji(encoder, kind: "lock-open", place: "top", shift: (0.2, 0.0))

  draw-arrow(make-arrow(image-ds, encoder, from-outer: true, auto-spacing: true, label: [input]))
  draw-arrow(make-arrow(encoder, latent, auto-spacing: false, label: [features]))
  draw-arrow(make-arrow(latent, detect, from-outer: true, auto-spacing: false, label: [decode]))
  draw-arrow(make-arrow(detect, compds, auto-spacing: false))
  draw-arrow(make-arrow(compds, cmp, out-side: "bottom", in-side: "left", auto-spacing: true, mode: "vh", label: [compare]))
  draw-arrow(make-arrow(depth-ds, cmp, out-side: "right", in-side: "bottom", from-outer: true, auto-spacing: true, mode: "hv", label: [ground truth]))
})
```

Rendered output:

#canvas(length: 0.72cm, {
  import draw: *
  let node-gap = 1.2

  let image-ds = make-dataset("Image\nDataset", images: 3, pos: (1.0, 0.0), color: rgb("#a8c8e8"))
  let depth-ds = make-dataset("Depth Map\nDataset", images: 3, pos: (1.0, -3.8), color: rgb("#c5a0f0"))
  let encoder = make-trapezoid("FeatureEncoder", subtitle: "Encoder", after: image-ds, gap: node-gap, color: rgb("#b0dba0"))
  let latent = make-dataset("H × W × #Features", images: 6, image-size: (1.15, 1.6), image-spacing: 0.11, title-position: "below", after: encoder, gap: node-gap, color: rgb("#fcd97a"), title-size: 0.50em)
  let detect = make-box("DepthRegressor", subtitle: "Depth Map\nEstimator", after: latent, gap: node-gap, size: (3.6, 2.2), color: rgb("#a0b8f5"))
  let compds = make-dataset("Computed\nDepth Maps", images: 3, after: detect, gap: node-gap, color: rgb("#a0e8be"))
  let cmp = make-box("Comparison Block", subtitle: "Loss / Evaluation", after: compds, gap: node-gap, y: -1.85, color: rgb("#f5a0be"))

  draw-node(image-ds)
  draw-node(depth-ds)
  draw-node(encoder)
  draw-node(latent)
  draw-node(detect)
  draw-node(compds)
  draw-node(cmp)
  draw-node-emoji(encoder, kind: "lock-open", place: "top", shift: (0.2, 0.0))

  draw-arrow(make-arrow(image-ds, encoder, from-outer: true, auto-spacing: true, label: [input]))
  draw-arrow(make-arrow(encoder, latent, auto-spacing: false, label: [features]))
  draw-arrow(make-arrow(latent, detect, from-outer: true, auto-spacing: false, label: [decode]))
  draw-arrow(make-arrow(detect, compds, auto-spacing: false))
  draw-arrow(make-arrow(compds, cmp, out-side: "bottom", in-side: "left", auto-spacing: true, mode: "vh", label: [compare]))
  draw-arrow(make-arrow(depth-ds, cmp, out-side: "right", in-side: "bottom", from-outer: true, auto-spacing: true, mode: "hv", label: [ground truth]))
})

== Publishing Notes

1. Keep this file next to `graph_utils.typ`.
2. For package publication, include both files and add this document as your main README/guide.
3. The examples are designed to be copied directly into Typst documents.
