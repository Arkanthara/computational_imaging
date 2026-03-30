#import "@preview/cetz:0.4.2": canvas, draw

#let clip-str(txt, n) = {
  let n = int(calc.floor(n))
  let chars = txt.clusters()
  if n <= 0 {
    ""
  } else if chars.len() <= n {
    txt
  } else {
    chars.slice(0, n).join("")
  }
}

#let truncate-title(txt, enabled: true, max-chars: 18) = {
  let chars = txt.clusters()
  if not enabled or chars.len() <= max-chars {
    txt
  } else if max-chars <= 3 {
    clip-str(txt, max-chars)
  } else {
    clip-str(txt, max-chars - 3) + "..."
  }
}

#let fit-lines(txt, max-chars: 16, max-lines: 2) = {
  if txt == none {
    none
  } else {
    // Manual control only: no automatic cuts, no auto line breaks.
    // If caller writes "\n", Typst renders multi-line text as provided.
    txt
  }
}

#let chars-cap(width, min: 8, scale: 4.2) = {
  calc.max(min, calc.floor(width * scale))
}

#let node-size(node, outer: false) = {
  if node.kind == "stack" {
    let add = if outer { (node.n - 1) * node.shift } else { 0 }
    (node.w + add, node.h + add)
  } else if node.kind == "trapezoid" {
    (node.w, 2 * calc.max(node.h-left, node.h-right))
  } else {
    (node.w, node.h)
  }
}

#let node-edge(node, side: "right", outer: false, outer-value: none) = {
  if node.kind == "stack" {
    let auto-add = (node.n - 1) * node.shift
    let ov = if outer-value == none { auto-add } else { outer-value }
    if side == "left" {
      node.cx - node.w / 2 - if outer { ov } else { 0 }
    } else if side == "right" {
      node.cx + node.w / 2 + if outer { ov } else { 0 }
    } else if side == "top" {
      node.cy + node.h / 2 + if outer { ov } else { 0 }
    } else {
      node.cy - node.h / 2 - if outer { ov } else { 0 }
    }
  } else {
    let (w, h) = node-size(node, outer: false)
    let ov = if outer and outer-value != none { outer-value } else { 0 }
    if side == "left" {
      node.cx - w / 2 - ov
    } else if side == "right" {
      node.cx + w / 2 + ov
    } else if side == "top" {
      node.cy + h / 2 + ov
    } else {
      node.cy - h / 2 - ov
    }
  }
}

#let auto-pos-right(after, width, gap: 1.0, y: none, outer-after: true) = {
  let cx = node-edge(after, side: "right", outer: outer-after) + gap + width / 2
  let cy = if y == none { after.cy } else { y }
  (cx, cy)
}

#let make-dataset(
  title,
  subtitle: none,
  legend: none,
  legend-position: "below",
  images: 3,
  image-size: (1.9, 2.3),
  image-spacing: 0.22,
  title-truncate: false,
  max-title-chars: 18,
  title-position: "inside",
  pos: none,
  after: none,
  gap: 1.0,
  y: none,
  color: rgb("#a8c8e8"),
  title-size: 0.58em,
  subtitle-size: 0.50em,
  legend-size: 0.48em,
  wrap-lines: 2,
) = {
  let w = image-size.at(0)
  let h = image-size.at(1)
  let (cx, cy) = if pos != none {
    pos
  } else if after != none {
    auto-pos-right(after, w, gap: gap, y: y)
  } else {
    (1.0, if y == none { 0.0 } else { y })
  }

  (
    kind: "stack",
    cx: cx,
    cy: cy,
    w: w,
    h: h,
    n: images,
    shift: image-spacing,
    color: color,
    title: truncate-title(title, enabled: title-truncate, max-chars: max-title-chars),
    subtitle: subtitle,
    legend: legend,
    legend-position: legend-position,
    title-size: title-size,
    subtitle-size: subtitle-size,
    legend-size: legend-size,
    wrap-lines: wrap-lines,
    title-position: title-position,
  )
}

#let make-trapezoid(
  title,
  subtitle: none,
  legend: none,
  legend-position: "below",
  mode: "encoder", // encoder: big->small, decoder: small->big
  width: 2.8,
  big-half: 1.65,
  small-half: 0.80,
  pos: none,
  after: none,
  gap: 1.0,
  y: none,
  color: rgb("#b0dba0"),
  title-size: 0.66em,
  subtitle-size: 0.54em,
  legend-size: 0.48em,
  wrap-lines: 2,
) = {
  let (cx, cy) = if pos != none {
    pos
  } else if after != none {
    auto-pos-right(after, width, gap: gap, y: y)
  } else {
    (4.5, if y == none { 0.0 } else { y })
  }

  let (hl, hr) = if mode == "decoder" {
    (small-half, big-half)
  } else {
    (big-half, small-half)
  }

  (
    kind: "trapezoid",
    cx: cx,
    cy: cy,
    w: width,
    h-left: hl,
    h-right: hr,
    color: color,
    title: title,
    subtitle: subtitle,
    legend: legend,
    legend-position: legend-position,
    title-size: title-size,
    subtitle-size: subtitle-size,
    legend-size: legend-size,
    wrap-lines: wrap-lines,
  )
}

#let make-box(
  title,
  subtitle: none,
  legend: none,
  legend-position: "below",
  size: none,
  pos: none,
  after: none,
  gap: 1.0,
  y: none,
  color: rgb("#a0b8f5"),
  title-size: 0.62em,
  subtitle-size: 0.52em,
  legend-size: 0.48em,
  wrap-lines: 2,
  pad-x: 0.65,
  min-w: 2.3,
  min-h: 1.7,
) = {
  let l1 = title.clusters().len()
  let l2 = if subtitle == none { 0 } else { subtitle.clusters().len() }
  let longest = calc.max(l1, l2)
  let auto-w = calc.max(min-w, longest * 0.17 + 2 * pad-x)
  let auto-h = if subtitle == none { min-h } else { min-h + 0.65 }
  let (w, h) = if size == none { (auto-w, auto-h) } else { size }

  let (cx, cy) = if pos != none {
    pos
  } else if after != none {
    auto-pos-right(after, w, gap: gap, y: y)
  } else {
    (8.0, if y == none { 0.0 } else { y })
  }

  (
    kind: "box",
    cx: cx,
    cy: cy,
    w: w,
    h: h,
    color: color,
    title: title,
    subtitle: subtitle,
    legend: legend,
    legend-position: legend-position,
    title-size: title-size,
    subtitle-size: subtitle-size,
    legend-size: legend-size,
    wrap-lines: wrap-lines,
  )
}

#let draw-node(node) = {
  import draw: *

  let (nw, nh) = node-size(node)
  let max-chars = chars-cap(nw)
  let t = fit-lines(node.title, max-chars: max-chars, max-lines: node.wrap-lines)
  let s = fit-lines(node.subtitle, max-chars: max-chars, max-lines: node.wrap-lines)
  let g = fit-lines(node.legend, max-chars: chars-cap(nw + 0.6), max-lines: node.wrap-lines)

  let draw-inner-text(x, y) = {
    if node.subtitle == none {
      content((x, y), align(center)[#text(size: node.title-size)[#t]])
    } else {
      content((x, y + 0.26), align(center)[#text(size: node.title-size)[#t]])
      content((x, y - 0.36), align(center)[#text(size: node.subtitle-size, fill: rgb("#333333"))[#s]])
    }
  }

  let draw-legend(x, y) = {
    if node.legend != none {
      let y-leg = if node.legend-position == "top" {
        y + nh / 2 + 0.55
      } else {
        y - nh / 2 - 0.55
      }
      content((x, y-leg), align(center)[#text(size: node.legend-size, fill: rgb("#555555"))[#g]])
    }
  }

  if node.kind == "stack" {
    for i in range(node.n) {
      let j = node.n - 1 - i
      let off = j * node.shift
      rect(
        (node.cx - node.w / 2 + off, node.cy - node.h / 2 + off),
        (node.cx + node.w / 2 + off, node.cy + node.h / 2 + off),
        fill: node.color,
        stroke: (paint: black, thickness: 0.5pt),
      )
    }

    if node.title-position == "below" {
      content((node.cx, node.cy - node.h / 2 - 0.55), align(center)[#text(size: node.title-size)[#t]])
      if node.subtitle != none {
        content((node.cx, node.cy - node.h / 2 - 0.95), align(center)[#text(size: node.subtitle-size, fill: rgb("#333333"))[#s]])
      }
    } else {
      draw-inner-text(node.cx, node.cy)
    }
    draw-legend(node.cx, node.cy)
  } else if node.kind == "trapezoid" {
    let lx = node.cx - node.w / 2
    let rx = node.cx + node.w / 2
    line(
      (lx, node.cy - node.h-left), (rx, node.cy - node.h-right),
      (rx, node.cy + node.h-right), (lx, node.cy + node.h-left),
      close: true,
      fill: node.color,
      stroke: (paint: black, thickness: 0.65pt),
    )
    if node.subtitle == none {
      content((node.cx, node.cy), align(center)[#text(size: node.title-size)[*#t*]])
    } else {
      content((node.cx, node.cy + 0.26), align(center)[#text(size: node.title-size)[*#t*]])
      content((node.cx, node.cy - 0.36), align(center)[#text(size: node.subtitle-size, fill: rgb("#333333"))[#s]])
    }
    draw-legend(node.cx, node.cy)
  } else {
    rect(
      (node.cx - node.w / 2, node.cy - node.h / 2),
      (node.cx + node.w / 2, node.cy + node.h / 2),
      fill: node.color,
      stroke: (paint: black, thickness: 0.65pt),
    )
    if node.subtitle == none {
      content((node.cx, node.cy), align(center)[#text(size: node.title-size)[*#t*]])
    } else {
      content((node.cx, node.cy + 0.26), align(center)[#text(size: node.title-size)[*#t*]])
      content((node.cx, node.cy - 0.36), align(center)[#text(size: node.subtitle-size)[#s]])
    }
    draw-legend(node.cx, node.cy)
  }
}

#let node-anchor(node, side: "right", outer: false, outer-value: none) = {
  if side == "left" {
    (node-edge(node, side: "left", outer: outer, outer-value: outer-value), node.cy)
  } else if side == "right" {
    (node-edge(node, side: "right", outer: outer, outer-value: outer-value), node.cy)
  } else if side == "top" {
    (node.cx, node-edge(node, side: "top", outer: outer, outer-value: outer-value))
  } else if side == "bottom" {
    (node.cx, node-edge(node, side: "bottom", outer: outer, outer-value: outer-value))
  } else {
    (node.cx, node.cy)
  }
}

#let side-dir(side) = {
  if side == "left" {
    (-1.0, 0.0)
  } else if side == "right" {
    (1.0, 0.0)
  } else if side == "top" {
    (0.0, 1.0)
  } else {
    (0.0, -1.0)
  }
}

#let make-arrow(
  from,
  to,
  out-side: "right",
  in-side: "left",
  from-outer: false,
  from-outer-value: none,
  to-outer: false,
  to-outer-value: none,
  auto-spacing: true,
  spacing: 0.45,
  mode: "hv", // any h/v string, ex: hv, vh, hvhvh
  mode-shift: none, // legacy single override: (segment-index, length)
  mode-shifts: none, // per-segment overrides array, ex: (none, 2.5, none, 1.0)
  label: none,
  label-side: "above",
  label-gap: 0.30,
  label-dx: 0.0,
  label-dy: 0.0,
) = {
  let same(a, b) = calc.abs(a.at(0) - b.at(0)) < 0.001 and calc.abs(a.at(1) - b.at(1)) < 0.001
  let p0 = node-anchor(from, side: out-side, outer: from-outer, outer-value: from-outer-value)
  let q0 = node-anchor(to, side: in-side, outer: to-outer, outer-value: to-outer-value)

  let dv-out = side-dir(out-side)
  let dv-in = side-dir(in-side)

  let p1 = if auto-spacing {
    (p0.at(0) + dv-out.at(0) * spacing, p0.at(1) + dv-out.at(1) * spacing)
  } else {
    p0
  }

  let q1 = if auto-spacing {
    (q0.at(0) + dv-in.at(0) * spacing, q0.at(1) + dv-in.at(1) * spacing)
  } else {
    q0
  }

  // Build and validate mode segments.
  let raw-segs = mode.clusters().filter(s => s == "h" or s == "v")
  let segs = if raw-segs.len() == 0 {
    ("h", "v")
  } else {
    raw-segs
  }
  let n = segs.len()

  let dx = q1.at(0) - p1.at(0)
  let dy = q1.at(1) - p1.at(1)
  let sx = if dx >= 0 { 1 } else { -1 }
  let sy = if dy >= 0 { 1 } else { -1 }
  let adx = calc.abs(dx)
  let ady = calc.abs(dy)

  // Segment-level override lookup (1-based index).
  let seg-override(i) = {
    if mode-shifts != none and i <= mode-shifts.len() {
      mode-shifts.at(i - 1, default: none)
    } else if mode-shift != none and mode-shift.len() >= 2 and mode-shift.at(0) == i {
      mode-shift.at(1)
    } else {
      none
    }
  }

  // Gather per-axis accounting.
  let h-known = 0.0
  let v-known = 0.0
  let h-unk = 0
  let v-unk = 0
  let h-last = 0
  let v-last = 0
  for i in range(1, n + 1) {
    let axis = segs.at(i - 1)
    let ov = seg-override(i)
    if axis == "h" {
      h-last = i
      if ov == none { h-unk += 1 } else { h-known += ov }
    } else {
      v-last = i
      if ov == none { v-unk += 1 } else { v-known += ov }
    }
  }

  let h-rem = adx - h-known
  let v-rem = ady - v-known
  let h-each = if h-unk > 0 { h-rem / h-unk } else { 0 }
  let v-each = if v-unk > 0 { v-rem / v-unk } else { 0 }
  let h-adjust = if h-unk == 0 { h-rem } else { 0 }
  let v-adjust = if v-unk == 0 { v-rem } else { 0 }

  // Compute all segment lengths.
  let seg-lens = ()
  for i in range(1, n + 1) {
    let axis = segs.at(i - 1)
    let ov = seg-override(i)
    let l = if ov != none {
      ov
    } else if axis == "h" {
      h-each
    } else {
      v-each
    }

    if axis == "h" and i == h-last {
      l += h-adjust
    }
    if axis == "v" and i == v-last {
      l += v-adjust
    }
    seg-lens.push(l)
  }

  // Build orthogonal core route from p1 to q1 according to mode pattern.
  let core = (p1,)
  let cx = p1.at(0)
  let cy = p1.at(1)
  for i in range(0, n) {
    let axis = segs.at(i)
    let l = seg-lens.at(i)
    if axis == "h" {
      cx += sx * l
    } else {
      cy += sy * l
    }
    core.push((cx, cy))
  }

  // If mode cannot fully absorb both axes, append orthogonal correction legs.
  if calc.abs(cx - q1.at(0)) > 0.001 {
    cx = q1.at(0)
    core.push((cx, cy))
  }
  if calc.abs(cy - q1.at(1)) > 0.001 {
    cy = q1.at(1)
    core.push((cx, cy))
  }
  if not same(core.last(), q1) {
    core.push(q1)
  }

  // Compose full path while removing duplicate consecutive points.
  let points = ()
  points.push(p0)
  for pt in core {
    if not same(points.last(), pt) {
      points.push(pt)
    }
  }
  if not same(points.last(), q0) {
    points.push(q0)
  }

  // Label target segment: longest segment in the routed core path.
  let label-a = p1
  let label-b = q1
  let best = -1.0
  for i in range(0, core.len() - 1) {
    let a = core.at(i)
    let b = core.at(i + 1)
    let d = calc.abs(b.at(0) - a.at(0)) + calc.abs(b.at(1) - a.at(1))
    if d > best {
      best = d
      label-a = a
      label-b = b
    }
  }

  (
    p0: p0,
    p1: p1,
    m: if points.len() > 2 { points.at(1) } else { p1 },
    q1: q1,
    q0: q0,
    has-bend: points.len() > 3,
    points: points,
    label: label,
    label-a: label-a,
    label-b: label-b,
    label-side: label-side,
    label-gap: label-gap,
    label-dx: label-dx,
    label-dy: label-dy,
  )
}

#let edge-label(a, b, txt, side: "above", gap: 0.22, dx: 0.0, dy: 0.0, size: 0.48em) = {
  let mx = (a.at(0) + b.at(0)) / 2 + dx
  let my = (a.at(1) + b.at(1)) / 2 + dy
  let horiz = calc.abs(a.at(1) - b.at(1)) < 0.001
  let ox = if horiz { 0 } else if side == "right" { gap } else { -gap }
  let oy = if horiz { if side == "below" { -gap } else { gap } } else { 0 }
  draw.content((mx + ox, my + oy), text(size: size, fill: rgb("#555555"))[#txt])
}

#let draw-arrow(arr) = {
  import draw: *

  if "points" in arr and arr.points.len() >= 2 {
    line(..arr.points,
      mark: (end: ">"),
      stroke: (paint: black, thickness: 0.75pt),
    )
  } else {
    // Backward compatibility fallback.
    line(arr.p0, arr.q0,
      mark: (end: ">"),
      stroke: (paint: black, thickness: 0.75pt),
    )
  }

  if arr.label != none {
    let a = if "label-a" in arr {
      arr.label-a
    } else {
      arr.p1
    }
    let b = if "label-b" in arr {
      arr.label-b
    } else {
      arr.q1
    }

    edge-label(
      a,
      b,
      arr.label,
      side: arr.label-side,
      gap: arr.label-gap,
      dx: arr.label-dx,
      dy: arr.label-dy,
    )
  }
}
