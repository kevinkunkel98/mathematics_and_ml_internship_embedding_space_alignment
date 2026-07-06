#import "../helpers.typ": *
#import "@preview/cetz:0.3.4": canvas, draw

// --------------------------------------------------------------------------
== Part 2 — Pipeline: From Dataset to Dashboard

#let code(body) = box(
  fill: luma(240),
  inset: (x: 0.35em, y: 0.15em),
  radius: 2pt,
  outset: (y: 0.15em),
)[#text(font: "DejaVu Sans Mono", size: 0.85em, fill: rgb("#8b1a1a"))[#body]]

#v(0.6em)
#align(center)[
  #canvas(length: 1.4cm, {
    import draw: *

    let stage(x, y, title, body, w: 4.6, h: 2.0) = {
      rect((x, y - h / 2), (x + w, y + h / 2), fill: rgb("#f5f8ff"), stroke: 1pt + navy, radius: 4pt)
      content((x + w / 2, y + h / 2 - 0.45), text(fill: navy, size: 0.3cm, weight: "bold")[#title])
      content((x + w / 2, y - 0.15), text(fill: luma(60), size: 0.22cm)[#body])
    }

    let arrow(..pts) = line(..pts, stroke: (paint: navy, thickness: 1.8pt), mark: (end: ">", size: 0.32))

    // ── Row 1 ──────────────────────────────────────────────────────────────
    let y1 = 1.7
    stage(-14.6, y1, [Anthropic/hh-rlhf], [4,000 chosen/\ rejected pairs])
    arrow((-10.0, y1), (-9.3, y1))
    stage(-9.3, y1, [Cluster extraction], [SLURM · Tulu-3-8B\ SFT / DPO / RLHF])
    arrow((-4.7, y1), (-4.0, y1))
    stage(-4.0, y1, [layers.h5], [33 layers × 4,000 ×\ 4,096, per checkpoint])

    // ── Row 2 ──────────────────────────────────────────────────────────────
    let y2 = -2.0
    stage(-11.9, y2, [Geometry + CKA], [SVC · Cohen's d ·\ anisotropy · rank · drift])
    arrow((-7.3, y2), (-6.6, y2))
    stage(-6.6, y2, [Dash dashboard], [interactive plots ·\ this deck's PNGs])

    // ── Connector row1 -> row2 (elbow) ───────────────────────────────────────
    arrow((-1.7, y1 - 1.0), (-1.7, -0.2), (-9.65, -0.2), (-9.65, y2 + 1.0))
  })

  #v(0.7em)
  #code[scripts/extract_embeddings.py] #text(fill: navy)[→] #code[app/rlhf_geometry_compute.py] / #code[app/vision_compute.py] #text(fill: navy)[→] #code[app/app.py]
]
