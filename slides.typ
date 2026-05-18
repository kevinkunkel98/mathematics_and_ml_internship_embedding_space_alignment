#import "@preview/touying:0.6.1": *
#import themes.metropolis: *
#import "@preview/cetz:0.3.4": canvas, draw

// ── Palette ───────────────────────────────────────────────────────────────────
#let navy = rgb("#1c3a5e")
#let blue = rgb("#1a4f8a")
#let sky = rgb("#dce9f7")
#let sage = rgb("#1e6b3c")
#let mint = rgb("#e4f4ec")
#let sand = rgb("#f7f4ef")

// ── Box helpers ───────────────────────────────────────────────────────────────
#let thm-box(title, body, fill: sky, stroke-color: blue) = block(
  width: 100%,
  inset: (x: 0.9em, y: 0.65em),
  radius: 3pt,
  fill: fill,
  stroke: (left: 3pt + stroke-color),
)[
  #text(weight: "bold", fill: stroke-color, size: 0.9em)[#title]
  #h(0.4em)
  #body
]

#let insight(title, body) = thm-box([Insight: #title], body)
#let definition(title, body) = thm-box([Definition: #title], body, fill: mint, stroke-color: sage)
#let remark(body) = thm-box([Remark], body, fill: rgb("#fef9ec"), stroke-color: rgb("#b7770d"))

// ── Slide setup ───────────────────────────────────────────────────────────────
#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  config-colors(
    primary: navy,
    primary-light: rgb("#1a4f8a"),
    secondary: rgb("#1c3a5e"),
    neutral-lightest: white,
    neutral-light: rgb("#edf2f8"),
  ),
  config-page(margin: (x: 2.8em, y: 2.2em)),
  config-info(
    title: [Representational Geometry in Neural Networks],
    subtitle: [From Vision Transformers to RLHF-Aligned Language Models],
    author: [Huxhold · Pollinger · Kunigk · Kunkel · Charki],
    date: [Summer Semester 2026],
    institution: [Universität Leipzig — Mathematics & Machine Learning Internship \ Supervisor: Dr. Diaaeldin Taha],
  ),
)

#set text(size: 19pt)

// ── Embedding space diagram for title slide ───────────────────────────────────
#let embedding-diagram = canvas(length: 1cm, {
  import draw: *

  let col-chosen = rgb("#1e6b3c")
  let col-rejected = rgb("#8b1a1a")
  let col-label = luma(60)

  // ── Base model panel (left) — mixed, overlapping ─────────────────────────
  rect((-5.2, -2.0), (-0.4, 2.2), fill: rgb("#f5f8ff"), stroke: 0.4pt + luma(200), radius: 3pt)
  content((-2.8, 1.85), text(fill: navy, size: 0.28cm, weight: "bold")[Base LLM])

  let chosen-base = (
    (-3.8, 0.8),
    (-2.2, 1.1),
    (-1.5, 0.3),
    (-3.0, -0.2),
    (-4.2, -0.5),
    (-2.6, 0.5),
    (-1.8, -0.8),
    (-3.5, 1.2),
  )
  let rejected-base = (
    (-4.0, 0.3),
    (-2.0, 0.8),
    (-1.2, -0.3),
    (-2.8, -0.9),
    (-3.7, -1.1),
    (-1.6, 0.1),
    (-4.5, 0.9),
    (-2.4, -0.5),
  )
  for p in chosen-base {
    circle(p, radius: 0.13, fill: col-chosen, stroke: none)
  }
  for p in rejected-base {
    circle(p, radius: 0.13, fill: col-rejected, stroke: none)
  }

  // ── Arrow ─────────────────────────────────────────────────────────────────
  line((-0.1, 0.1), (0.5, 0.1), stroke: (paint: navy, thickness: 1.8pt), mark: (end: ">", size: 0.3))
  content((0.2, 0.55), text(fill: navy, size: 0.22cm, style: "italic")[RLHF])

  // ── RLHF model panel (right) — cleanly separated ──────────────────────────
  rect((0.7, -2.0), (5.5, 2.2), fill: rgb("#f5f8ff"), stroke: 0.4pt + luma(200), radius: 3pt)
  content((3.1, 1.85), text(fill: navy, size: 0.28cm, weight: "bold")[RLHF LLM])

  let chosen-rlhf = ((3.5, 0.9), (4.2, 1.2), (4.6, 0.5), (3.9, 0.4), (4.0, 1.0), (4.8, 0.9), (3.6, 0.6))
  let rejected-rlhf = ((1.2, -0.6), (1.8, -1.1), (2.3, -0.8), (1.5, -1.3), (2.0, -0.4), (1.1, -1.0), (2.5, -1.1))
  for p in chosen-rlhf {
    circle(p, radius: 0.13, fill: col-chosen, stroke: none)
  }
  for p in rejected-rlhf {
    circle(p, radius: 0.13, fill: col-rejected, stroke: none)
  }

  // Separating hyperplane
  line((0.9, 0.4), (5.3, -0.5), stroke: (paint: navy, thickness: 1.2pt, dash: "dashed"))

  // ── Legend ─────────────────────────────────────────────────────────────────
  circle((-5.0, -1.65), radius: 0.11, fill: col-chosen, stroke: none)
  content((-4.55, -1.65), text(fill: col-label, size: 0.24cm)[Chosen])
  circle((-3.4, -1.65), radius: 0.11, fill: col-rejected, stroke: none)
  content((-2.95, -1.65), text(fill: col-label, size: 0.24cm)[Rejected])
})

// ── Slides ────────────────────────────────────────────────────────────────────

// Custom title slide
#slide(
  config: config-methods(header: _ => none, footer: _ => none),
  composer: (1fr, 1.1fr),
  align: horizon,
)[
  #set align(left)
  #v(1fr)
  #text(size: 22pt, weight: "bold", fill: navy)[Representational Geometry \ in Neural Networks]
  #v(0.3em)
  #text(size: 13pt, fill: luma(80))[From Vision Transformers to RLHF-Aligned Language Models]
  #v(0.9em)
  #line(length: 80%, stroke: 1pt + navy)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.25em,
    text(size: 10.5pt)[Marla Huxhold], text(size: 10.5pt)[Sarah Pollinger],
    text(size: 10.5pt)[Ellen Kunigk], text(size: 10.5pt)[Kevin Kunkel],
    text(size: 10.5pt)[Abdellah Charki], [],
  )
  #v(0.35em)
  #text(size: 10pt, fill: luma(100))[Summer Semester 2026]
  #v(0.15em)
  #text(size: 10pt, fill: luma(100))[
    Universität Leipzig — Mathematics & Machine Learning Internship \
    Supervisor: Dr. Diaaeldin Taha
  ]
  #v(1fr)
][
  #align(center + horizon)[
    #image("assets/slides/dashboard_vit_umap.png")
    #v(0.3em)
    #text(size: 8pt, fill: luma(140))[ViT-B/16 · layer 12 · CIFAR-10 · real data]
  ]
]

// --------------------------------------------------------------------------
= Goal & Structure

== What We Are Studying

#v(0.3em)
#block(
  fill: sand,
  stroke: (left: 3pt + navy),
  inset: (x: 0.9em, y: 0.7em),
  radius: 3pt,
)[
  *Core question:* How does training geometrically reorganize the internal representation
  space of a neural network — and can we make that transformation literally visible?
]

#v(0.6em)
#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  block(fill: sky, inset: 0.8em, radius: 3pt, stroke: (left: 3pt + blue))[
    *Part 1 — Vision* \
    #v(0.3em)
    ResNet-18 vs. ViT-B/16 on CIFAR-10 \
    *How similar* are their internal layers? \
    *Which regions* drive their predictions? \
    #v(0.3em)
    #text(size: 0.85em, fill: luma(80))[CKA + Class Activation Maps]
  ],
  block(fill: mint, inset: 0.8em, radius: 3pt, stroke: (left: 3pt + sage))[
    *Part 2 — Language* \
    #v(0.3em)
    Llama-3-8B vs. Llama-3-8B-Instruct \
    Does RLHF *linearly separate* preferred \ from rejected responses? \
    #v(0.3em)
    #text(size: 0.85em, fill: luma(80))[LinearSVC + UMAP · `Anthropic/hh-rlhf`]
  ],
)

// --------------------------------------------------------------------------
= Approach

== Methods at a Glance

#v(0.3em)
#table(
  columns: (auto, 1fr, 1fr),
  align: (left, left, left),
  stroke: none,
  fill: (_, row) => if row == 0 { navy } else if calc.odd(row) { rgb("#f0f4f9") } else { white },
  table.hline(stroke: 0.5pt + navy),
  [#text(fill: white, weight: "bold")[Part]],
  [#text(fill: white, weight: "bold")[What we measure]],
  [#text(fill: white, weight: "bold")[How]],
  table.hline(stroke: 0.3pt + luma(200)),
  [Vision],
  [Layer-wise representational similarity between CNN and ViT],
  [*CKA* — score in $[0,1]$, invariant to rotation & scaling],
  [Vision],
  [Which image regions activate each architecture's layers],
  [*GradCAM* — gradient-weighted spatial attention maps],
  [Language],
  [Linear separability of chosen vs. rejected response embeddings],
  [*LinearSVC* per layer — accuracy & margin across 32 layers],
  [Language],
  [Geometric effect of RLHF alignment across the full model],
  [*UMAP* scatter — base vs. instruct, layer slider],
  table.hline(stroke: 0.5pt + navy),
)

// --------------------------------------------------------------------------
= Current State

== What We Have Built

#let status-card(title, header-fill, body-fill, stroke-clr, body) = block(
  width: 100%, radius: 5pt, stroke: 1pt + stroke-clr, clip: true, inset: 0pt,
)[
  #set block(spacing: 0em)
  #block(width: 100%, fill: header-fill, inset: (x: 0.9em, y: 0.55em))[
    #text(fill: white, weight: "bold", size: 0.95em)[#title]
  ]
  #block(width: 100%, fill: body-fill, inset: (x: 0.9em, y: 0.7em))[#body]
]

#v(0.15em)
#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  align: top,
  status-card([Done], sage, mint, sage)[
    - Dash app · CKA heatmap, CAM viewer & layer UMAP
    - Real CIFAR-10 embeddings (ResNet-18 & ViT-B/16)
    - Real LLM embeddings (Llama-3-8B base + Instruct)
    - CKA + saliency + UMAP pipeline · 24 tests
  ],
  status-card([Up next], blue, sky, blue)[
    - Fine-tune ResNet-18 on CIFAR-10
    - Re-train ViT (*49.9%* → target ≥ 85%)
    - Saliency maps & bias probe experiments
    - Correlate CKA profile with UMAP layer separation
  ],
)

#v(0.3em)
#block(
  fill: sky,
  stroke: (left: 3pt + blue),
  inset: (x: 0.9em, y: 0.55em),
  radius: 3pt,
)[
  #text(weight: "bold", fill: blue, size: 0.9em)[Early finding] —
  ViT-B/16 builds class-discriminative structure monotonically across layers (UMAP ratio 0.44 → 3.79).
  ResNet without fine-tuning shows no class clustering — motivates proper training.
]

// --------------------------------------------------------------------------
= Dashboard

== Part 1 — CKA Heatmap & Layer Activation UMAP

#v(0.2em)
#grid(
  columns: (1fr, 1fr),
  gutter: 0.8em,
  image("assets/slides/dashboard_cka.png"),
  [
    #image("assets/slides/dashboard_vit_umap.png")
    #v(0.15em)
    #text(size: 8.5pt, fill: luma(120))[ViT-B/16 · layer 12 · CIFAR-10 test set (real data)]
  ],
)

// --------------------------------------------------------------------------
== Part 2 — RLHF Embedding Space & Separation Score

#v(0.15em)
#grid(
  columns: (1fr, 1fr),
  gutter: 0.8em,
  image("assets/slides/umap_scatter.png"),
  [
    #image("assets/slides/svc_line.png")
    #v(0.3em)
    #block(
      fill: sky,
      stroke: (left: 3pt + blue),
      inset: (x: 0.8em, y: 0.6em),
      radius: 3pt,
    )[
      #text(weight: "bold", fill: blue, size: 0.85em)[Key finding so far] \
      The aligned model shows markedly higher SVC accuracy in deeper layers —
      RLHF *geometrically* encodes human preference as a linear structure.
    ]
  ],
)
