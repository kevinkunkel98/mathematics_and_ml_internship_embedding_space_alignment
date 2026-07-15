#import "@preview/touying:0.6.1": *
#import themes.metropolis: *
#import "@preview/cetz:0.3.4": canvas, draw

#import "helpers.typ": *

// ── Slide setup ───────────────────────────────────────────────────────────────
#show: metropolis-theme.with(
  footer-right: context {
    if state("show-slide-number", true).get() { utils.slide-counter.display() }
  },
  aspect-ratio: "16-9",
  config-colors(
    primary: navy,
    primary-light: rgb("#1a4f8a"),
    secondary: rgb("#1c3a5e"),
    neutral-lightest: white,
    neutral-light: rgb("#edf2f8"),
  ),
  config-page(margin: (x: 2.8em, y: 2.2em)),
  config-methods(new-section-slide-fn: none, title-slide-fn: none),
  config-info(
    title: [Representational Geometry in Neural Networks],
    subtitle: [From Vision Transformers to RLHF-Aligned Language Models],
    author: [Huxhold · Pollinger · Kunigk · Kunkel · Charki],
    date: [Summer Semester 2026],
    institution: [Universität Leipzig — Mathematics & Machine Learning Internship \ Supervisor: Dr. Diaaeldin Taha],
  ),
  footer: [Leipzig, 09.07.2026],
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

// Custom title slide — dark theme
#let accent = rgb("#8ab4d4")
#slide(
  config: config-methods(header: _ => none, footer: _ => none) + config-page(fill: navy),
  composer: (1fr, 1.1fr),
  align: horizon,
)[
  #set align(left)
  #v(1fr)
  #text(size: 28pt, weight: "bold", fill: white)[Representational Geometry \ in Neural Networks]
  #v(0.35em)
  #text(size: 13pt, fill: accent)[From Vision Transformers to RLHF-Aligned Language Models]
  #v(0.9em)
  #line(length: 80%, stroke: 1pt + accent)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.25em,
    text(size: 10.5pt, fill: white)[Marla Huxhold], text(size: 10.5pt, fill: white)[Sarah Pollinger],
    text(size: 10.5pt, fill: white)[Ellen Kunigk], text(size: 10.5pt, fill: white)[Kevin Kunkel],
    text(size: 10.5pt, fill: white)[Abdellah Charki], [],
  )
  #v(0.35em)
  #text(size: 10pt, fill: accent)[Summer Semester 2026]
  #v(0.15em)
  #text(size: 10pt, fill: accent)[
    Universität Leipzig — Mathematics & Machine Learning Internship \
    Supervisor: Dr. Diaaeldin Taha
  ]
  #v(1fr)
][
  #align(center + horizon)[
    #block(stroke: 1pt + accent, radius: 4pt, clip: true)[
      #image("assets/slides/dashboard_vit_umap.png")
    ]
    #v(0.3em)
    #text(size: 8pt, fill: accent)[ViT-B/16 · layer 12 · CIFAR-10 · real data]
  ]
]

#include "sections/01_intro.typ"

#include "sections/02_pipeline.typ"

#include "sections/03_model_stages.typ"

#include "sections/04_metrics.typ"

#include "sections/05_part2_results.typ"

#include "sections/05b_context_and_limits.typ"

#include "sections/06_plot_gallery.typ"

#include "sections/07_findings_story.typ"

#include "sections/08_part1_outlook.typ"

#include "sections/09_conclusion.typ"

#include "sections/10_references.typ"

#include "sections/10b_references_part2.typ"
