#import "@preview/touying:0.6.1": *
#import themes.metropolis: *
#import "@preview/cetz:0.3.4": canvas, draw

// ── Palette ───────────────────────────────────────────────────────────────────
#let navy  = rgb("#1c3a5e")
#let blue  = rgb("#1a4f8a")
#let sky   = rgb("#dce9f7")
#let sage  = rgb("#1e6b3c")
#let mint  = rgb("#e4f4ec")
#let sand  = rgb("#f7f4ef")

// ── Box helpers ───────────────────────────────────────────────────────────────
#let thm-box(title, body, fill: sky, stroke-color: blue) = block(
  width: 100%, inset: (x: 0.9em, y: 0.65em), radius: 3pt,
  fill: fill, stroke: (left: 3pt + stroke-color),
)[
  #text(weight: "bold", fill: stroke-color, size: 0.9em)[#title]
  #h(0.4em)
  #body
]

#let insight(title, body)    = thm-box([Insight: #title], body)
#let definition(title, body) = thm-box([Definition: #title], body,
  fill: mint, stroke-color: sage)
#let remark(body)            = thm-box([Remark], body,
  fill: rgb("#fef9ec"), stroke-color: rgb("#b7770d"))

// ── Slide setup ───────────────────────────────────────────────────────────────
#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  config-colors(
    primary:          navy,
    primary-light:    rgb("#1a4f8a"),
    secondary:        rgb("#1c3a5e"),
    neutral-lightest: white,
    neutral-light:    rgb("#edf2f8"),
  ),
  config-page(margin: (x: 2.8em, y: 2.2em)),
  config-info(
    title:       [Representational Geometry in Neural Networks],
    subtitle:    [From Vision Transformers to RLHF-Aligned Language Models],
    author:      [Huxhold · Pollinger · Kunigk · Kunkel · Charki],
    date:        [Summer Semester 2026],
    institution: [Universität Leipzig — Mathematics & Machine Learning Internship \ Supervisor: Dr. Diaaeldin Taha],
  ),
)

#set text(size: 19pt)

// ── Embedding space diagram for title slide ───────────────────────────────────
#let embedding-diagram = canvas(length: 1cm, {
  import draw: *

  let col-chosen   = rgb("#1e6b3c")
  let col-rejected = rgb("#8b1a1a")
  let col-label    = luma(60)

  // ── Base model panel (left) — mixed, overlapping ─────────────────────────
  rect((-5.2, -2.0), (-0.4, 2.2),
    fill: rgb("#f5f8ff"), stroke: 0.4pt + luma(200), radius: 3pt)
  content((-2.8, 1.85), text(fill: navy, size: 0.28cm, weight: "bold")[Base LLM])

  let chosen-base = ((-3.8, 0.8), (-2.2, 1.1), (-1.5, 0.3), (-3.0, -0.2),
                     (-4.2, -0.5), (-2.6, 0.5), (-1.8, -0.8), (-3.5, 1.2))
  let rejected-base = ((-4.0, 0.3), (-2.0, 0.8), (-1.2, -0.3), (-2.8, -0.9),
                       (-3.7, -1.1), (-1.6, 0.1), (-4.5, 0.9), (-2.4, -0.5))
  for p in chosen-base {
    circle(p, radius: 0.13, fill: col-chosen, stroke: none)
  }
  for p in rejected-base {
    circle(p, radius: 0.13, fill: col-rejected, stroke: none)
  }

  // ── Arrow ─────────────────────────────────────────────────────────────────
  line((-0.1, 0.1), (0.5, 0.1),
    stroke: (paint: navy, thickness: 1.8pt),
    mark: (end: ">", size: 0.3))
  content((0.2, 0.55), text(fill: navy, size: 0.22cm, style: "italic")[RLHF])

  // ── RLHF model panel (right) — cleanly separated ──────────────────────────
  rect((0.7, -2.0), (5.5, 2.2),
    fill: rgb("#f5f8ff"), stroke: 0.4pt + luma(200), radius: 3pt)
  content((3.1, 1.85), text(fill: navy, size: 0.28cm, weight: "bold")[RLHF LLM])

  let chosen-rlhf = ((3.5, 0.9), (4.2, 1.2), (4.6, 0.5), (3.9, 0.4),
                     (4.0, 1.0), (4.8, 0.9), (3.6, 0.6))
  let rejected-rlhf = ((1.2, -0.6), (1.8, -1.1), (2.3, -0.8), (1.5, -1.3),
                       (2.0, -0.4), (1.1, -1.0), (2.5, -1.1))
  for p in chosen-rlhf {
    circle(p, radius: 0.13, fill: col-chosen, stroke: none)
  }
  for p in rejected-rlhf {
    circle(p, radius: 0.13, fill: col-rejected, stroke: none)
  }

  // Separating hyperplane
  line((0.9, 0.4), (5.3, -0.5),
    stroke: (paint: navy, thickness: 1.2pt, dash: "dashed"))

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
  #grid(columns: (1fr, 1fr), gutter: 0.25em,
    text(size: 10.5pt)[Marla Huxhold],
    text(size: 10.5pt)[Sarah Pollinger],
    text(size: 10.5pt)[Ellen Kunigk],
    text(size: 10.5pt)[Kevin Kunkel],
    text(size: 10.5pt)[Abdellah Charki],
    [],
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
    #embedding-diagram
  ]
]

// --------------------------------------------------------------------------
= Motivation

== Why Study Representational Geometry?

#v(0.4em)
- Neural networks are powerful — but *what do their internal representations look like?*
- Layers encode rich geometric structure: clusters, manifolds, linear subspaces
- Training transforms this geometry in non-trivial, measurable ways

#v(0.6em)
#block(
  fill: sand, stroke: (left: 3pt + navy), inset: (x: 0.9em, y: 0.7em), radius: 3pt,
)[
  *Goal:* Perform a geometric audit of what training does to the internal state of a
  neural network — making the "blackbox" of embeddings literally visible and explorable.
]

// --------------------------------------------------------------------------
= Research Questions

== What We Are Investigating

#v(0.4em)
#table(
  columns: (auto, 1fr, 1fr),
  align: (left, left, left),
  stroke: none,
  fill: (_, row) => if row == 0 { navy } else if calc.odd(row) { rgb("#f0f4f9") } else { white },
  table.hline(stroke: 0.5pt + navy),
  [#text(fill: white, weight: "bold")[Part]],
  [#text(fill: white, weight: "bold")[Question]],
  [#text(fill: white, weight: "bold")[Method]],
  table.hline(stroke: 0.3pt + luma(200)),
  [Vision],
    [How similar are CNN vs. ViT internal representations, and which image regions drive cluster formation?],
    [CKA + Class Activation Maps],
  [Language],
    [Can a linear hyperplane separate preferred from rejected response vectors after RLHF, and how does this evolve across layers?],
    [LinearSVC + Margin Analysis],
  [Language],
    [Can political or cultural biases introduced by RLHF be identified as geometric structures?],
    [Bias Probes + Cluster Analysis],
  table.hline(stroke: 0.5pt + navy),
)

// --------------------------------------------------------------------------
= Part 1 — Vision Models

== CNN vs. Vision Transformer

#v(0.3em)
Two dominant paradigms for image recognition, trained on *CIFAR-10*:

#v(0.4em)
#grid(columns: (1fr, 1fr), gutter: 1.2em,
  block(fill: sand, inset: 0.8em, radius: 3pt, stroke: 0.4pt + luma(200))[
    *CNN* \
    #v(0.3em)
    Inductive bias: local spatial structure \
    Hierarchical feature extraction \
    Fixed receptive field per layer
  ],
  block(fill: sand, inset: 0.8em, radius: 3pt, stroke: 0.4pt + luma(200))[
    *Vision Transformer (ViT)* \
    #v(0.3em)
    Global self-attention from layer 1 \
    Patch-based tokenization \
    No explicit spatial inductive bias
  ],
)

#v(0.5em)
*Question:* Do they encode the same information in their intermediate layers, despite fundamentally different architectures?

// --------------------------------------------------------------------------
== Centered Kernel Alignment (CKA)

#definition([CKA])[
  Given activation matrices $X in RR^(n times p)$ and $Y in RR^(n times q)$, *linear CKA* is:
  $
    "CKA"(X, Y) = frac(norm(Y^top X)_F^2, norm(X^top X)_F dot.op norm(Y^top Y)_F)
  $
  $"CKA" in [0, 1]$, invariant to orthogonal transforms and isotropic scaling.
]

#v(0.3em)
- $"CKA" = 1$: representations are linearly equivalent
- $"CKA" = 0$: representations are unrelated
- Enables *layer-by-layer* comparison across different architectures

// --------------------------------------------------------------------------
== Class Activation Maps

#v(0.3em)
*CAMs* reveal which spatial regions of an image drive a model's prediction:

$
  "CAM"_c (x, y) = sum_k w_k^c dot A_k(x, y)
$

where $A_k$ is the $k$-th feature map and $w_k^c$ are class-specific weights.

#v(0.5em)
#remark[
  Overlaying CAMs on images and comparing CNN vs. ViT lets us ask: do architectures
  with high CKA also attend to the *same spatial regions*?
]

// --------------------------------------------------------------------------
= Part 2 — Language Models

== RLHF and Geometric Alignment

#v(0.3em)
*Reinforcement Learning from Human Feedback* fine-tunes a language model to produce outputs humans prefer.

#v(0.4em)
#grid(columns: (1fr, 1fr), gutter: 1.2em,
  block(fill: sand, inset: 0.8em, radius: 3pt, stroke: 0.4pt + luma(200))[
    *Base: Llama-3-8B* \
    #v(0.3em)
    Pretrained on raw text \
    No explicit preference signal \
    Embedding space reflects language structure
  ],
  block(fill: sand, inset: 0.8em, radius: 3pt, stroke: 0.4pt + luma(200))[
    *Aligned: Llama-3-8B-Instruct* \
    #v(0.3em)
    RLHF fine-tuned \
    Optimized for human preference \
    Embedding space geometrically reorganized
  ],
)

// --------------------------------------------------------------------------
== Linear Separability of Preferences

#v(0.3em)
*Key question:* Can a linear hyperplane separate preferred from rejected response vectors?

#v(0.4em)
#insight([Linear Probe])[
  We train a *LinearSVC* on embeddings of chosen vs. rejected responses from
  `Anthropic/hh-rlhf`. High accuracy $=>$ preferences are *linearly encoded* in the geometry.
]

#v(0.4em)
- Evaluated per-layer across all transformer layers
- Margin width as proxy for *geometric separation strength*
- Base vs. aligned model comparison quantifies the alignment effect

// --------------------------------------------------------------------------
== Bias Probes in Embedding Space

#v(0.3em)
RLHF may introduce systematic biases that manifest as geometric structure:

#v(0.4em)
- *Political bias:* Do "left-leaning" vs. "right-leaning" prompts cluster separately?
- *Cultural bias:* Are certain demographic groups encoded into separable subspaces?

#v(0.5em)
#remark[
  If biases are *linearly encoded*, they can in principle be identified and mitigated
  via geometric interventions — e.g. activation steering or representation editing.
]

// --------------------------------------------------------------------------
= Datasets & Methods

== Experimental Setup

#v(0.4em)
#table(
  columns: (auto, 1fr, 1fr),
  align: (left, left, left),
  stroke: none,
  fill: (_, row) => if row == 0 { navy } else if calc.odd(row) { rgb("#f0f4f9") } else { white },
  table.hline(stroke: 0.5pt + navy),
  [#text(fill: white, weight: "bold")[Part]],
  [#text(fill: white, weight: "bold")[Dataset]],
  [#text(fill: white, weight: "bold")[Methods]],
  table.hline(stroke: 0.3pt + luma(200)),
  [Vision],    [CIFAR-10],                    [CKA, Class Activation Maps],
  [Language],  [`Anthropic/hh-rlhf`],         [LinearSVC, Margin Analysis],
  [Language],  [`OpenAssistant/oasst1`],       [Bias Probes, Cluster Analysis],
  table.hline(stroke: 0.5pt + navy),
)

#v(0.5em)
*Toolkit:* PyTorch · HuggingFace Transformers · TruncatedSVD · UMAP · t-SNE · Plotly Dash

// --------------------------------------------------------------------------
= Summary

== Results at a Glance

#v(0.3em)
#table(
  columns: (auto, 1fr, 1fr),
  align: (left, left, left),
  stroke: none,
  fill: (_, row) => if row == 0 { navy } else if calc.odd(row) { rgb("#f0f4f9") } else { white },
  table.hline(stroke: 0.5pt + navy),
  [#text(fill: white, weight: "bold")[Part]],
  [#text(fill: white, weight: "bold")[Question]],
  [#text(fill: white, weight: "bold")[Expected Finding]],
  table.hline(stroke: 0.3pt + luma(200)),
  [Vision],    [CNN vs. ViT representational similarity?], [CKA profile diverges in mid-layers],
  [Vision],    [Which regions drive clusters?],            [CAMs reveal architecture-specific attention],
  [Language],  [Linear separability after RLHF?],          [High LinearSVC accuracy post-alignment],
  [Language],  [Bias as geometric structure?],              [Cluster analysis reveals bias subspaces],
  table.hline(stroke: 0.5pt + navy),
)

// --------------------------------------------------------------------------
== Conclusion & Open Questions

#v(0.3em)
*What we aim to show:*
- RLHF geometrically reorganizes embedding space in a measurable, linear way
- CNN and ViT representations diverge in intermediate layers despite similar accuracy
- Biases introduced by RLHF are geometrically identifiable via linear probes

#v(0.5em)
*Open questions:*
- Does geometric separation correlate with downstream task performance?
- Which layers are most affected by RLHF — early, middle, or late?
- Are the discovered bias subspaces *causally* linked to model outputs?

// --------------------------------------------------------------------------
= References

== References

#set text(size: 13pt)

#v(0.3em)
- Kucukahmetler et al. (2026). *Relative Geometry of Neural Forecasters: Linking Accuracy and Alignment in Learned Latent Geometry.* arXiv:2602.15676
- Kornblith et al. (2019). *Similarity of Neural Network Representations Revisited.* ICML. arXiv:1905.00414
- Ouyang et al. (2022). *Training language models to follow instructions with human feedback.* NeurIPS. arXiv:2203.02155
- Christiano et al. (2017). *Deep Reinforcement Learning from Human Preferences.* NeurIPS. arXiv:1706.03741
- McInnes et al. (2018). *UMAP: Uniform Manifold Approximation and Projection.* arXiv:1802.03426
- Dosovitskiy et al. (2020). *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.* arXiv:2010.11929
