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
  config-methods(new-section-slide-fn: none, title-slide-fn: none),
  config-info(
    title: [Representational Geometry in Neural Networks],
    subtitle: [From Vision Transformers to RLHF-Aligned Language Models],
    author: [Huxhold · Pollinger · Kunigk · Kunkel · Charki],
    date: [Summer Semester 2026],
    institution: [Universität Leipzig — Mathematics & Machine Learning Internship \ Supervisor: Dr. Diaaeldin Taha],
  ),
)

#set text(size: 17pt)

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

// --------------------------------------------------------------------------
== Core Questions

#v(0.3em)
#grid(
  columns: (1fr, 1fr),
  gutter: 1.0em,
  align: top,
  [
    #block(fill: navy, inset: (x: 0.8em, y: 0.55em), radius: (top: 3pt))[
      #text(fill: white, weight: "bold", size: 0.9em)[Part 1 — Cross-modal alignment]
    ]
    #block(fill: sky, inset: (x: 0.8em, y: 0.55em), radius: (bottom: 3pt), stroke: (left: 3pt + blue), width: 100%)[
      When a vision model and a language model train on the *same task*, does a shared representational structure emerge across modalities?

      #v(0.3em)
      - *Hypothesis A:* cross-modal signal is new information → improves model
      - *Hypothesis B:* models converge regardless — extra signal adds little

      #v(0.3em)
      #text(size: 0.8em, fill: luma(80))[CKA before/after cross-modal fine-tuning · DINOv2 + Llama · MS-COCO]
    ]
  ],
  [
    #block(fill: navy, inset: (x: 0.8em, y: 0.45em), radius: (top: 3pt))[
      #text(fill: white, weight: "bold", size: 0.85em)[Part 2 — RLHF geometry]
    ]
    #block(fill: mint, inset: (x: 0.8em, y: 0.55em), radius: (bottom: 3pt), stroke: (left: 3pt + sage), width: 100%)[
      Does RLHF geometrically transform a language model's embedding space — encoding human preference as a *linearly separable structure*?

      #v(0.3em)
      - Does Instruct separate chosen/rejected responses that base cannot?
      - How does separation evolve *layer by layer*?

      #v(0.3em)
      #text(size: 0.8em, fill: luma(80))[UMAP + LinearSVC per layer · Llama-3-8B base vs. Instruct · `Anthropic/hh-rlhf`]
    ]
  ],
)

// --------------------------------------------------------------------------
== Setup & Expected Findings

#v(0.2em)
#grid(
  columns: (1fr, 1fr),
  gutter: 0.9em,
  align: top,
  [
    #block(fill: navy, inset: (x: 0.7em, y: 0.4em), radius: (top: 3pt), width: 100%)[
      #text(fill: white, weight: "bold", size: 0.82em)[Part 1 — Cross-modal · DINOv2 + Llama · MS-COCO · multi-class prediction]
    ]
    #block(fill: sky, inset: (x: 0.8em, y: 0.5em), radius: (bottom: 3pt), stroke: (left: 3pt + blue), width: 100%)[
      Both models trained on *same task* (object category prediction) · measure CKA · fine-tune with loss term encouraging similarity to the other model's representation · compare

      #v(0.25em)
      *CKA high before fine-tuning:* models converge inherently *(Platonic Representation Hypothesis)*

      *CKA rises after fine-tuning:* cross-modal signal is genuinely new information

      #v(0.2em)
      #text(size: 0.78em, fill: luma(60))[Layer-wise CKA map shows *where* shared meaning lives · UMAP + dashboard to visualize]
    ]
  ],
  [
    #block(fill: navy, inset: (x: 0.7em, y: 0.4em), radius: (top: 3pt), width: 100%)[
      #text(fill: white, weight: "bold", size: 0.82em)[Part 2 — RLHF geometry · Llama-3-8B base vs. Instruct]
    ]
    #block(fill: mint, inset: (x: 0.8em, y: 0.5em), radius: (bottom: 3pt), stroke: (left: 3pt + sage), width: 100%)[
      UMAP + LinearSVC per layer · `Anthropic/hh-rlhf`

      #v(0.2em)
      *Phase transition:* base near-chance SVC — Instruct shows sharp jump, pinpointing *where* alignment lives

      *UMAP split:* chosen/rejected mixed in base, cleanly separated in Instruct

      #v(0.2em)
      *Perspective:* extend to RLHF training *snapshots* — plot cross-modal CKA vs. training step to see if alignment shifts *during* RLHF

      #v(0.15em)
      #text(size: 0.78em, fill: luma(60))[He-Trott-Khosla 2025 · pile of LMs + vision models]
    ]
  ],
)

// --------------------------------------------------------------------------
== Outlook & Deliverable

#v(0.2em)
#grid(
  columns: (1.1fr, 0.9fr),
  gutter: 0.9em,
  align: top,
  [
    #image("assets/slides/dashboard_cka.png")
    #v(0.1em)
    #text(size: 8.5pt, fill: luma(120))[Interactive Plotly Dash · CKA heatmap · UMAP layer slider · LinearSVC score]
  ],
  [
    #block(fill: navy, inset: (x: 0.7em, y: 0.4em), radius: (top: 3pt), width: 100%)[
      #text(fill: white, weight: "bold", size: 0.8em)[Next steps — toward July]
    ]
    #block(fill: sky, inset: (x: 0.7em, y: 0.45em), radius: (bottom: 3pt), stroke: (left: 3pt + blue), width: 100%)[
      - Cross-modal CKA: DINOv2 vs. Llama on MS-COCO pairs
      - *CLIP (ViT-B/32)* as upper bound — explicitly trained cross-modal alignment
      - Compare base vs. Instruct CKA scores
    ]
    #v(0.3em)
    #block(fill: navy, inset: (x: 0.7em, y: 0.4em), radius: (top: 3pt), width: 100%)[
      #text(fill: white, weight: "bold", size: 0.8em)[Project deliverable]
    ]
    #block(fill: mint, inset: (x: 0.7em, y: 0.45em), radius: (bottom: 3pt), stroke: (left: 3pt + sage), width: 100%)[
      The dashboard *is* the deliverable — interactive visualization of all findings: layer slider, CKA heatmaps, UMAP scatter, SVC score
    ]
  ],
)

// --------------------------------------------------------------------------
== References

#set text(size: 12pt)
#v(0.3em)
#grid(
  columns: (1fr, 1fr),
  gutter: (0.6em, 0.4em),
  align: top,
  [
    #text(weight: "bold", fill: navy)[He, Trott, Khosla (2025)] \
    _Shared Latent Representations across Vision and Language_ \
    #text(fill: luma(100))[arXiv:2509.20751 · *Anchor paper*]
  ],
  [
    #text(weight: "bold", fill: navy)[Kucukahmetler et al. (2026)] \
    _Relative Geometry of Neural Forecasters_ \
    #text(fill: luma(100))[TMLR · arXiv:2602.15676]
  ],
  [
    #text(weight: "bold", fill: navy)[Kornblith et al. (2019)] \
    _Similarity of Neural Network Representations Revisited_ \
    #text(fill: luma(100))[ICML · arXiv:1905.00414 · *CKA method*]
  ],
  [
    #text(weight: "bold", fill: navy)[Ouyang et al. (2022)] \
    _Training LMs to Follow Instructions with Human Feedback_ \
    #text(fill: luma(100))[NeurIPS · arXiv:2203.02155 · *InstructGPT / RLHF*]
  ],
  [
    #text(weight: "bold", fill: navy)[Christiano et al. (2017)] \
    _Deep RL from Human Preferences_ \
    #text(fill: luma(100))[NeurIPS · arXiv:1706.03741 · *RLHF foundations*]
  ],
  [
    #text(weight: "bold", fill: navy)[McInnes et al. (2018)] \
    _UMAP: Uniform Manifold Approximation and Projection_ \
    #text(fill: luma(100))[arXiv:1802.03426 · *Dim. reduction method*]
  ],
)
