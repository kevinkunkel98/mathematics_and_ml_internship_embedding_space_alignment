# Final Presentation Slides Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `slides.typ` from the midterm deck (speculative "expected findings") into the final-presentation deck (2026-07-09) with real Tulu-3 RLHF results for Part 2, GNN-deck-style footer/slide-counter/title-slide, and content split into `sections/*.typ` files.

**Architecture:** This is a Typst document, not a software library — there is no unit-test framework. "Test" in every task below means: `typst compile slides.typ out.pdf` succeeds (exit 0) and `pdfinfo out.pdf | grep Pages` matches the expected page count for that task. Two tasks that change visual layout (footer/title slide) additionally render page 1 or 2 to PNG via ImageMagick (`magick`) and get inspected with the Read tool — this project's established way of visually verifying Typst output (see `slides.typ` history, midterm deck was checked the same way).

**Tech Stack:** Typst 0.15.0, `@preview/touying:0.6.1` (metropolis theme), `@preview/cetz:0.3.4`, ImageMagick `magick` for PNG rendering.

## Global Constraints

- Never `git commit` — this repo's standing rule is no commits without the user explicitly asking. Every "commit" step below is replaced with `git add` (stage only). Do not run `git commit`.
- Reuse the exact existing color palette (`navy #1c3a5e`, `blue #1a4f8a`, `sky #dce9f7`, `sage #1e6b3c`, `mint #e4f4ec`, `sand #f7f4ef`) — do not introduce new colors.
- Part 1 (cross-modal) content must stay speculative/hypothesis-framed — do not claim real results for Part 1.
- Part 2 real-results numbers are unknown until the dashboard (`app/app.py`, running separately) finishes computing — use literal bracket tokens `[LAYER]` and `[ACCURACY]` as content placeholders in `sections/02_part2_results.typ`, to be hand-filled later. This is the one intentional exception to "no placeholders" — it's a documented, approved part of the design spec (`docs/superpowers/specs/2026-07-05-final-presentation-slides-design.md`), not a shortcut.
- Presentation date for the footer: `09.07.2026`.

---

### Task 1: Extract shared palette/helpers into `helpers.typ`

**Files:**
- Create: `helpers.typ`
- Modify: `slides.typ:5-28` (remove inline palette/helper defs, replace with import)
- Test: manual compile (`typst compile slides.typ`)

**Interfaces:**
- Produces: `helpers.typ` exports `navy`, `blue`, `sky`, `sage`, `mint`, `sand`, `thm-box`, `insight`, `definition`, `remark` — every later task that adds a `sections/*.typ` file imports these via `#import "../helpers.typ": *`.

- [ ] **Step 1: Create `helpers.typ` with the palette and box helpers currently inline in `slides.typ`**

```typst
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
```

- [ ] **Step 2: Replace the inline defs in `slides.typ` with an import**

In `slides.typ`, replace lines 5-28 (the `// ── Palette ──` comment through the `#let remark(...)` line) with:

```typst
#import "helpers.typ": *
```

- [ ] **Step 3: Compile and verify it still produces 5 pages**

Run: `typst compile slides.typ /tmp/final-slides-check.pdf && pdfinfo /tmp/final-slides-check.pdf | grep Pages`
Expected: `Pages:           5` (no content changed, just moved to another file)

- [ ] **Step 4: Stage (do not commit)**

```bash
git add helpers.typ slides.typ
```

---

### Task 2: Copy the university logo asset

**Files:**
- Create: `leipziglogo.png` (copy of `seminar_recurrent_graph_neural_networks/leipziglogo.png`)

**Interfaces:**
- Produces: `leipziglogo.png` at repo root — Task 9 (title slide) references it as `image("leipziglogo.png", width: 12em)`.

- [ ] **Step 1: Copy the file**

```bash
cp seminar_recurrent_graph_neural_networks/leipziglogo.png leipziglogo.png
```

- [ ] **Step 2: Verify it copied correctly**

Run: `ls -la leipziglogo.png && file leipziglogo.png`
Expected: file exists, `file` reports a PNG image (not a broken/zero-byte copy).

- [ ] **Step 3: Stage**

```bash
git add leipziglogo.png
```

---

### Task 3: Extract Core Questions slide into `sections/01_intro.typ`

**Files:**
- Create: `sections/01_intro.typ`
- Modify: `slides.typ:161-199` (remove the `== Core Questions` slide, replace with an include)

**Interfaces:**
- Consumes: `helpers.typ` (`navy`, `blue`, `sky`, `sage`, `mint`) from Task 1.
- Produces: nothing consumed by later tasks — this is a self-contained content slide.

- [ ] **Step 1: Create `sections/01_intro.typ` with the Core Questions content moved verbatim from `slides.typ`**

```typst
#import "../helpers.typ": *

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
      - Does alignment progressively separate chosen/rejected responses across SFT → DPO → RLHF?
      - How does separation evolve *layer by layer*?

      #v(0.3em)
      #text(size: 0.8em, fill: luma(80))[UMAP + LinearSVC per layer · Tulu-3-8B SFT/DPO/RLHF · `Anthropic/hh-rlhf`]
    ]
  ],
)
```

(Note: the Part 2 bullet text is updated from "Does Instruct separate..." to "Does alignment progressively separate... across SFT → DPO → RLHF?" since the final deck's Part 2 now uses the 3-checkpoint Tulu-3 trajectory, not a 2-way base/Instruct comparison. Caption line updated to match.)

- [ ] **Step 2: Remove the `== Core Questions` slide block from `slides.typ` (currently lines 161-199) and replace with an include, placed after the title slide's closing `]`**

```typst
#include "sections/01_intro.typ"
```

- [ ] **Step 3: Compile and verify still 5 pages**

Run: `typst compile slides.typ /tmp/final-slides-check.pdf && pdfinfo /tmp/final-slides-check.pdf | grep Pages`
Expected: `Pages:           5`

- [ ] **Step 4: Stage**

```bash
git add sections/01_intro.typ slides.typ
```

---

### Task 4: Extract References slide into `sections/05_references.typ`, add Tulu-3 citation

**Files:**
- Create: `sections/05_references.typ`
- Modify: `slides.typ:278-317` (remove the `== References` slide, replace with an include)

**Interfaces:**
- Consumes: `helpers.typ` (`navy`) from Task 1.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Create `sections/05_references.typ` — existing 6 citations plus a new Tulu-3 entry**

```typst
#import "../helpers.typ": *

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
  [
    #text(weight: "bold", fill: navy)[Lambert et al. (2024)] \
    _Tulu 3: Pushing Frontiers in Open Language Model Post-Training_ \
    #text(fill: luma(100))[arXiv:2411.15124 · *Models used for Part 2*]
  ],
)
```

- [ ] **Step 2: Remove the `== References` slide block from `slides.typ` (currently lines 278-317) and replace with an include, placed at the end of the file**

```typst
#include "sections/05_references.typ"
```

- [ ] **Step 3: Compile and verify still 5 pages**

Run: `typst compile slides.typ /tmp/final-slides-check.pdf && pdfinfo /tmp/final-slides-check.pdf | grep Pages`
Expected: `Pages:           5`

- [ ] **Step 4: Stage**

```bash
git add sections/05_references.typ slides.typ
```

---

### Task 5: Replace the two two-column slides with `sections/03_part1_outlook.typ` (Part 1 only, kept speculative)

**Files:**
- Create: `sections/03_part1_outlook.typ`
- Modify: `slides.typ` (remove the `== Setup & Expected Findings` slide and the `== Outlook & Deliverable` slide, replace both with a single include)

**Interfaces:**
- Consumes: `helpers.typ` (`navy`, `blue`, `sky`) from Task 1; `assets/slides/dashboard_cka.png` (existing asset, unchanged).
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Create `sections/03_part1_outlook.typ` — Part 1 content merged from both old slides**

```typst
#import "../helpers.typ": *

// --------------------------------------------------------------------------
== Part 1 — Cross-modal Alignment · Outlook

#v(0.2em)
#grid(
  columns: (1fr, 1fr),
  gutter: 0.9em,
  align: top,
  [
    #block(fill: navy, inset: (x: 0.7em, y: 0.4em), radius: (top: 3pt), width: 100%)[
      #text(fill: white, weight: "bold", size: 0.82em)[Setup · DINOv2 + Llama · MS-COCO · multi-class prediction]
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
    #image("assets/slides/dashboard_cka.png")
    #v(0.1em)
    #text(size: 8.5pt, fill: luma(120))[Interactive Plotly Dash · CKA heatmap]

    #v(0.3em)
    #block(fill: navy, inset: (x: 0.7em, y: 0.4em), radius: (top: 3pt), width: 100%)[
      #text(fill: white, weight: "bold", size: 0.8em)[Next steps — toward final]
    ]
    #block(fill: sky, inset: (x: 0.7em, y: 0.45em), radius: (bottom: 3pt), stroke: (left: 3pt + blue), width: 100%)[
      - Cross-modal CKA: DINOv2 vs. Llama on MS-COCO pairs
      - *CLIP (ViT-B/32)* as upper bound — explicitly trained cross-modal alignment
      - Compare base vs. Instruct CKA scores
    ]
  ],
)
```

- [ ] **Step 2: Remove the `== Setup & Expected Findings` and `== Outlook & Deliverable` slide blocks from `slides.typ`, replace both with a single include placed where `== Setup & Expected Findings` used to be**

```typst
#include "sections/03_part1_outlook.typ"
```

- [ ] **Step 3: Compile and verify page count dropped to 4**

Run: `typst compile slides.typ /tmp/final-slides-check.pdf && pdfinfo /tmp/final-slides-check.pdf | grep Pages`
Expected: `Pages:           4` (two old slides removed, one new slide added: 5 − 2 + 1 = 4)

- [ ] **Step 4: Stage**

```bash
git add sections/03_part1_outlook.typ slides.typ
```

---

### Task 6: Add `sections/02_part2_results.typ` — real Tulu-3 results (placeholders for numbers)

**Files:**
- Create: `sections/02_part2_results.typ`
- Modify: `slides.typ` (add include between the Task 3 include and the Task 5 include)

**Interfaces:**
- Consumes: `helpers.typ` (`navy`, `blue`, `sage`, `mint`, `sand`) from Task 1.
- Produces: nothing consumed by later tasks. Contains literal `[LAYER]` / `[ACCURACY]` tokens per the Global Constraints — hand-filled once the dashboard finishes computing.

- [ ] **Step 1: Create `sections/02_part2_results.typ`**

```typst
#import "../helpers.typ": *

// --------------------------------------------------------------------------
== Part 2 — RLHF Geometry · Real Results

#v(0.3em)
#block(fill: navy, inset: (x: 0.8em, y: 0.55em), radius: (top: 3pt), width: 100%)[
  #text(fill: white, weight: "bold", size: 0.9em)[Tulu-3-8B SFT → DPO → RLHF · `Anthropic/hh-rlhf` · 4,000 samples · 33 layers]
]
#block(fill: mint, inset: (x: 0.9em, y: 0.65em), radius: (bottom: 3pt), stroke: (left: 3pt + sage), width: 100%)[
  #text(weight: "bold", fill: sage, size: 0.9em)[Insight:] Chosen/rejected separability is near-chance in SFT, then jumps sharply after preference optimization

  #v(0.35em)
  - Peak linear separability: layer *[LAYER]*, LinearSVC accuracy *[ACCURACY]*
  - SFT checkpoint: near-chance separability across all layers — no preference signal yet
  - DPO / RLHF checkpoints: sharp phase transition, pinpointing *where* alignment lives
]

#v(0.4em)
#align(center)[
  #block(stroke: 1pt + blue, radius: 4pt, clip: true, width: 80%)[
    // TODO: replace this placeholder rect with a real dashboard screenshot
    // (Part 2 tab, UMAP scatter at the peak layer) once computation completes.
    #rect(width: 100%, height: 200pt, fill: sand)[
      #align(center + horizon)[#text(fill: luma(140), style: "italic")[UMAP scatter — layer [LAYER] — SFT vs. DPO vs. RLHF]]
    ]
  ]
  #v(0.2em)
  #text(size: 8pt, fill: luma(120))[Tulu-3-8B alignment trajectory · dashboard Part 2 tab]
]
```

- [ ] **Step 2: Add the include in `slides.typ`, right after `#include "sections/01_intro.typ"` and before `#include "sections/03_part1_outlook.typ"`**

```typst
#include "sections/02_part2_results.typ"
```

- [ ] **Step 3: Compile and verify page count is now 5**

Run: `typst compile slides.typ /tmp/final-slides-check.pdf && pdfinfo /tmp/final-slides-check.pdf | grep Pages`
Expected: `Pages:           5`

- [ ] **Step 4: Stage**

```bash
git add sections/02_part2_results.typ slides.typ
```

---

### Task 7: Add `sections/04_conclusion.typ` — deliverable + perspective wrap-up

**Files:**
- Create: `sections/04_conclusion.typ`
- Modify: `slides.typ` (add include between the Task 5 include and the Task 4 include)

**Interfaces:**
- Consumes: `helpers.typ` (`navy`, `blue`, `sky`, `sage`, `mint`) from Task 1; `assets/slides/dashboard_vit_umap.png` (existing asset, unchanged).
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Create `sections/04_conclusion.typ`**

```typst
#import "../helpers.typ": *

// --------------------------------------------------------------------------
== Deliverable & Next Steps

#v(0.3em)
#grid(
  columns: (1fr, 1fr),
  gutter: 0.9em,
  align: top,
  [
    #image("assets/slides/dashboard_vit_umap.png")
    #v(0.1em)
    #text(size: 8.5pt, fill: luma(120))[Interactive Plotly Dash · CKA heatmap · UMAP layer slider · LinearSVC score]
  ],
  [
    #block(fill: navy, inset: (x: 0.7em, y: 0.4em), radius: (top: 3pt), width: 100%)[
      #text(fill: white, weight: "bold", size: 0.8em)[Project deliverable]
    ]
    #block(fill: mint, inset: (x: 0.7em, y: 0.45em), radius: (bottom: 3pt), stroke: (left: 3pt + sage), width: 100%)[
      The dashboard *is* the deliverable — interactive visualization of all findings: layer slider, CKA heatmaps, UMAP scatter, SVC score
    ]

    #v(0.3em)
    #block(fill: navy, inset: (x: 0.7em, y: 0.4em), radius: (top: 3pt), width: 100%)[
      #text(fill: white, weight: "bold", size: 0.8em)[Perspective]
    ]
    #block(fill: sky, inset: (x: 0.7em, y: 0.45em), radius: (bottom: 3pt), stroke: (left: 3pt + blue), width: 100%)[
      Extend to RLHF training *snapshots* — plot cross-modal CKA vs. training step to see if alignment shifts *during* RLHF
    ]
  ],
)
```

- [ ] **Step 2: Add the include in `slides.typ`, right after `#include "sections/03_part1_outlook.typ"` and before `#include "sections/05_references.typ"`**

```typst
#include "sections/04_conclusion.typ"
```

- [ ] **Step 3: Compile and verify page count is now 6**

Run: `typst compile slides.typ /tmp/final-slides-check.pdf && pdfinfo /tmp/final-slides-check.pdf | grep Pages`
Expected: `Pages:           6`

- [ ] **Step 4: Stage**

```bash
git add sections/04_conclusion.typ slides.typ
```

---

### Task 8: Add footer with slide counter and presentation date

**Files:**
- Modify: `slides.typ` (the `#show: metropolis-theme.with(...)` call)

**Interfaces:**
- Consumes: nothing new.
- Produces: nothing consumed by later tasks — purely a visual/theme change.

- [ ] **Step 1: Add `footer-right` and `footer` to the theme config**

In `slides.typ`, the `#show: metropolis-theme.with(...)` block currently starts:

```typst
#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  config-colors(
```

Change it to:

```typst
#show: metropolis-theme.with(
  footer-right: context {
    if state("show-slide-number", true).get() { utils.slide-counter.display() }
  },
  aspect-ratio: "16-9",
  config-colors(
```

and after the closing of `config-info(...)`, before the final closing `)` of `metropolis-theme.with(`, add:

```typst
  footer: [Leipzig, 09.07.2026],
```

(matching the exact pattern in `seminar_recurrent_graph_neural_networks/slides.typ:8-10,27`)

- [ ] **Step 2: Compile and verify it still succeeds with 6 pages**

Run: `typst compile slides.typ /tmp/final-slides-check.pdf && pdfinfo /tmp/final-slides-check.pdf | grep Pages`
Expected: `Pages:           6`

- [ ] **Step 3: Render page 2 (first content slide, not the title slide) to PNG and visually verify the footer shows a slide number and the date**

```bash
magick -density 150 /tmp/final-slides-check.pdf[1] /tmp/final-slides-check-page2.png
```

Then use the Read tool on `/tmp/final-slides-check-page2.png` and confirm: bottom-right shows a slide number, footer shows "Leipzig, 09.07.2026". (Title slide, page 1, overrides `header`/`footer` to `_ => none` so it correctly shows neither — do not check page 1 for this.)

- [ ] **Step 4: Stage**

```bash
git add slides.typ
```

---

### Task 9: Restyle the title slide (eyebrow line + logo, drop the 5-name grid)

**Files:**
- Modify: `slides.typ` (the custom title `#slide(...)` block)

**Interfaces:**
- Consumes: `leipziglogo.png` from Task 2.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Replace the title slide's left-column content**

The current left column of the title `#slide(...)` (composer `(1fr, 1.1fr)`) is:

```typst
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
```

Replace it with:

```typst
  #set align(left)
  #v(1fr)
  #text(size: 10pt, fill: accent, tracking: 2pt)[MATHEMATICS \& ML INTERNSHIP · SS 2026]
  #v(0.5em)
  #set par(leading: 0.75em)
  #text(size: 28pt, weight: "bold", fill: white)[Representational Geometry \ in Neural Networks]
  #v(0.3em)
  #text(size: 13pt, fill: accent)[From Vision Transformers to RLHF-Aligned Language Models]
  #v(0.85em)
  #line(length: 100%, stroke: 1pt + accent)
  #v(0.65em)
  #grid(
    columns: (1fr, auto),
    align: (left + horizon, right + horizon),
    [
      #text(size: 11pt, weight: "bold", fill: white)[Huxhold · Pollinger · Kunigk · Kunkel · Charki]
      #v(0.15em)
      #text(size: 9.5pt, fill: accent)[
        Universität Leipzig — Supervisor: Dr. Diaaeldin Taha
      ]
    ],
    [#image("leipziglogo.png", width: 9em)],
  )
  #v(1fr)
```

(The `\&` escapes the ampersand — Typst treats bare `&` as a markup construct in some contexts. Font size dropped slightly from 10.5pt to fit the tighter single-line author row; institution line shortened since the eyebrow line already states "Mathematics & ML Internship".)

- [ ] **Step 2: Compile and verify it still succeeds with 6 pages**

Run: `typst compile slides.typ /tmp/final-slides-check.pdf && pdfinfo /tmp/final-slides-check.pdf | grep Pages`
Expected: `Pages:           6`

- [ ] **Step 3: Render page 1 to PNG and visually verify the new title slide layout**

```bash
magick -density 150 /tmp/final-slides-check.pdf[0] /tmp/final-slides-check-page1.png
```

Then use the Read tool on `/tmp/final-slides-check-page1.png` and confirm: small-caps eyebrow line above the title, single author line (not a 5-cell grid), logo visible bottom-right of the left column, no visual overlap or clipping.

- [ ] **Step 4: Stage**

```bash
git add slides.typ
```

---

### Task 10: Final full-deck verification

**Files:** none (verification only)

- [ ] **Step 1: Full compile from a clean state**

```bash
rm -f /tmp/final-slides-check.pdf
typst compile slides.typ /tmp/final-slides-check.pdf
```

Expected: exit 0, no errors or warnings printed.

- [ ] **Step 2: Verify final page count is 6**

```bash
pdfinfo /tmp/final-slides-check.pdf | grep Pages
```

Expected: `Pages:           6` (title, intro, part2-results, part1-outlook, conclusion, references)

- [ ] **Step 3: Render all 6 pages to PNG and visually skim each one**

```bash
magick -density 150 /tmp/final-slides-check.pdf /tmp/final-slides-check-%d.png
```

Read each `/tmp/final-slides-check-0.png` through `-5.png` with the Read tool. Confirm: no overflow/clipped text, footer/slide-counter present on pages 2-6 (not page 1), logo renders on page 1, Part 2 slide shows the `[LAYER]`/`[ACCURACY]` placeholder tokens clearly (not broken Typst syntax).

- [ ] **Step 4: Report remaining manual work to the user**

Tell the user: deck compiles clean at 6 pages; `sections/02_part2_results.typ` still has `[LAYER]` / `[ACCURACY]` placeholders and a placeholder gray rect instead of a real UMAP screenshot — both need to be filled in once the dashboard (`app/app.py`) finishes computing and the real Tulu-3 numbers/screenshot are available.
