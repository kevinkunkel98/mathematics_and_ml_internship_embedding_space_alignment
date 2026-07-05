# Final Presentation Slides — Design

## Context

`slides.typ` currently holds the midterm deck (5 slides: title, Core Questions,
Setup & Expected Findings, Outlook & Deliverable, References). It frames Part 2
(RLHF geometry) as speculative "expected findings" because at the time no real
Tulu-3 extraction had run.

As of 2026-07-05, real Tulu-3 SFT/DPO/RLHF embeddings have been extracted on
the cluster (33 layers x 4000 samples per checkpoint) and are loading into the
local dashboard (`app/app.py`) for inspection. The final presentation
(2026-07-09) should replace Part 2's speculative framing with real results.

Part 1 (cross-modal, DINOv2 x Llama x MS-COCO) still only has mock data
(`data/embeddings/crossmodal/*.h5`, dated 2026-05-18) — it stays speculative
future-work framing, same as midterm.

Style reference: `seminar_recurrent_graph_neural_networks/slides.typ` (GNN
seminar deck, same author) — already shares this project's color palette and
`thm-box` helper. Structural elements to adopt: footer with live slide-number
counter + date, content split into per-section files included from
`slides.typ`, and a compact eyebrow-line + logo title slide instead of the
5-name author grid.

## Goals

- Swap Part 2 from "expected findings" to real Tulu-3 alignment-trajectory
  results once dashboard numbers/screenshots are available.
- Adopt GNN deck's footer/slide-counter and title-slide style.
- Split slide content into `sections/*.typ` files for maintainability as the
  deck grows with real results.
- Keep Part 1 (cross-modal) speculative, unchanged in substance from midterm.

## Non-goals

- No content changes to Part 1's research question/hypothesis framing.
- No redesign of the color palette or `thm-box`/`definition`/`remark` helpers
  — already shared with the GNN deck, kept as-is.
- Not blocking on the dashboard finishing — Part 2 numbers ship as
  placeholders (`[LAYER]`, `[ACCURACY]`, screenshot TODO) to be filled in once
  the dashboard computation completes.

## Design

### File structure

```
slides.typ                          # theme config + title slide + includes
sections/01_intro.typ                # Core Questions (Part 1 + Part 2), trimmed
sections/02_part2_results.typ        # NEW — real Tulu-3 SFT->DPO->RLHF results
sections/03_part1_outlook.typ        # cross-modal, kept speculative
sections/04_conclusion.typ           # NEW — short wrap-up: deliverable + next steps
sections/05_references.typ           # existing 6 citations + Tulu-3 citation
leipziglogo.png                      # copied from seminar_recurrent_graph_neural_networks/
```

### Theme config changes (`slides.typ`)

Add to `metropolis-theme.with(...)`:

```typst
footer-right: context {
  if state("show-slide-number", true).get() { utils.slide-counter.display() }
},
footer: [Leipzig, 09.07.2026],
```

(mirrors `seminar_recurrent_graph_neural_networks/slides.typ` lines 8-10, 27)

### Title slide

Replace the 5-name grid layout with GNN-style:

- Small-caps eyebrow line: `MATHEMATICS & ML INTERNSHIP · SS 2026`
- Title + subtitle (unchanged text)
- Single author line: `Huxhold · Pollinger · Kunigk · Kunkel · Charki`
- Institution line (unchanged)
- Bottom row: `grid(columns: (1fr, auto), ...)` — author/institution left,
  `leipziglogo.png` right (matches GNN deck's title-slide bottom grid)

Keep the existing embedding-space `cetz` diagram and dashboard screenshot on
the right composer panel — those are project-specific and not part of the
GNN-style borrow.

### `sections/01_intro.typ` — Core Questions

Content unchanged from current midterm slide (Part 1 + Part 2 hypothesis
grid) — just extracted into its own file, `#include`d from `slides.typ`.

### `sections/02_part2_results.typ` — Part 2 real results (NEW, centerpiece)

Structure:

- Section heading: `RLHF Geometry — Real Results · Tulu-3-8B SFT -> DPO -> RLHF`
- Phase-transition claim box (`insight` helper): base/SFT near-chance
  LinearSVC separability -> DPO/RLHF shows a jump at layer `[LAYER]`,
  reaching `[ACCURACY]`
- UMAP scatter screenshot placeholder: `// TODO: screenshot from dashboard
  once computation completes — chosen/rejected split at layer [LAYER]`
- Short takeaway line connecting back to the RLHF-as-geometric-audit framing
  from the project README

Placeholders use literal bracketed tokens (`[LAYER]`, `[ACCURACY]`) so a
find-and-replace pass fills them in once the dashboard is up — no code
changes needed at that point, just content edits to this one file.

### `sections/03_part1_outlook.typ` — Part 1, kept speculative

Content ports over the current midterm "Setup & Expected Findings" Part 1
box and "Outlook & Deliverable" Part 1 content unchanged — hypothesis framing,
CLIP upper-bound baseline, planned method. No claims of real results.

### `sections/04_conclusion.typ` — NEW

Short single slide: dashboard screenshot (existing `dashboard_cka.png` or a
new Part 2 screenshot), one-line "the dashboard is the deliverable" callout,
2-3 bullet next-steps (extending Part 1 to real fine-tuning runs, RLHF
training-snapshot trajectory as stretch goal).

### `sections/05_references.typ`

Existing 6 citations carried over unchanged, plus one new entry:

```typst
[
  #text(weight: "bold", fill: navy)[Lambert et al. (2024)] \
  _Tulu 3: Pushing Frontiers in Open Language Model Post-Training_ \
  #text(fill: luma(100))[arXiv:2411.15124 · *Models used for Part 2*]
],
```

## Open items (resolved as placeholders, not blockers)

- Exact peak-layer accuracy and layer index for Part 2 — filled in once
  dashboard finishes computing (currently running, ~30+ min elapsed on real
  4000-sample x 4096-dim data across 33 layers x 3 checkpoints x
  {UMAP, t-SNE, LinearSVC}).
- Part 2 UMAP screenshot — captured from the running dashboard once available.
