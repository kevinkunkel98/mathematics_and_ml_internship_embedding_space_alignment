# Part 2 Expansion — Pipeline, Math Foundations, Plot Gallery — Design

## Context

`sections/02_part2_results.typ` currently gives Part 2 (RLHF geometry) one
slide of text-only findings (LinearSVC, Cohen's d, CKA drift numbers) with a
footnote pointing to "the dashboard" for full curves. Kevin's talk segment on
this part is 5 minutes and currently has no visual explanation of (a) how the
numbers were computed, or (b) what the metrics mathematically mean — both
expected for a Math & ML course audience.

Kevin exported 6 static PNGs from the dashboard into `assets/slides/`
(`Anisotropy.png`, `Cohensd.png`, `Rank.png`, `Drift.png`, `RLHF.png`,
`SVC.png` — line charts per Tulu-3-8B checkpoint, plus one t-SNE scatter).
These are not yet referenced anywhere in `slides.typ`.

`paper.typ` has an existing CKA derivation (HSIC, Gram-matrix formulas) but
it's older and more derivation-heavy than needed here — formulas for this
deck are written fresh from the actual running code
(`app/vision_compute.py::linear_cka`, `app/rlhf_geometry_compute.py`), not
copied from the paper.

## Goals

- Add a **pipeline slide**: how the Tulu-3 numbers were computed, dataset to
  dashboard.
- Add a **math foundations slide**: clean formal definitions of the 4
  metrics used in Part 2's findings (CKA, Cohen's d, anisotropy, effective
  rank), derived from the actual code, not `paper.typ`.
- Add a **plot gallery slide**: all 6 new PNGs from `assets/slides/`,
  compactly captioned, as visual backup for the results slide's claims.
- Keep the existing `02_part2_results.typ` text findings unchanged.

## Non-goals

- No changes to Part 1 (cross-modal) content.
- No copying of `paper.typ`'s HSIC/Gram-matrix derivation — final formulas
  only, matching what the code actually computes.
- Not fitting everything onto fewer slides — 3 new slides is accepted given
  the 5-minute segment; pacing is Kevin's concern, not this spec's.

## Design

### File structure (rename existing files, add 3 new)

```
sections/01_intro.typ                unchanged
sections/02_pipeline.typ             NEW — pipeline diagram
sections/03_metrics.typ              NEW — math foundations (4 definition boxes)
sections/04_part2_results.typ        (renamed from 02_part2_results.typ, content unchanged)
sections/05_plot_gallery.typ         NEW — 3x2 grid of the 6 PNGs
sections/06_part1_outlook.typ        (renamed from 03_part1_outlook.typ)
sections/07_conclusion.typ           (renamed from 04_conclusion.typ)
sections/08_references.typ           (renamed from 05_references.typ)
```

`slides.typ`'s `#include` list is updated to match the new order and
filenames. Renames are mechanical (`git mv` + path edit), no content changes
to the renamed files.

Deck order for Part 2: Intro -> Pipeline -> Math foundations -> Results
(text findings) -> Plot gallery (visual evidence) -> Part 1 outlook ->
Conclusion -> References.

### `sections/02_pipeline.typ` — Pipeline diagram

`cetz` canvas diagram, same visual language as the title-slide embedding
diagram in `slides.typ` (navy-stroked boxes, arrows, small caption text).
Horizontal chain of 5 stages:

1. `Anthropic/hh-rlhf` — 4,000 chosen/rejected pairs
2. Cluster extraction (SLURM) — Tulu-3-8B SFT / DPO / RLHF, forward pass,
   last-token pooled hidden states per layer
3. `data/embeddings/*/layers.h5` — 33 layers x 4,000 samples x 4,096 dims,
   per checkpoint
4. Geometry & CKA metrics — LinearSVC, Cohen's d, anisotropy, effective
   rank, CKA drift -> cached to `data/cache/*.pkl`
5. Dash dashboard (`app/app.py`) — interactive plots; this deck's PNGs are
   exports of it

Small gray caption line under the diagram naming the actual scripts
(`extract_embeddings.py`, `rlhf_geometry_compute.py`, `app.py`) for
credibility/reproducibility.

### `sections/03_metrics.typ` — Math foundations

2x2 grid of `definition()`-style boxes (reuse `helpers.typ` `definition`
helper — mint fill, sage stroke). No images on this slide — kept purely
formal. Each box: name, formula, one-line interpretation. Formulas derived
directly from the code, e.g.:

- **Linear CKA** (`vision_compute.linear_cka`): for column-centered
  $X, Y$,
  $ "CKA"(X,Y) = frac(norm(X^top Y)_F^2, norm(X^top X)_F dot.c norm(Y^top Y)_F) $
  — 1 = proportional representations (no drift), 0 = orthogonal.

- **Cohen's d** (`rlhf_geometry_compute.compute_cohens_d`): direction
  $v = (mu_"chosen" - mu_"rejected") / norm(mu_"chosen" - mu_"rejected")$,
  $ d = frac((mu_"chosen" - mu_"rejected") dot v, sigma_"pooled") $
  — effect size of the best linear separating direction.

- **Anisotropy** (`rlhf_geometry_compute.compute_anisotropy`):
  $ "aniso"(X) = EE_(i eq.not j) [cos(x_i, x_j)] $
  over random sampled pairs — near 1 = representations collapse to a
  narrow cone.

- **Effective rank** (`rlhf_geometry_compute.compute_effective_rank`),
  participation ratio of covariance eigenvalues $lambda_i$:
  $ "erank"(X) = frac((sum_i lambda_i)^2, sum_i lambda_i^2) $
  — how many dimensions are effectively in use.

### `sections/05_plot_gallery.typ` — Plot gallery

3x2 `grid` of the 6 PNGs from `assets/slides/`, each with a one-line
caption underneath (metric name + reading), small text (~0.7em) to match
the deck's existing density:

- `Drift.png` — "CKA drift — shift concentrated SFT to DPO, later layers"
- `Cohensd.png` — "Cohen's d — small, flat across all layers"
- `Anisotropy.png` — "Anisotropy — dips mid-stack, rises again late"
- `Rank.png` — "Effective rank — dimensionality usage per layer"
- `SVC.png` — "LinearSVC accuracy — pinned near chance (0.5)"
- `RLHF.png` — "t-SNE, layer 32 — chosen/rejected heavily overlap"

### `sections/04_part2_results.typ`

No content changes — file is renamed only (`02_` -> `04_` prefix).

## Open items

None — scope, formulas, and layout confirmed with Kevin during brainstorming.
