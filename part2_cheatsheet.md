---
title: Part 2 Cheat Sheet — RLHF Geometry Dashboard
---

# Part 2: RLHF Geometry — What Was Computed & Results

Setup: Tulu-3-8B, 3 checkpoints (**SFT → DPO → RLHF/RLVR**), same base model, extracted at each post-training stage. `Anthropic/hh-rlhf`, 4,000 chosen/rejected texts (2,000 pairs), last-token pooled, 33 layers (layer 0–32), 4,096-dim.

## 1. UMAP / t-SNE + LinearSVC (`app/compute.py`)

**What:** per layer, per checkpoint — fit UMAP (2D) and t-SNE (2D) on the 4,096-dim activations; fit LinearSVC (5-fold CV) to classify chosen vs. rejected.

**Dashboard controls:** model selector (SFT/DPO/RLHF) · projection selector (UMAP/t-SNE) · layer slider.

**Result — negative finding:**
| Checkpoint | Peak LinearSVC accuracy | Peak layer |
|---|---|---|
| SFT | 0.5435 | 14 |
| DPO | 0.5443 | 14 |
| RLHF | 0.5502 | 14 |

- Near chance (~0.55) at **every layer, every checkpoint** — no phase transition, no layer where separation suddenly appears.
- UMAP/t-SNE scatter: chosen/rejected visibly intermixed at all layers, all 3 checkpoints — no visual cluster separation either.
- **Takeaway:** preference is not linearly encoded in raw last-token-pooled representations, at any stage of alignment.

## 2. Representational Drift (`app/rlhf_drift_compute.py`)

**What:** linear CKA between checkpoints, same input, same layer index (matched-layer drift) — how much does the geometry change SFT→DPO, DPO→RLHF, SFT→RLHF.

**Dashboard:** `drift-line` graph, one line per checkpoint pair, x-axis = layer.

**Result:**
| Pair | Layer 0 | Layer 32 (final) |
|---|---|---|
| SFT vs. DPO | 1.0000 | **0.9764** |
| DPO vs. RLHF | 1.0000 | 0.9990 |
| SFT vs. RLHF | 1.0000 | 0.9746 |

- **Most of the geometric shift happens at the SFT → DPO step.**
- DPO → RLHF barely moves the geometry further (stays ≥0.999 throughout).
- So: DPO reshapes representations substantially; the subsequent RL step is a small refinement on top, not a second big shift.

## 3. Full Layer × Layer CKA Heatmaps (`app/rlhf_drift_compute.py::fit_rlhf_cka_matrices`)

**What:** full 33×33 CKA matrix between every layer of checkpoint A and every layer of checkpoint B (not just matched indices) — checks whether representations shift to a *different depth*, not just drift in place.

**Dashboard:** 3 heatmaps (SFT×DPO, DPO×RLHF, SFT×RLHF).

**Result:** peak alignment stays on/near the diagonal (same layer index) in all 3 comparisons — no evidence that post-training pushes function to a different depth, consistent with the matched-layer drift finding above.

## 4. Geometry Diagnostics (`app/rlhf_geometry_compute.py`)

Three per-layer, per-checkpoint metrics computed on raw activations (not requiring any classifier):

| Metric | What it measures | Near value = |
|---|---|---|
| **Anisotropy** | avg. cosine similarity between random embedding pairs | 1 → representations collapse into a narrow cone |
| **Cohen's d** | chosen/rejected effect size along the single best linear direction (mean-difference direction) — catches a subtle signal even where regularized LinearSVC finds none | 0 → no separation at all |
| **Effective rank** | participation ratio of the covariance spectrum | how many dimensions are actually in use |

**Result (SFT / DPO / RLHF are nearly identical at every layer — shown here for layer 0 / 16 / 32):**

| Layer | Anisotropy | Cohen's d | Effective rank |
|---|---|---|---|
| 0 | 0.467 | 0.127 | 1.3 |
| 16 | ~0.36 | ~0.175 | ~12–14 |
| 32 | 0.30–0.39 (DPO/RLHF lower than SFT: 0.30 vs. 0.39) | 0.18–0.19 | 13.5 (SFT) vs. 23–23.5 (DPO/RLHF) |

- Cohen's d stays small (~0.13–0.19) everywhere — confirms the LinearSVC negative finding from a completely different angle: even the *best possible single linear direction* barely separates chosen/rejected.
- Effective rank roughly **doubles** after DPO (13.5 → 23.2) at the final layer, while SFT stays lower — DPO/RLHF use more dimensions in the later layers than SFT, even though this doesn't translate into preference separability.
- Anisotropy drops at later layers after DPO/RLHF (0.39 → 0.30) — representations become less cone-collapsed post-DPO.

## One-paragraph summary for slides

> Across all three Tulu-3-8B checkpoints (SFT, DPO, RLHF) and all 33 layers, chosen/rejected preference is **not linearly separable** in last-token-pooled activations (LinearSVC peak ≈0.55, near chance; Cohen's d stays ≈0.13–0.19 everywhere). Yet the **geometry itself does shift**: representational drift (CKA) shows most of the change happens at the **SFT → DPO** step (CKA drops to 0.976 by the final layer), while DPO → RLHF barely moves the geometry further (CKA ≥0.999). DPO/RLHF also show higher effective rank at later layers than SFT (23 vs. 13.5), and lower anisotropy — the representation reorganizes and spreads out, without becoming more linearly readable for preference.
