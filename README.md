# Representational Geometry in Neural Networks
### *From Vision Transformers to RLHF-Aligned Language Models*

> *"We are not fine-tuning a model — we are performing a geometric audit of what training does to the internal state of a neural network."*

---

## Overview

This project investigates whether a **shared representational structure** emerges when vision and language models are trained on the same task — and how RLHF alignment geometrically transforms language model embedding spaces.

**Part 1 — Cross-modal alignment:** DINOv2 (vision) and Llama-3-8B (language) are both fine-tuned on a multi-class prediction task using matched (image, caption) pairs from MS-COCO. We measure representational alignment via Centered Kernel Alignment (CKA) before and after cross-modal fine-tuning — where each model is regularized toward the other's latent representation. The core question: do models inherently converge to a shared representation, or does cross-modal guidance provide genuinely new information?

**Part 2 — RLHF geometry:** We compare Llama-3-8B (base) with Llama-3-8B-Instruct (RLHF-aligned) layer by layer, using UMAP and LinearSVC to make the geometric effect of alignment literally visible. Future work extends this to RLHF training snapshots — tracking how cross-modal alignment evolves *during* RLHF fine-tuning.

Both parts are surfaced in a single **interactive Plotly Dash dashboard**.

| Part 1 — CKA: Vision vs. Language | Part 2 — UMAP: RLHF Embedding Space |
|:---:|:---:|
| ![CKA Heatmap](assets/slides/cka_heatmap.png) | ![UMAP Scatter](assets/slides/umap_scatter.png) |

---

## Research Questions

| Part | Question | Method |
|------|----------|--------|
| Cross-modal | Does a shared representation emerge when vision and language models train on the same task? | CKA before/after cross-modal fine-tuning |
| Cross-modal | Does fine-tuning with the other model's latent representation as a loss term improve performance? | CKA + task accuracy comparison |
| Cross-modal | How far are our models from explicit cross-modal alignment? | CLIP (ViT-B/32) as upper-bound baseline |
| Language | Does RLHF geometrically separate preferred from rejected responses? | LinearSVC per layer across 32 layers |
| Language | How does this geometric separation evolve layer by layer? | UMAP scatter · layer slider |
| Language | Does RLHF alignment shift language representations toward visual geometry? | Cross-modal CKA: base vs. Instruct vs. CLIP |

---

## Datasets

Datasets are **not included in this repository** and must be downloaded manually before running the extraction scripts.

### MS-COCO (cross-modal, Part 1)

Matched (image, caption) pairs — fetched automatically via the `datasets` library.

### HuggingFace Datasets (auto-downloaded)

- [`Anthropic/hh-rlhf`](https://huggingface.co/datasets/Anthropic/hh-rlhf) — Human preference pairs (chosen vs. rejected responses)
- [`OpenAssistant/oasst1`](https://huggingface.co/datasets/OpenAssistant/oasst1) — Open-source preference dataset

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Vision Model | DINOv2 (ViT-based, self-supervised — Meta AI) |
| Language Model | Llama-3-8B base + Instruct (HuggingFace, 4-bit quantized) |
| Cross-modal Baseline | CLIP ViT-B/32 (OpenAI) — upper bound for cross-modal alignment |
| Cross-modal Dataset | MS-COCO matched (image, caption) pairs |
| Representational Similarity | Linear CKA, LinearSVC, TruncatedSVD |
| Dimensionality Reduction | UMAP, t-SNE |
| Visualization | Plotly Dash (unified two-tab dashboard) |

---

## Dashboard

A single Dash app (`app/app.py`) with two tabs, running on `http://127.0.0.1:8050`.

**Tab 1 — Vision: CKA & Activations**
- CKA heatmap: pairwise layer similarity between vision and language representations
- CAM comparison: Class Activation Maps with class filter dropdown

**Tab 2 — RLHF Embedding Space**
- UMAP / t-SNE scatter of chosen vs. rejected embeddings per layer
- Layer slider + model toggle (base vs. instruct)
- LinearSVC separation score line chart across all layers

### Project Structure

```
app/
  app.py                  # Unified two-tab Dash app
  callbacks.py            # Part 2 callback logic
  vision_callbacks.py     # Part 1 callback logic
  compute.py              # UMAP, t-SNE, LinearSVC fitting + pickle cache
  vision_compute.py       # Linear CKA computation + pickle cache
  crossmodal_compute.py   # Cross-modal CKA (vision vs. language)
  figures.py              # Part 2 Plotly figure builders
  vision_figures.py       # Part 1 Plotly figure builders (CKA heatmap, CAM)
scripts/
  extract_embeddings.py        # Offline: embed hh-rlhf with Llama-3-8B, save HDF5
  extract_vision_embeddings.py # Offline: fine-tune vision model, extract activations
  generate_mock_data.py        # Synthetic LLM embeddings for UI testing
  generate_mock_vision.py      # Synthetic vision activations + CAMs for UI testing
  generate_mock_crossmodal.py  # Synthetic cross-modal embeddings for UI testing
  export_slide_figures.py      # Export static PNGs from mock data for slides
  data.py                      # hh-rlhf sampling
  io.py                        # HDF5 read/write (LLM + vision)
tests/                    # pytest test suite (24 tests)
data/
  embeddings/             # LLM HDF5 files (gitignored)
  vision/                 # Vision HDF5 files (gitignored)
  cache/                  # Projection + CKA cache (gitignored)
assets/
  slides/                 # Exported PNGs for slides (gitignored)
```

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note (macOS):** `bitsandbytes` is Linux/GPU only and not required for the dashboard or mock data. Install without it:
> ```bash
> grep -v bitsandbytes requirements.txt | pip install -r /dev/stdin
> ```

### Option A — Test with mock data (no GPU required)

Generates synthetic data shaped identically to real model output:

```bash
python scripts/generate_mock_data.py
python scripts/generate_mock_vision.py
python app/app.py
```

Open [http://127.0.0.1:8050](http://127.0.0.1:8050).

First launch fits UMAP, t-SNE, LinearSVC, and CKA for all layer combinations (~5–10 min for mock data). Results are cached to `data/cache/` — subsequent launches start in seconds.

### Option B — Prebuilt data (no GPU, real embeddings)

Downloads a prebuilt `data/` archive (all embeddings, cache, CIFAR-10) from Google Drive:

```bash
python scripts/download_data.py
python app/app.py
```

Fetches from a shared Drive link. Override with `--file-id <id>` or `DATA_GDRIVE_ID` env var if the file moves.

### Option C — Real embeddings (requires GPU + HF token)

**Part 1 — Vision:**

```bash
python scripts/extract_vision_embeddings.py --epochs 10 --n-train 10000
```

**Part 2 — Language:**

Models are gated on Hugging Face — request access to `meta-llama/Meta-Llama-3-8B` and `meta-llama/Meta-Llama-3-8B-Instruct` first.

```bash
export HF_TOKEN=<your_token>

python scripts/extract_embeddings.py --model meta-llama/Meta-Llama-3-8B --n-rows 500
python scripts/extract_embeddings.py --model meta-llama/Meta-Llama-3-8B-Instruct --n-rows 500
```

Extraction takes ~1–2 hours per model on an RTX 2080 Super (4-bit quantized, ~5 GB VRAM). See `SETUP.md` for full instructions.

### Run tests

```bash
pytest tests/ -v   # 24 tests
```

---

## Team

| Name | Program |
|------|------|
| Marla Huxhold | M.Sc. Computer Science |
| Sarah Pollinger | M.Sc. Computer Science |
| Ellen Kunigk | M.Sc. Computer Science |
| Kevin Kunkel | M.Sc. Computer Science |
| Abdellah Charki | M.Sc. Data Science |

---

## References

- He, Trott, Khosla (2025). *Shared Latent Representations across Vision and Language.* EMNLP. [arXiv:2509.20751](https://arxiv.org/abs/2509.20751) — **anchor paper**
- Kucukahmetler et al. (2026). *Relative Geometry of Neural Forecasters: Linking Accuracy and Alignment in Learned Latent Geometry.* TMLR. [arXiv:2602.15676](https://arxiv.org/abs/2602.15676)
- Kornblith et al. (2019). *Similarity of Neural Network Representations Revisited.* ICML. [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)
- Ouyang et al. (2022). *Training language models to follow instructions with human feedback.* NeurIPS. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- Christiano et al. (2017). *Deep Reinforcement Learning from Human Preferences.* NeurIPS. [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)
- McInnes et al. (2018). *UMAP: Uniform Manifold Approximation and Projection.* [arXiv:1802.03426](https://arxiv.org/abs/1802.03426)
- Dosovitskiy et al. (2020). *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.* [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

---

*Mathematics & Machine Learning Internship — University of Leipzig, SoSe 2026*
*Supervisor: Dr. Diaaeldin Taha*
