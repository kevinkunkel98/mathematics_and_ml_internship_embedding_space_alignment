# Representational Geometry in Neural Networks
### *From Vision Transformers to RLHF-Aligned Language Models*

> *"We are not fine-tuning a model — we are performing a geometric audit of what training does to the internal state of a neural network."*

---

## Overview

This project studies how training transforms the internal representation spaces of neural networks — across two complementary settings:

**Part 1 — Vision Models:** We compare the representational geometry of CNNs and Vision Transformers on image classification benchmarks (CIFAR-10), using Centered Kernel Alignment (CKA) to quantify how similarly the two architectures encode visual information, and Class Activation Maps to identify which image regions drive cluster formation.

**Part 2 — Language Models:** We investigate how Reinforcement Learning from Human Feedback (RLHF) geometrically transforms the embedding space of large language models, comparing Llama-3-8B (base) with Llama-3-8B-Instruct (RLHF-aligned) to make the alignment transformation literally visible and explorable.

Both parts aim to make the blackbox that are embedding representations more explainable.

---

## Research Questions

| Part | Question | Method |
|------|----------|--------|
| Vision | How similar are the internal representations of CNNs vs. ViTs, and which image regions drive cluster formation? | CKA + Class Activation Maps |
| Language | Can a linear hyperplane separate preferred from rejected response vectors after RLHF, and how does this evolve across layers? | LinearSVC + Margin Analysis |
| Language | Can political or cultural biases introduced by RLHF be identified as geometric structures in the embedding space? | Bias Probes + Cluster Analysis |

---

## Datasets

- **CIFAR-10** — Standard image classification benchmark
- [`Anthropic/hh-rlhf`](https://huggingface.co/datasets/Anthropic/hh-rlhf) — Human preference pairs (chosen vs. rejected responses)
- [`OpenAssistant/oasst1`](https://huggingface.co/datasets/OpenAssistant/oasst1) — Open-source preference dataset

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Vision Models | CNN + ViT (PyTorch) |
| Language Models | HuggingFace Transformers (Llama-3-8B) |
| Representational Similarity | CKA, LinearSVC, TruncatedSVD |
| Dimensionality Reduction | UMAP, t-SNE |
| Visualization | Plotly Dash |

---

## RLHF Embedding Visualizer (Part 2)

An interactive Plotly Dash dashboard that visualizes how RLHF transforms the embedding geometry of Llama-3-8B across all 33 layers. Features a layer slider, UMAP/t-SNE toggle, model toggle (base vs. instruct), and a LinearSVC separation score line chart.

### Project Structure

```
app/
  app.py            # Dash app — loads HDF5, fits projections, serves dashboard
  callbacks.py      # Dash callback logic
  compute.py        # UMAP, t-SNE, LinearSVC fitting + pickle cache
  figures.py        # Plotly figure builders
scripts/
  extract_embeddings.py   # Offline: load models, embed hh-rlhf, save HDF5
  generate_mock_data.py   # Generate synthetic embeddings for UI testing
  data.py                 # hh-rlhf sampling
  io.py                   # HDF5 read/write
tests/                    # pytest test suite (12 tests)
data/
  embeddings/             # HDF5 files (gitignored — generate or extract)
  cache/                  # Projection cache (gitignored)
```

### Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Option A — Test with mock data (no GPU required)

Generates synthetic embeddings shaped identically to the real model output, with a built-in separation signal so the dashboard behaves realistically:

```bash
python scripts/generate_mock_data.py
python app/app.py
```

Open [http://127.0.0.1:8050](http://127.0.0.1:8050).

First launch fits UMAP, t-SNE, and LinearSVC for all 66 layer × model combinations (~5–10 min for mock data). Results are cached to `data/cache/` — subsequent launches start in seconds.

### Option B — Real embeddings (requires GPU + HF token)

Models: `meta-llama/Meta-Llama-3-8B` and `meta-llama/Meta-Llama-3-8B-Instruct` (gated — request access on Hugging Face first).

```bash
export HF_TOKEN=<your_token>

python scripts/extract_embeddings.py --model meta-llama/Meta-Llama-3-8B --n-rows 500
python scripts/extract_embeddings.py --model meta-llama/Meta-Llama-3-8B-Instruct --n-rows 500

python app/app.py
```

Extraction takes ~30–60 min per model on a single A100. Reduce `--batch-size` (default 8) if you hit OOM.

### Run tests

```bash
pytest tests/ -v
```

---

## Team

| Name | Program |
|------|------|
| Marla Huxhold | M.Sc. Data Science |
| Sarah Pollinger | M.Sc. Data Science |
| Ellen Kunigk | M.Sc. Computer Science |
| Kevin Kunkel | M.Sc. Computer Science |
| Abdellah Charki | M.Sc. Data Science |

---

## References

- Kucukahmetler et al. (2026). *Relative Geometry of Neural Forecasters: Linking Accuracy and Alignment in Learned Latent Geometry.* [arXiv:2602.15676](https://arxiv.org/abs/2602.15676)
- Kornblith et al. (2019). *Similarity of Neural Network Representations Revisited.* ICML. [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)
- Ouyang et al. (2022). *Training language models to follow instructions with human feedback.* NeurIPS. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- Christiano et al. (2017). *Deep Reinforcement Learning from Human Preferences.* NeurIPS. [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)
- McInnes et al. (2018). *UMAP: Uniform Manifold Approximation and Projection.* [arXiv:1802.03426](https://arxiv.org/abs/1802.03426)
- [Anthropic HH-RLHF Dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [awesome-rlhf](https://github.com/wassname/awesome-rlhf)
---

*Mathematics & Machine Learning Internship — University of Leipzig, SoSe 2026*
