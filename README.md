# Representational Geometry in Neural Networks
### *From Vision Transformers to RLHF-Aligned Language Models*

> *"We are not fine-tuning a model — we are performing a geometric audit of what training does to the internal state of a neural network."*

---

## Overview

This project studies how training transforms the internal representation spaces of neural networks — across two complementary settings:

**Part 1 — Vision Models:** We compare the representational geometry of CNNs and Vision Transformers on image classification benchmarks (CIFAR-10), using Centered Kernel Alignment (CKA) to quantify how similarly the two architectures encode visual information, and Class Activation Maps to identify which image regions drive cluster formation.

**Part 2 — Language Models:** We investigate how Reinforcement Learning from Human Feedback (RLHF) geometrically transforms the embedding space of large language models, comparing Llama-3-8B (base) with Llama-3-8B-Instruct (RLHF-aligned) to make the alignment transformation literally visible and explorable.

Both parts share a common methodology: dimensionality reduction (UMAP/t-SNE), geometric analysis, and interactive visualization.

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

## Team

| Name | Role |
|------|------|
| [Marla Huxhold] | |
| [Sarah Pollinger] | |
| [Ellen Kunigk] | |
| [Kevin Kunkel] | |
| [Abdellah Charki] | |

---

## References

- Kornblith et al. (2019). *Similarity of Neural Network Representations Revisited.* ICML.
- Ouyang et al. (2022). *Training language models to follow instructions with human feedback.* NeurIPS.
- McInnes et al. (2018). *UMAP: Uniform Manifold Approximation and Projection.*
- [Anthropic HH-RLHF Dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf)

---

*Mathematics & Machine Learning Internship — University of Leipzig, SoSe 2026*