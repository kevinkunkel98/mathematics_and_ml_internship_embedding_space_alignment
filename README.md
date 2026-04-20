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


