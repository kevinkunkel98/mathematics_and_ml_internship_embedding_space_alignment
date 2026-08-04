#import "../helpers.typ": *

// --------------------------------------------------------------------------
== References: Part 2 (RLHF Geometry)

#set text(size: 12pt)
#v(0.3em)
#grid(
  columns: (1fr, 1fr),
  gutter: (0.6em, 0.4em),
  align: top,
  [
    #text(weight: "bold", fill: navy)[Bai et al. (2022)] \
    _Training a Helpful and Harmless Assistant with RLHF_ \
    #text(fill: luma(100))[arXiv:2204.05862 · *Original `hh-rlhf` dataset paper*]
  ],
  [
    #text(weight: "bold", fill: navy)[Lee, Bai, Pres, Wattenberg, Kummerfeld, Mihalcea (2024)] \
    _A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity_ \
    #text(fill: luma(100))[ICML · arXiv:2401.01967 · *DPO learns an "offset", not erasure*]
  ],
  [
    #text(weight: "bold", fill: navy)[Sinha, Garg, Elluru, Singh, Garg (2026)] \
    _Mechanistic Analysis of Alignment Algorithms in Language Models_ \
    #text(fill: luma(100))[arXiv:2606.09850 · *Objective-dependent separability · re-check*]
  ],
  [
    #text(weight: "bold", fill: navy)[Ziegler et al. (2019)] \
    _Fine-Tuning Language Models from Human Preferences_ \
    #text(fill: luma(100))[arXiv:1909.08593 · *Labeler agreement baselines*]
  ],
  [
    #text(weight: "bold", fill: navy)[Stiennon et al. (2020)] \
    _Learning to Summarize from Human Feedback_ \
    #text(fill: luma(100))[NeurIPS · arXiv:2009.01325 · *77%±2% agreement, contrast to `hh-rlhf`*]
  ],
  [
    #text(weight: "bold", fill: navy)[Kirin (2026)] \
    _Relational Preference Encoding in Looped Transformer Internal States_ \
    #text(fill: luma(100))[arXiv:2604.09870 · *Paired-diff probes reach ~84.5% · re-check*]
  ],
  [
    #text(weight: "bold", fill: navy)[Zhao, Gong, Chen, Kang, Li (2026)] \
    _A Layer-wise Analysis of Supervised Fine-Tuning_ \
    #text(fill: luma(100))[arXiv:2604.11838 · *Final layers most SFT-sensitive · re-check*]
  ],
  [
    #text(weight: "bold", fill: navy)[Kumar et al. (2026)] \
    _When Synthetic Speech Is All You Have: Better Call GRPO_ \
    #text(fill: luma(100))[arXiv:2607.08409 · *GRPO changes behavior, not geometry · re-check*]
  ],
)
