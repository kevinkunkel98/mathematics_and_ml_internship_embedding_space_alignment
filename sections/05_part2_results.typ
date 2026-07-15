#import "../helpers.typ": *

// --------------------------------------------------------------------------
== Part 2 — RLHF Geometry · Real Results

#text(size: 0.85em, fill: luma(90))[Tulu-3-8B SFT → DPO → RLHF · `Anthropic/hh-rlhf` · 4,000 samples · 33 layers]

#v(0.3em)
#grid(
  columns: (1fr, 1fr),
  gutter: 0.9em,
  align: top,
  [
    #text(weight: "bold", fill: navy, size: 1em)[Negative finding]
    #v(0.25em)
    - Not linearly encoded, any stage
    - LinearSVC peak: 0.543 / 0.544 / 0.550
    - Near chance, all 3 checkpoints (layer 14)
    - Cohen's d peak: 0.183 → 0.193 → 0.194
    - No phase transition in preference geometry
  ],
  [
    #text(weight: "bold", fill: navy, size: 1em)[But the geometry does shift]
    #v(0.25em)
    - Linear CKA, same inputs, layer by layer
    - CKA(SFT, DPO) drops to *0.976* by layer 32
    - Most of the shift happens SFT→DPO
    - CKA(DPO, RLHF) stays near *0.999*
    - RLHF/PPO barely moves geometry further
    #v(0.2em)
    #text(size: 0.72em, fill: luma(80))[Full curves in the dashboard · Part 2 tab]
  ],
)

#v(0.3em)
// NOTE: 63% figure per Bai et al. 2022 abstract+recollection — verify against full PDF text
// before final submission (abstract pages don't expose body stats).
#remark[
  Peak accuracy near 0.55 matches published `hh-rlhf` baselines (0.57–0.66) and the dataset's own
  ~63% label-agreement ceiling (Bai et al., 2022).
]
