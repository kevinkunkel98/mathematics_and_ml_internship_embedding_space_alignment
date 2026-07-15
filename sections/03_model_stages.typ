#import "../helpers.typ": *

// --------------------------------------------------------------------------
== Part 2 — What Each Checkpoint Actually Is

#v(0.2em)
#text(size: 0.85em, fill: luma(90))[Tulu-3-8B alignment trajectory — 3 public AllenAI checkpoints, same base model, extracted at each post-training stage]

#v(0.4em)
#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 0.7em,
  [
    #text(weight: "bold", fill: sage, size: 1.1em)[SFT]
    #v(0.2em)
    #text(size: 0.72em)[`Llama-3.1-Tulu-3-8B-SFT`]
    #v(0.2em)
    - Instruction fine-tuning
    - No preference signal yet
  ],
  [
    #text(weight: "bold", fill: sage, size: 1.1em)[DPO]
    #v(0.2em)
    #text(size: 0.72em)[`Llama-3.1-Tulu-3-8B-DPO`]
    #v(0.2em)
    - Chosen > rejected, directly
    - No reward model
  ],
  [
    #text(weight: "bold", fill: sage, size: 1.1em)[RLHF]
    #v(0.2em)
    #text(size: 0.72em)[`Llama-3.1-Tulu-3-8B`]
    #v(0.2em)
    - Final RL stage (Tulu-3's RLVR)
    - Labeled *RLHF* for Part 1 consistency
  ],
)

#v(0.5em)
#remark[
  All 3 are the *same* underlying 8B model at 3 successive training stages — not 3 different architectures. This is what makes the SFT → DPO → RLHF comparison a clean trajectory rather than a cross-model comparison.
]
