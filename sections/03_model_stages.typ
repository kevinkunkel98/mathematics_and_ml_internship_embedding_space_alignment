#import "../helpers.typ": *

// --------------------------------------------------------------------------
== Part 2 — What Each Checkpoint Actually Is

#v(0.2em)
#text(size: 0.85em, fill: luma(90))[Tulu-3-8B alignment trajectory — 3 public AllenAI checkpoints, same base model, extracted at each post-training stage]

#v(0.4em)
#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 0.7em,
  definition([SFT], [
    #text(size: 0.8em)[
      `Llama-3.1-Tulu-3-8B-SFT` \
      Supervised fine-tuning on curated instruction/response demonstrations. Model learns to follow instructions — no preference signal yet.
    ]
  ]),
  definition([DPO], [
    #text(size: 0.8em)[
      `Llama-3.1-Tulu-3-8B-DPO` \
      Direct Preference Optimization on chosen/rejected pairs. Directly increases $p("chosen")$ over $p("rejected")$ — no explicit reward model.
    ]
  ]),
  definition([RLHF], [
    #text(size: 0.8em)[
      `Llama-3.1-Tulu-3-8B` \
      Final RL stage on top of DPO (Tulu-3's RLVR — verifiable rewards on math/precise tasks). Labeled *RLHF* here for consistency with Part 1.
    ]
  ]),
)

#v(0.5em)
#remark[
  All 3 are the *same* underlying 8B model at 3 successive training stages — not 3 different architectures. This is what makes the SFT → DPO → RLHF comparison a clean trajectory rather than a cross-model comparison.
]
