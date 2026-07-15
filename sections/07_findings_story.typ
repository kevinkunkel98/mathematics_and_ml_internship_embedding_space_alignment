#import "../helpers.typ": *

// --------------------------------------------------------------------------
== Part 2 — What the Geometry Tells Us

#v(0.15em)
#text(weight: "bold", fill: navy, size: 1em)[Insight: no readable linear "preference direction"]
#v(0.15em)
- Flat, near-chance at *every* layer — all 3 checkpoints
- Peak SVC: 0.543 / 0.544 / 0.550
- DPO/PPO preserve or degrade separability; KTO/GRPO construct it
- Looks objective-specific, not universal to "RLHF"

#v(0.2em)
#text(weight: "bold", fill: navy, size: 1em)[Insight: but geometry is not static]
#v(0.15em)
- Real structural reorganization, concentrated at *SFT → DPO*
- CKA(SFT, DPO) drops to 0.976; CKA(DPO, RLHF) stays near 0.999
- Most "alignment work" happens before PPO

#v(0.15em)
#remark[
  #text(size: 0.9em)[Anisotropy and effective rank shift in that same window — one transition, not three.]
]

#v(0.15em)
#text(size: 0.75em, style: "italic", fill: navy)[
  Alignment reshapes geometry *globally*, not a single axis. Lee et al. (2024): DPO *bypasses*
  a capability rather than erasing it.
]
