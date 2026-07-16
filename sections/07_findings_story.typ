#import "../helpers.typ": *

// --------------------------------------------------------------------------
== Part 2 — What the Geometry Tells Us

#v(0.15em)
#text(weight: "bold", fill: navy, size: 1em)[No readable linear "preference direction"]
#v(0.15em)
- Flat, near-chance at every layer, all 3 checkpoints
- DPO/PPO-style objectives preserve or degrade separability, not universal to "RLHF"

#v(0.2em)
#text(weight: "bold", fill: navy, size: 1em)[But geometry is not static]
#v(0.15em)
- Real reorganization, concentrated at *SFT → DPO*
- Most "alignment work" happens before PPO

#v(0.15em)
#text(size: 0.8em, style: "italic", fill: navy)[
  Alignment reshapes geometry *globally*, not a single axis.
]
