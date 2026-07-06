#import "../helpers.typ": *

// --------------------------------------------------------------------------
== Part 2 — What the Geometry Tells Us

#v(0.3em)
#insight([RLHF does not carve a linear "preference direction"], [
  We asked: does alignment training create a direction in the residual stream
  that separates chosen from rejected? LinearSVC and Cohen's d stay flat and
  near-chance at *every* layer, in *all three* checkpoints — SFT, DPO, RLHF.
])

#v(0.35em)
#insight([But the representation space is not static], [
  Linear CKA shows real structural reorganization — concentrated almost
  entirely at *SFT → DPO* (CKA drops to 0.976 by layer 32). The *DPO → RLHF*
  step barely moves the geometry further (CKA stays above 0.999) — most of
  the "alignment work" already happened before PPO.
])

#v(0.35em)
#remark[
  Anisotropy and effective rank both shift in that same SFT→DPO window —
  consistent with one genuine representational transition, not three
  separate ones. It just isn't a transition any single linear probe can read.
]

#v(0.4em)
#text(size: 0.85em, style: "italic", fill: navy)[
  Alignment reshapes geometry *globally*, not by writing a single readable
  "preference axis" — evidence for treating RLHF's effect as a geometric
  audit, not a steering vector to extract.
]
