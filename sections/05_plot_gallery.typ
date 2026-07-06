#import "../helpers.typ": *

#let plot-slide(title, path, caption) = [
  == Part 2 — #title

  #v(0.3em)
  #align(center)[
    #image(path, width: 88%)
    #v(0.4em)
    #text(size: 0.85em, fill: luma(80))[#caption]
  ]
]

// --------------------------------------------------------------------------
#plot-slide([CKA Drift], "../assets/slides/Drift.png", [CKA drift — shift concentrated SFT to DPO, later layers])

#plot-slide([Cohen's d], "../assets/slides/Cohensd.png", [Cohen's d — small, flat across all layers])

#plot-slide([Anisotropy], "../assets/slides/Anisotropy.png", [Anisotropy — dips mid-stack, rises again late])

#plot-slide([Effective Rank], "../assets/slides/Rank.png", [Effective rank — dimensionality usage per layer])

#plot-slide([LinearSVC Accuracy], "../assets/slides/SVC.png", [LinearSVC accuracy — pinned near chance (0.5)])

#plot-slide([t-SNE Layer 32], "../assets/slides/RLHF.png", [t-SNE, layer 32 — chosen/rejected heavily overlap])
