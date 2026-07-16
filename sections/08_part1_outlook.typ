#import "../helpers.typ": *

// --------------------------------------------------------------------------
== Part 1 — Cross-modal Alignment · Real Results

#v(0.2em)
#grid(
  columns: (1fr, 1fr),
  gutter: 0.9em,
  align: top,
  [
    #block(fill: navy, inset: (x: 0.7em, y: 0.4em), radius: (top: 3pt), width: 100%)[
      #text(fill: white, weight: "bold", size: 0.82em)[Alignment is built by training, not inherited]
    ]
    #block(fill: sky, inset: (x: 0.8em, y: 0.5em), radius: (bottom: 3pt), stroke: (left: 3pt + blue), width: 100%)[
      #text(size: 0.85em)[
        *Baseline (phase_1):* near-zero everywhere — mean CKA 0.005, max 0.007. No depth structure at all.

        *After CKA fine-tuning:* mean 0.048, max 0.176 — roughly *25×* higher

        #v(0.2em)
        Peak stays at the *same* layer pair both times (vision layer 16 × language layer 28) — training *amplifies* an existing affinity, doesn't create one elsewhere

        #v(0.2em)
        #text(size: 0.9em, fill: luma(60))[Contradicts naive Platonic Representation Hypothesis reading: untrained models do *not* already converge here]
      ]
    ]
  ],
  [
    #image("../assets/slides/part1_supercategory.png")
    #v(0.1em)
    #text(size: 8.5pt, fill: luma(120))[Real dashboard export · CKA per COCO supercategory, phase_1 vs. full_train_set]

    #v(0.3em)
    #block(fill: navy, inset: (x: 0.7em, y: 0.4em), radius: (top: 3pt), width: 100%)[
      #text(fill: white, weight: "bold", size: 0.8em)[By COCO supercategory]
    ]
    #block(fill: sky, inset: (x: 0.7em, y: 0.45em), radius: (bottom: 3pt), stroke: (left: 3pt + blue), width: 100%)[
      #text(size: 0.85em)[
        - 11/12 supercategories gained alignment; `outdoor` flat, none regressed
        - Biggest movers: *vehicle* (+0.21), *person* (+0.13)
      ]
    ]
  ],
)
