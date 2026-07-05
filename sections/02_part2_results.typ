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
    #block(fill: navy, inset: (x: 0.7em, y: 0.4em), radius: (top: 3pt), width: 100%)[
      #text(fill: white, weight: "bold", size: 0.82em)[Negative finding]
    ]
    #block(fill: mint, inset: (x: 0.8em, y: 0.5em), radius: (bottom: 3pt), stroke: (left: 3pt + sage), width: 100%)[
      #text(size: 0.85em)[Chosen/rejected preference is not linearly encoded in last-token pooled representations, at any alignment stage]

      #v(0.25em)
      #text(size: 0.8em)[
        - Peak LinearSVC accuracy near chance in all 3 checkpoints — SFT 0.543, DPO 0.544, RLHF 0.550 (layer 14)
        - Peak Cohen's d small and flat — 0.183 → 0.193 → 0.194
        - No phase transition: alignment does not linearly reorganize preference geometry
      ]
    ]
  ],
  [
    #block(fill: navy, inset: (x: 0.7em, y: 0.4em), radius: (top: 3pt), width: 100%)[
      #text(fill: white, weight: "bold", size: 0.82em)[But the geometry does shift]
    ]
    #block(fill: sky, inset: (x: 0.8em, y: 0.5em), radius: (bottom: 3pt), stroke: (left: 3pt + blue), width: 100%)[
      #text(size: 0.85em)[Representational drift (Linear CKA, same inputs, layer by layer):]

      #v(0.25em)
      #text(size: 0.8em)[
        - CKA(SFT, DPO) drops to *0.976* by layer 32 — most of the shift happens here
        - CKA(DPO, RLHF) stays above *0.999* through layer 32 — RLHF/PPO barely moves geometry further
        - Shift is concentrated at SFT→DPO — not a chosen/rejected separator, a general representational shift
      ]

      #v(0.2em)
      #text(size: 0.72em, fill: luma(80))[Full curves (SVC, Cohen's d, CKA drift, anisotropy, effective rank) in the dashboard · Part 2 tab]
    ]
  ],
)
