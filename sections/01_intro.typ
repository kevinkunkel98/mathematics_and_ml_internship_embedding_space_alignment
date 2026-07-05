#import "../helpers.typ": *

// --------------------------------------------------------------------------
== Core Questions

#v(0.3em)
#grid(
  columns: (1fr, 1fr),
  gutter: 1.0em,
  align: top,
  [
    #block(fill: navy, inset: (x: 0.8em, y: 0.55em), radius: (top: 3pt))[
      #text(fill: white, weight: "bold", size: 0.9em)[Part 1 — Cross-modal alignment]
    ]
    #block(fill: sky, inset: (x: 0.8em, y: 0.55em), radius: (bottom: 3pt), stroke: (left: 3pt + blue), width: 100%)[
      When a vision model and a language model train on the *same task*, does a shared representational structure emerge across modalities?

      #v(0.3em)
      - *Hypothesis A:* cross-modal signal is new information → improves model
      - *Hypothesis B:* models converge regardless — extra signal adds little

      #v(0.3em)
      #text(size: 0.8em, fill: luma(80))[CKA before/after cross-modal fine-tuning · DINOv2 + Llama · MS-COCO]
    ]
  ],
  [
    #block(fill: navy, inset: (x: 0.8em, y: 0.45em), radius: (top: 3pt))[
      #text(fill: white, weight: "bold", size: 0.85em)[Part 2 — RLHF geometry]
    ]
    #block(fill: mint, inset: (x: 0.8em, y: 0.55em), radius: (bottom: 3pt), stroke: (left: 3pt + sage), width: 100%)[
      Does RLHF geometrically transform a language model's embedding space — encoding human preference as a *linearly separable structure*?

      #v(0.3em)
      - Does alignment progressively separate chosen/rejected responses across SFT → DPO → RLHF?
      - How does separation evolve *layer by layer*?

      #v(0.3em)
      #text(size: 0.8em, fill: luma(80))[UMAP + LinearSVC per layer · Tulu-3-8B SFT/DPO/RLHF · `Anthropic/hh-rlhf`]
    ]
  ],
)
