#import "../helpers.typ": *

// --------------------------------------------------------------------------
== Part 2 · Motivation: Why RLHF?

#v(0.3em)
#text(size: 0.9em)[Pretraining and SFT optimize *likelihood*, not *preference*.]

#v(0.5em)
#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  align: top,
  [
    #block(fill: navy, inset: (x: 0.8em, y: 0.5em), radius: (top: 3pt), width: 100%)[
      #text(fill: white, weight: "bold", size: 0.9em)[Without RLHF]
    ]
    #block(fill: sky, inset: (x: 0.8em, y: 0.6em), radius: (bottom: 3pt), stroke: (left: 3pt + blue), width: 100%)[
      - Next-token prediction rewards *plausible* text
      - No signal for helpful / harmless / honest
      - A great answer and a subtly wrong one can look equally likely
    ]
  ],
  [
    #block(fill: navy, inset: (x: 0.8em, y: 0.5em), radius: (top: 3pt), width: 100%)[
      #text(fill: white, weight: "bold", size: 0.9em)[With RLHF]
    ]
    #block(fill: mint, inset: (x: 0.8em, y: 0.6em), radius: (bottom: 3pt), stroke: (left: 3pt + sage), width: 100%)[
      - Optimizes directly for *human preference*
      - Signal comes from comparisons, not a fixed label set
      - Model improves on axes humans care about, not just fluency
    ]
  ],
)

#v(0.6em)
#insight([Goal], [
  Shift probability mass toward responses humans rank higher, without collapsing fluency or drifting into reward hacking.
])

// --------------------------------------------------------------------------
== Part 2 · Three Steps of RLHF

#v(0.15em)
#text(size: 0.8em, fill: luma(90))[Classic formulation (Ouyang et al., 2022 · InstructGPT): the paradigm Part 2 measures the geometric effect of]

#v(0.3em)
#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 0.8em,
  align: top,
  [
    #text(weight: "bold", fill: sage, size: 1.05em)[1. Collect preference data]
    #v(0.2em)
    #text(size: 0.85em)[
      - Sample *multiple* completions per prompt from the SFT model
      - Humans compare pairs → chosen $y_w$, rejected $y_l$
      - Output: dataset of $(x, y_w, y_l)$ triples
    ]
  ],
  [
    #text(weight: "bold", fill: sage, size: 1.05em)[2. Train a reward model]
    #v(0.2em)
    #text(size: 0.85em)[
      - $r_phi (x,y)$ learns to *score* a completion
      - Bradley–Terry pairwise loss:
      #text(size: 0.68em)[$ cal(L)(phi) = -EE[log sigma(r_phi (x,y_w) - r_phi (x,y_l))] $]
      - Learns to rank, not to generate
    ]
  ],
  [
    #text(weight: "bold", fill: sage, size: 1.05em)[3. Policy optimization]
    #v(0.2em)
    #text(size: 0.85em)[
      - $pi_theta$ generates, $r_phi$ scores, PPO updates $theta$
      #text(size: 0.68em)[$ max_theta EE_(y tilde pi_theta) [r_phi (x,y)] - beta "KL"(pi_theta || pi_"ref") $]
      - KL term stops *reward hacking*
    ]
  ],
)

#v(0.4em)
#remark[
  #text(size: 0.85em)[
    Neither of our checkpoints follows this exactly: DPO skips the reward model entirely, and our "RLHF" checkpoint is actually *RLVR* (rule-based, verifiable rewards, no learned reward model), kept as "RLHF" only for consistency with Part 1's vocabulary.
  ]
]
