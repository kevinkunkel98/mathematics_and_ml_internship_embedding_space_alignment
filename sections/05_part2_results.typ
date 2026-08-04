#import "../helpers.typ": *

// --------------------------------------------------------------------------
== Part 2: RLHF Geometry · Real Results

#text(size: 0.85em, fill: luma(90))[Tulu-3-8B SFT → DPO → RLHF · `Anthropic/hh-rlhf` · 4,000 samples · 33 layers]

#v(0.3em)
#grid(
  columns: (1fr, 1fr),
  gutter: 0.9em,
  align: top,
  [
    #text(weight: "bold", fill: navy, size: 1em)[Negative finding]
    #v(0.25em)
    - Preference not linearly encoded, any stage
    - LinearSVC peak: 0.543 / 0.544 / 0.550, near chance
    - No phase transition in preference geometry
  ],
  [
    #text(weight: "bold", fill: navy, size: 1em)[But the geometry does shift]
    #v(0.25em)
    - CKA(SFT, DPO) drops to *0.976* by layer 32
    - CKA(DPO, RLHF) stays near *0.999*
    - Most of the shift happens SFT → DPO
  ],
)
