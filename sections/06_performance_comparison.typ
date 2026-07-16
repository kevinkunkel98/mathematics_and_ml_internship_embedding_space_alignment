#import "../helpers.typ": *

// --------------------------------------------------------------------------
== Part 1 — Model Performance Comparison

#text(size: 0.85em, fill: luma(90))[Test dataset: Validation 2014 · COCO multi-label · macro mAP]

#v(0.4em)
#table(
  columns: (1.1fr, 1.5fr, 1fr, 1fr),
  stroke: 0.4pt + luma(200),
  inset: 8pt,
  align: (left, left, center, center),
  fill: (x, y) => if y == 0 { sky } else { white },
  [*Model*], [*Phase*], [*\# Training epochs*], [*Test mAP*],
  [Language], [Phase 1], [29], [0.2044],
  [Language], [Phase 2 #text(size: 0.8em, fill: sage)[(+ CKA)]], [30], [#text(fill: sage, weight: "bold")[0.2606]],
  [Vision], [Phase 1 (Teacher)], [8], [0.0845],
)

#v(0.5em)
#insight("CKA alignment does not cost task performance")[
  #text(size: 0.85em)[
    Phase 2 (CKA-aligned) reaches the highest mAP (*0.2606*), above the
    Phase-1 language baseline (0.2044) — the geometry alignment did not
    degrade multi-label accuracy.
  ]
]
