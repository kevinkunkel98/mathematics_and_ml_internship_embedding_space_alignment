#import "../helpers.typ": *

// --------------------------------------------------------------------------
== Deliverable & Honest Limitations

#v(0.3em)
#grid(
  columns: (1fr, 1fr),
  gutter: 0.9em,
  align: top,
  [
    #image("../assets/slides/Drift.png")
    #v(0.1em)
    #text(size: 8.5pt, fill: luma(120))[Interactive Plotly Dash · CKA heatmaps · layer sliders · UMAP/t-SNE · SVC score]
  ],
  [
    #block(fill: navy, inset: (x: 0.7em, y: 0.4em), radius: (top: 3pt), width: 100%)[
      #text(fill: white, weight: "bold", size: 0.8em)[Project deliverable]
    ]
    #block(fill: mint, inset: (x: 0.7em, y: 0.45em), radius: (bottom: 3pt), stroke: (left: 3pt + sage), width: 100%)[
      The dashboard *is* the deliverable, an interactive visualization of all findings: layer slider, CKA heatmaps, UMAP scatter, SVC score
    ]

    #v(0.3em)
    #block(fill: navy, inset: (x: 0.7em, y: 0.4em), radius: (top: 3pt), width: 100%)[
      #text(fill: white, weight: "bold", size: 0.8em)[Limitations & learnings]
    ]
    #block(fill: sky, inset: (x: 0.7em, y: 0.45em), radius: (bottom: 3pt), stroke: (left: 3pt + blue), width: 100%)[
      #text(size: 0.78em)[
        - *Part 1:* float16 overflow in extraction; later vision layers, most `phase_1` language layers came back `inf`
        - *Part 2:* last-token pooling only; paired-difference probes reach ~84% in the literature vs. our ~55% peak
      ]
    ]
  ],
)
