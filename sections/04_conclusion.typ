#import "../helpers.typ": *

// --------------------------------------------------------------------------
== Deliverable & Next Steps

#v(0.3em)
#grid(
  columns: (1fr, 1fr),
  gutter: 0.9em,
  align: top,
  [
    #image("../assets/slides/dashboard_vit_umap.png")
    #v(0.1em)
    #text(size: 8.5pt, fill: luma(120))[Interactive Plotly Dash · CKA heatmap · UMAP layer slider · LinearSVC score]
  ],
  [
    #block(fill: navy, inset: (x: 0.7em, y: 0.4em), radius: (top: 3pt), width: 100%)[
      #text(fill: white, weight: "bold", size: 0.8em)[Project deliverable]
    ]
    #block(fill: mint, inset: (x: 0.7em, y: 0.45em), radius: (bottom: 3pt), stroke: (left: 3pt + sage), width: 100%)[
      The dashboard *is* the deliverable — interactive visualization of all findings: layer slider, CKA heatmaps, UMAP scatter, SVC score
    ]

    #v(0.3em)
    #block(fill: navy, inset: (x: 0.7em, y: 0.4em), radius: (top: 3pt), width: 100%)[
      #text(fill: white, weight: "bold", size: 0.8em)[Perspective]
    ]
    #block(fill: sky, inset: (x: 0.7em, y: 0.45em), radius: (bottom: 3pt), stroke: (left: 3pt + blue), width: 100%)[
      Extend to RLHF training *snapshots* — plot cross-modal CKA vs. training step to see if alignment shifts *during* RLHF
    ]
  ],
)
