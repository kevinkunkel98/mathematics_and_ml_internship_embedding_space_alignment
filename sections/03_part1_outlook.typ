#import "../helpers.typ": *

// --------------------------------------------------------------------------
== Part 1 — Cross-modal Alignment · Outlook

#v(0.2em)
#grid(
  columns: (1fr, 1fr),
  gutter: 0.9em,
  align: top,
  [
    #block(fill: navy, inset: (x: 0.7em, y: 0.4em), radius: (top: 3pt), width: 100%)[
      #text(fill: white, weight: "bold", size: 0.82em)[Setup · DINOv2 + Llama · MS-COCO · multi-class prediction]
    ]
    #block(fill: sky, inset: (x: 0.8em, y: 0.5em), radius: (bottom: 3pt), stroke: (left: 3pt + blue), width: 100%)[
      Both models trained on *same task* (object category prediction) · measure CKA · fine-tune with loss term encouraging similarity to the other model's representation · compare

      #v(0.25em)
      *CKA high before fine-tuning:* models converge inherently *(Platonic Representation Hypothesis)*

      *CKA rises after fine-tuning:* cross-modal signal is genuinely new information

      #v(0.2em)
      #text(size: 0.78em, fill: luma(60))[Layer-wise CKA map shows *where* shared meaning lives · UMAP + dashboard to visualize]
    ]
  ],
  [
    #image("../assets/slides/dashboard_cka.png")
    #v(0.1em)
    #text(size: 8.5pt, fill: luma(120))[Interactive Plotly Dash · CKA heatmap]

    #v(0.3em)
    #block(fill: navy, inset: (x: 0.7em, y: 0.4em), radius: (top: 3pt), width: 100%)[
      #text(fill: white, weight: "bold", size: 0.8em)[Next steps — toward final]
    ]
    #block(fill: sky, inset: (x: 0.7em, y: 0.45em), radius: (bottom: 3pt), stroke: (left: 3pt + blue), width: 100%)[
      - Cross-modal CKA: DINOv2 vs. Llama on MS-COCO pairs
      - *CLIP (ViT-B/32)* as upper bound — explicitly trained cross-modal alignment
      - Compare base vs. Instruct CKA scores
    ]
  ],
)
