#import "../helpers.typ": *
#import "@preview/cetz:0.3.4": canvas, draw

// --------------------------------------------------------------------------
== Part 1 · Setup: Two Independent Baselines on MS-COCO

#v(1fr)
#align(center)[
  #canvas(length: 1.9cm, {
    import draw: *

    let box(x, y, title, subtitle: none, w: 4.2, h: 1.8, fill: sky, stroke-color: blue) = {
      rect((x, y - h / 2), (x + w, y + h / 2), fill: fill, stroke: 1.2pt + stroke-color, radius: 5pt)
      content((x + w / 2, y + (if subtitle != none { 0.2 } else { 0 })), text(fill: stroke-color, size: 0.42cm, weight: "bold")[#title])
      if subtitle != none {
        content((x + w / 2, y - 0.45), text(fill: luma(70), size: 0.32cm)[#subtitle])
      }
    }

    let arrow(..pts) = line(..pts, stroke: (paint: navy, thickness: 2.2pt), mark: (end: ">", size: 0.36))

    // Row 1: Vision
    let y1 = 1.6
    box(-6.5, y1, [DINOv2], subtitle: [pretrained vision], fill: mint, stroke-color: sage, w: 3.2)
    arrow((-3.3, y1), (-2.5, y1))
    content((-2.9, y1 + 0.6), text(fill: navy, size: 0.34cm)[images])
    box(-2.5, y1, [Coco Vision Baseline], subtitle: [→ category logits], w: 4.6)

    // Row 2: Language
    let y2 = -1.6
    box(-6.5, y2, [Llama], subtitle: [pretrained language], fill: mint, stroke-color: sage, w: 3.2)
    arrow((-3.3, y2), (-2.5, y2))
    content((-2.9, y2 + 0.6), text(fill: navy, size: 0.34cm)[captions])
    box(-2.5, y2, [Coco Language Baseline], subtitle: [→ category logits], w: 4.6)
  })
]
#v(1fr)

// --------------------------------------------------------------------------
== Part 1 · Example: MS-COCO 2017

#v(0.3em)
#grid(
  columns: (1fr, 2fr),
  gutter: 1.2em,
  align: top,
  [
    #text(size: 0.85em, fill: luma(90))[Categories: `person`, `tie`, `umbrella`, `handbag`]

    #v(0.3em)
    #text(size: 0.85em, fill: luma(90))[Captions:]
    #text(size: 0.8em)[
      - A girl in grey jacket and tie standing on a street.
      - Little girl posing for the camera with an adult sized striped tie on.
      - A kid with a tie standing in the street.
      - A young girl wearing a necktie and smiling broadly.
      - Girl smiles for picture in busy Asian plaza.
    ]
  ],
  [
    #image("../assets/slides/mscoco.png", width: 100%)
  ],
)

// --------------------------------------------------------------------------
== Part 1 · Method: Teacher → Student CKA Fine-tuning

#v(1fr)
#align(center)[
  #canvas(length: 1.9cm, {
    import draw: *

    let box(x, y, title, subtitle: none, w: 3.6, h: 1.8, fill: sky, stroke-color: blue) = {
      rect((x, y - h / 2), (x + w, y + h / 2), fill: fill, stroke: 1.2pt + stroke-color, radius: 5pt)
      content((x + w / 2, y + (if subtitle != none { 0.2 } else { 0 })), text(fill: stroke-color, size: 0.4cm, weight: "bold")[#title])
      if subtitle != none {
        content((x + w / 2, y - 0.45), text(fill: luma(70), size: 0.3cm)[#subtitle])
      }
    }

    let arrow(..pts) = line(..pts, stroke: (paint: navy, thickness: 2.2pt), mark: (end: ">", size: 0.36))

    // Row 1: Vision = Teacher
    let y1 = 1.7
    box(-7.6, y1, [Vision], subtitle: [Teacher], fill: mint, stroke-color: sage, w: 2.6)
    arrow((-5.0, y1), (-4.3, y1))
    box(-4.3, y1, [Coco Vision], subtitle: [Categories], w: 3.6)
    line((-0.7, y1 - 0.65), (1.6, -0.5), stroke: (paint: blue.lighten(20%), thickness: 1.8pt), mark: (end: ">", size: 0.36))
    content((0.9, y1 - 0.25), text(fill: blue.lighten(10%), size: 0.36cm, weight: "bold")[\+ CKA])

    // Row 2: Language = Student
    let y2 = -1.7
    box(-7.6, y2, [Language], subtitle: [Student], fill: mint, stroke-color: sage, w: 2.6)
    arrow((-5.0, y2), (-4.3, y2))
    content((-4.65, y2 + 0.75), text(fill: navy, size: 0.3cm)[Phase 1])
    box(-4.3, y2, [Coco Language], subtitle: [Categories], w: 3.6)
    arrow((-0.7, y2), (1.6, y2))
    content((0.45, y2 + 0.65), text(fill: navy, size: 0.3cm)[Combined loss])

    box(1.6, 0, [Phase 2], subtitle: [Categories], w: 3.2, h: 3.0)
  })
]
#v(1fr)
