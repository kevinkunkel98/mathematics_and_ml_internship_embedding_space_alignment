// ── Palette ───────────────────────────────────────────────────────────────────
#let navy = rgb("#1c3a5e")
#let blue = rgb("#1a4f8a")
#let sky = rgb("#dce9f7")
#let sage = rgb("#1e6b3c")
#let mint = rgb("#e4f4ec")
#let sand = rgb("#f7f4ef")

// ── Box helpers ───────────────────────────────────────────────────────────────
#let thm-box(title, body, fill: sky, stroke-color: blue) = block(
  width: 100%,
  inset: (x: 0.9em, y: 0.65em),
  radius: 3pt,
  fill: fill,
  stroke: (left: 3pt + stroke-color),
)[
  #text(weight: "bold", fill: stroke-color, size: 0.9em)[#title]
  #h(0.4em)
  #body
]

#let insight(title, body) = thm-box([Insight: #title], body)
#let definition(title, body) = thm-box([Definition: #title], body, fill: mint, stroke-color: sage)
#let remark(body) = thm-box([Remark], body, fill: rgb("#fef9ec"), stroke-color: rgb("#b7770d"))
