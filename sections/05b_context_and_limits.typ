#import "../helpers.typ": *

// --------------------------------------------------------------------------
== Part 2 · How Deep Can a Probe Even Go Here?

#text(size: 0.85em, fill: luma(90))[Reading the ~0.55 baseline before the flat curves show up]

#v(0.3em)
#grid(
  columns: (1fr, 1fr),
  gutter: 0.9em,
  align: top,
  [
    #text(weight: "bold", fill: navy, size: 1em)[What caps the accuracy?]
    #v(0.25em)
    - Bai et al. (2022): ~63% researcher–crowdworker agreement
    - Stiennon et al. (2020): 77%±2% with careful onboarding
    - Label quality is a dataset property, not a constant
    - Reward models on `hh-rlhf`: 57–66%, rarely above 72%
  ],
  [
    #text(weight: "bold", fill: navy, size: 1em)[What that means for our ~0.55]
    #v(0.25em)
    - Can't beat the label consistency of the data
    - 0.55 is low, not far outside published baselines
    - Real finding: the *missing layer trend*
    - Not the raw accuracy number itself
  ],
)

#v(0.3em)
#text(size: 0.65em, fill: luma(100))[Sources: Bai et al. 2022 · Stiennon et al. 2020 · reward-model benchmarks on `hh-rlhf`]

// --------------------------------------------------------------------------
== Part 2 · "No Linear Separation": an RLHF Finding, or a DPO Finding?

#v(0.4em)
#text(weight: "bold", fill: navy, size: 1em)[Six objectives, side by side]
#v(0.25em)
- Sinha et al. (2026, `arXiv:2606.09850`): layer-wise probing, 6 objectives
- *KTO, GRPO* build linear separability (constructive)
- *DPO, ORPO* degrade it (geometric rotation)
- *PPO, SimPO* largely preserve baseline geometry

#v(0.3em)
- Our RLHF checkpoint ≈ RLVR/PPO-like → fits "preserves" pattern
- Our DPO step: biggest CKA jump, *no* separability gain
- Fits "degrades via rotation," not "constructs a new axis"

#v(0.3em)
#remark[
  Our negative finding is likely specific to DPO/PPO-style objectives, not to "RLHF" in general.
]

#v(0.2em)
// Author list confirmed against arxiv.org/abs/2606.09850 (Sinha, Garg, Elluru, Singh, Garg) —
// still a very recent, not-yet-cited preprint; re-check before final submission.
#text(size: 0.65em, fill: luma(100))[Source: Sinha et al. 2026 (`arXiv:2606.09850`), very recent preprint, re-check before citation]

// --------------------------------------------------------------------------
== Part 2 · Methodology Check: Pooling Decides a Lot

#v(0.35em)
#table(
  columns: (1.3fr, 1.7fr, 1fr),
  stroke: 0.4pt + luma(200),
  inset: 8pt,
  align: (left, left, center),
  fill: (x, y) => if y == 0 { sky } else { white },
  [*Approach*], [*What's measured*], [*Typical accuracy*],
  [Single-point probe \ (our approach)], [Chosen/rejected as independent points, last-token pooled], [~0.5–0.6],
  [Paired-difference probe], [chosen − rejected per prompt pair, removes prompt variance], [~0.84–0.86 \ (`arXiv:2604.09870`)],
)

#v(0.3em)
- Our LinearSVC fights prompt-to-prompt variance
- Paired-difference removes that variance
- Not a weakness, a deliberately stricter question
- "Readable in raw space?" vs. "readable given pairs?"

#v(0.25em)
#remark[
  Next step: re-run a paired-difference probe as a robustness check to see whether separability
  rises once prompt variance is removed.
]

#v(0.15em)
// arXiv:2604.09870 = Kirin (2026), "Relational Preference Encoding in Looped Transformer Internal
// States" — single author confirmed via arxiv.org. Paired-diff figure (84.5%) matches; that paper's
// own single-point baseline (21.75%, a different model/method) is not directly comparable to ours
// and is intentionally left out of this table. Re-check before final submission.
#text(size: 0.65em, fill: luma(100))[Source: Kirin 2026 (`arXiv:2604.09870`), very recent preprint, re-check before citation]

// --------------------------------------------------------------------------
== Part 2 · CKA Drift Pattern: Not a One-Off

#text(size: 0.85em, fill: luma(90))[Independent cross-domain evidence for "SFT is the real turning point"]

#v(0.35em)
#grid(
  columns: (1fr, 1fr),
  gutter: 0.9em,
  align: top,
  [
    #text(weight: "bold", fill: sage, size: 1em)[Language, layer-wise SFT]
    #v(0.2em)
    - `arXiv:2604.11838`
    - Final layers far more sensitive to SFT
    - Early/middle layers stay stable
    - Matches our SFT→DPO jump
  ],
  [
    #text(weight: "bold", fill: sage, size: 1em)[Speech domain, GRPO vs. SFT]
    #v(0.2em)
    - `arXiv:2607.08409`
    - GRPO changes behavior, not representations
    - Late RL step ≈ small geometry change
    - Consistent with our DPO→RLHF pattern
  ],
)

#v(0.3em)
#text(size: 0.85em, style: "italic", fill: navy)[
  Our "SFT→DPO is the turning point" finding rhymes elsewhere, which strengthens the interpretation.
]

#v(0.15em)
// Both papers are very recent (2026) preprints with a looser thematic fit than a direct replication
// — treat as loosely-consistent supporting context, not confirmation. Re-check before submission.
#text(size: 0.65em, fill: luma(100))[Sources: Zhao et al. 2026 (`arXiv:2604.11838`) · Kumar et al. 2026 (`arXiv:2607.08409`), very recent preprints, re-check before citation]
