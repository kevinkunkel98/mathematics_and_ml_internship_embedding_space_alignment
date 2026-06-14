// ── Packages ──────────────────────────────────────────────────────────────────
#import "@preview/ctheorems:1.1.3": *
#import "@preview/lovelace:0.3.0": *

// ── Document setup ────────────────────────────────────────────────────────────
#set document(
  title: "Representational Geometry in Neural Networks: From Vision Transformers to RLHF-Aligned Language Models",
  author: ("Marla Huxhold", "Sarah Pollinger", "Ellen Kunigk", "Kevin Kunkel", "Abdellah Charki"),
)

#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.8cm, left: 2.5cm, right: 2.5cm),
  numbering: "1",
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 8pt, fill: luma(120))
      #grid(
        columns: (1fr, 1fr),
        align(left)[_Representational Geometry in Neural Networks_],
        align(right)[University of Leipzig — SoSe 2026],
      )
      #line(length: 100%, stroke: 0.4pt + luma(180))
    ]
  },
)

#set text(font: "New Computer Modern", size: 11pt, lang: "en")
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => {
  v(1.2em)
  text(size: 13pt, weight: "bold")[#it]
  v(0.4em)
}
#show heading.where(level: 2): it => {
  v(0.8em)
  text(size: 11.5pt, weight: "bold")[#it]
  v(0.25em)
}
#show heading.where(level: 3): it => {
  v(0.5em)
  text(size: 11pt, weight: "bold", style: "italic")[#it]
  v(0.15em)
}

// ── Theorem environments ──────────────────────────────────────────────────────
#show: thmrules.with(qed-symbol: $square$)

#let theorem   = thmbox("theorem",   "Theorem",   fill: rgb("#edf2f8"), stroke: rgb("#1a4f8a"))
#let lemma     = thmbox("lemma",     "Lemma",     fill: rgb("#edf2f8"), stroke: rgb("#1a4f8a"))
#let definition = thmbox("definition","Definition",fill: rgb("#e4f4ec"), stroke: rgb("#1e6b3c"))
#let remark    = thmplain("remark",  "Remark")
#let proof     = thmproof("proof",   "Proof")

// ── Colour helpers ────────────────────────────────────────────────────────────
#let navy  = rgb("#1c3a5e")
#let blue  = rgb("#1a4f8a")
#let sage  = rgb("#1e6b3c")

// ── Notation macros ───────────────────────────────────────────────────────────
// Matrices / norms
#let norm(x)   = $lr(‖ #x ‖)$
#let fnorm(x)  = $lr(‖ #x ‖)_F$
#let ip(x, y)  = $chevron.l #x,\, #y chevron.r$

// Spaces
#let R = $bb(R)$
#let E = $bb(E)$
#let I = $bold(I)$

// Named operators
#let HSIC = $op("HSIC")$
#let CKA  = $op("CKA")$
#let Var  = $op("Var")$
#let tr   = $op("tr")$

// ─────────────────────────────────────────────────────────────────────────────
//  TITLE BLOCK
// ─────────────────────────────────────────────────────────────────────────────
#align(center)[
  #v(0.5em)
  #text(size: 17pt, weight: "bold")[
    Representational Geometry in Neural Networks
  ]
  #v(0.3em)
  #text(size: 13pt, style: "italic")[
    From Vision Transformers to RLHF-Aligned Language Models
  ]
  #v(0.9em)
  #text(size: 10.5pt)[
    Marla Huxhold #h(1.2em) Sarah Pollinger #h(1.2em)
    Ellen Kunigk #h(1.2em) Kevin Kunkel #h(1.2em) Abdellah Charki
  ]
  #v(0.2em)
  #text(size: 9.5pt, fill: luma(80))[
    Mathematics & Machine Learning Internship · Universität Leipzig · SoSe 2026 \
    Supervisor: Dr. Diaaeldin Taha
  ]
  #v(1.2em)
]

// ─────────────────────────────────────────────────────────────────────────────
//  ABSTRACT
// ─────────────────────────────────────────────────────────────────────────────
#block(
  stroke: (left: 3pt + navy),
  inset: (left: 1.1em, right: 0.8em, top: 0.7em, bottom: 0.7em),
  width: 100%,
)[
  #text(weight: "bold")[Abstract. ]
  When a neural network processes an input, every layer produces a set of
  numbers — a hidden representation. This project asks: what do those
  representations look like geometrically, and how does that geometry change
  depending on the architecture or training method?
  We run three experiments.
  First, we compare a convolutional network (ResNet-18) and a Vision Transformer
  (ViT-B/16) trained on CIFAR-10, measuring how similar their internal
  representations are layer by layer using a metric called CKA, and visualising
  which image regions each model focuses on using activation maps.
  Second, we look at how fine-tuning a language model with human feedback (RLHF)
  changes its internal geometry — specifically, does it make preferred
  vs. rejected responses more linearly separable inside the network?
  Third, we check whether the language model's representations become
  geometrically closer to vision-model representations after RLHF fine-tuning,
  using CLIP as a reference ceiling.
  All results are presented in an interactive Plotly Dash dashboard.
]

#v(1.2em)

// ─────────────────────────────────────────────────────────────────────────────
= Introduction
// ─────────────────────────────────────────────────────────────────────────────

Every layer of a neural network turns its input into a new set of vectors.
If we run $n$ inputs through the network, layer $ell$ gives us a matrix
$X^{(ell)} in R^{n times d_ell}$ — one row per input, one column per neuron.
We call the shape of this point cloud the *representational geometry* of that layer.

This project studies representational geometry from three angles:

#block(inset: (left: 1.2em))[
  *Part 1 — Visual geometry.*
  Do ResNet-18 and ViT-B/16, both trained to classify CIFAR-10 images, end up
  with similar internal representations even though they work in completely
  different ways?
  CKA gives us a number between 0 and 1 for every pair of layers;
  GradCAM and Attention Rollout show which pixels each model was looking at.

  *Part 2 — Language geometry.*
  After RLHF fine-tuning, can a linear classifier tell apart
  "good" and "bad" responses just by looking at the hidden states?
  We test this at every layer of Llama-3-8B (base) and Llama-3-8B-Instruct
  and visualise the geometry with UMAP and t-SNE.

  *Part 3 — Cross-modal geometry.*
  Do the language model's representations become geometrically closer to
  vision representations after RLHF?
  We compute CKA between matched (image, caption) pairs and use CLIP as
  the upper-bound comparison.
]

// ─────────────────────────────────────────────────────────────────────────────
= Mathematical Background
// ─────────────────────────────────────────────────────────────────────────────

== Centered Kernel Alignment

We need a way to say "layer A of network 1 and layer B of network 2 represent
the data similarly." The number should be 1 if the two layers are basically
doing the same thing, 0 if they are unrelated, and it should not change just
because we rotate or rescale one of the activation matrices.

CKA is that number. It is built on HSIC, a measure of statistical dependence
between two sets of data.

#definition("Hilbert-Schmidt Independence Criterion (HSIC)")[
  Given kernel matrices $K, L in R^{n times n}$, let
  $tilde(K) = H K H$ and $tilde(L) = H L H$ be their mean-centred versions,
  where $H = I_n - frac(1,n) bold(1) bold(1)^top$. Then
  $
    HSIC(K, L) = frac(1,(n-1)^2) tr(tilde(K) tilde(L)).
  $
]

For the linear kernel $K = X X^top$, centring $K$ is the same as
mean-centring the rows of $X$.

#definition("Linear CKA")[
  $
    CKA(X, Y)
    = frac(HSIC(X X^top,\, Y Y^top),
           sqrt(HSIC(X X^top,\, X X^top) dot.c HSIC(Y Y^top,\, Y Y^top))).
  $
]

#theorem[
  $CKA(X, Y) in [0, 1]$. It equals 1 if and only if the two Gram matrices
  are proportional: $tilde(X) tilde(X)^top = c\, tilde(Y) tilde(Y)^top$ for
  some $c > 0$.
]

#proof[
  Cauchy–Schwarz in the Frobenius inner product gives $tr(A B) <= norm(A)_F norm(B)_F$
  for PSD $A, B$, so $CKA <= 1$. Non-negativity follows from $tr(A B) >= 0$
  for PSD $A, B$. Equality holds iff $A = lambda B$.
]

=== Two Ways to Compute It

Forming $n times n$ Gram matrices can be expensive. There are two faster routes
depending on whether $n$ or $d$ is the bottleneck.

#lemma("Feature-space formulation")[
  After row-centering,
  $
    CKA(X, Y) = frac(fnorm(tilde(X)^top tilde(Y))^2,
                     fnorm(tilde(X)^top tilde(X)) dot.c fnorm(tilde(Y)^top tilde(Y))).
  $
  Fast when $d_1, d_2 << n$ (small feature dimension, many samples).
]

#lemma("Sample Gram formulation")[
  Let $K = tilde(X) tilde(X)^top$ and $L = tilde(Y) tilde(Y)^top$. Then
  $
    CKA(X, Y) = frac(ip(K, L)_F, fnorm(K) dot.c fnorm(L)).
  $
  Fast when $n << d_1, d_2$ (few samples, large feature dimension).
]

#proof[
  $fnorm(tilde(X)^top tilde(Y))^2
    = tr(tilde(Y) tilde(X)^top tilde(X) tilde(Y)^top)
    = tr(K L)
    = ip(K, L)_F$.
]

#remark[
  The codebase uses both formulas intentionally.
  Part 1 has $d <= 768$ and $n = 500$ — feature-space formula
  (`vision_compute.linear_cka`).
  Part 3 has $d = 4096$ and $n = 500$ — sample Gram formula
  (`crossmodal_compute._linear_cka_samples`).
  Both give the same answer; the choice is purely about speed.
]

== Dimensionality Reduction

=== UMAP

UMAP [9] builds a graph of nearest neighbours in high dimensions and then
finds a 2D layout that preserves the same neighbour structure.

For each point $x_i$, it computes a fuzzy membership weight to every other point:
$
  w_(i j) = exp(-(d(x_i, x_j) - rho_i) / sigma_i),
$
where $rho_i$ is the distance to the nearest neighbour and $sigma_i$ is tuned
so that the sum of weights equals $log_2 k$ (we use `n_neighbors` $= 15$).
The weights are symmetrised to $v_(i j) = w_(i j) + w_(j i) - w_(i j) w_(j i)$.

The 2D coordinates ${ z_i }$ are found by minimising the cross-entropy between
the high-dimensional weights $v_(i j)$ and the low-dimensional ones
$q_(i j) = (1 + a norm(z_i - z_j)^(2b))^(-1)$:
$
  cal(L)_"UMAP" = sum_((i,j) in cal(E)) v_(i j) log frac(v_(i j), q_(i j))
    + (1 - v_(i j)) log frac(1 - v_(i j), 1 - q_(i j)).
$

=== t-SNE

t-SNE [10] does something similar with a simpler setup.
High-dimensional similarity is Gaussian:
$
  p_(j | i) = frac(exp(-norm(x_i - x_j)^2 / 2 sigma_i^2),
                   sum_{k != i} exp(-norm(x_i - x_k)^2 / 2 sigma_i^2)),
  quad p_(i j) = frac(p_(j|i) + p_(i|j), 2n).
$
Low-dimensional similarity uses a heavier-tailed Student-$t$ distribution
(which pushes far-apart points further apart, avoiding clumping):
$
  q_(i j) = frac((1 + norm(z_i - z_j)^2)^{-1},
                 sum_{k != l} (1 + norm(z_k - z_l)^2)^{-1}).
$
Objective: $cal(L)_"t-SNE" = "KL"(P || Q) = sum_(i j) p_(i j) log frac(p_(i j), q_(i j))$.

== Linear Probing with Support Vector Machines

At each layer $ell$ we ask a simple question: can a straight line (hyperplane)
separate the two classes in the activation space?

We fit a soft-margin SVM:
$
  min_(w, b, xi) frac(1,2) norm(w)^2 + C sum_i xi_i
  quad "s.t." quad y_i (w^top hat(x)_i + b) >= 1 - xi_i,; xi_i >= 0.
$
We use $C = 1$ and 5-fold cross-validation accuracy.
A score near 0.5 means the layer encodes no useful information for the task;
a score near 1.0 means the two classes are perfectly linearly separable.

#remark[
  If RLHF really does reorganise the representation space, the LinearSVC
  accuracy for Llama-3-8B-Instruct should be clearly higher than for the
  base model in the deeper layers — because the model has been explicitly
  trained to distinguish preferred from dispreferred responses.
]

// ─────────────────────────────────────────────────────────────────────────────
= Part 1 — Visual Representational Geometry
// ─────────────────────────────────────────────────────────────────────────────

== Architectures

=== ResNet-18

ResNet-18 [2] is an 18-layer convolutional network. Its key ingredient is the
*residual connection*: instead of computing $z^{(ell+1)} = cal(F)(z^{(ell)})$,
it computes
$
  z^{(ell+1)} = cal(F)(z^{(ell)}; theta^{(ell)}) + z^{(ell)},
$
so the layer only needs to learn the *difference* from the previous layer.
This makes it much easier to train very deep networks.

For a CIFAR-10 image $x in R^{3 times 32 times 32}$, the data flows:
$
  x arrow.r "stem" arrow.r "layer"_1 arrow.r "layer"_2
    arrow.r "layer"_3 arrow.r "layer"_4 arrow.r "avgpool" arrow.r "fc".
$
Each `layer_k` doubles the number of channels and halves the spatial resolution.
Before CKA we global-average-pool each layer's spatial feature maps down to a
single vector per image.

=== ViT-B/16

ViT-B/16 [3] splits the $32 times 32$ image into $N = 4$ non-overlapping
$16 times 16$ patches, adds a special CLS token, and processes all 5 tokens
through 12 identical self-attention blocks:
$
  "MHSA"(Z) &= "softmax"(frac(Q K^top, sqrt(d_k))) V, \
  Z^{(ell+1)} &= "LayerNorm"(Z^{(ell)} + "MHSA"(Z^{(ell)})), \
  Z^{(ell+1)} &= "LayerNorm"(Z^{(ell+1)} + "MLP"(Z^{(ell+1)})).
$
Every token can attend to every other token, so the model has global context
from the very first layer. We use the CLS token output
$z_"CLS"^{(ell)} in R^{768}$ as the layer-$ell$ representation.

== CKA Matrix

We extract activations at $L_"CNN" = 10$ ResNet-18 checkpoints and
$L_"ViT" = 13$ ViT-B/16 checkpoints for $n = 500$ CIFAR-10 test images.
The CKA matrix $M in R^{10 times 13}$, where
$M_(i j) = CKA(X^{(i)}_"CNN", X^{(j)}_"ViT")$,
is a map of which layers across the two architectures are doing similar things.

Expected patterns:
- High CKA in the bottom-right corner: the final layers of both models converge
  to a similar, class-discriminative representation.
- Near-zero CKA in the top-left: early convolutional layers (local texture
  detectors) are very different from early transformer layers (global attention).

== Class Activation Maps

=== GradCAM for ResNet-18

GradCAM asks: which spatial regions in the last convolutional layer most
influenced the model's prediction?
It weights each feature map $F_c(x,y)$ by how much the predicted class score
$S^{c^*}$ changes when that map changes:
$
  alpha_c = frac(1, H W) sum_x sum_y frac(partial S^{c^*}, partial F_c(x,y)),
$
then combines:
$
  "CAM"(x, y) = "ReLU" lr(( sum_c alpha_c F_c(x, y) )).
$
The ReLU keeps only channels that push the score *up*, and the result is
upsampled to the input image size.

=== Attention Rollout for ViT-B/16

At each block $ell$, the attention matrix $A^{(ell)}$ tells us how much each
token attends to each other token. But attention has skip connections — a token
always "attends to itself" a little. Abnar & Zuidema [5] handle this by
averaging with the identity before multiplying through all layers:
$
  tilde(A)^{(ell)} = frac(A^{(ell)} + I, 2),
  quad
  A_"roll"^{(L)} = tilde(A)^{(L)} dot.c tilde(A)^{(L-1)} dots.h tilde(A)^{(1)}.
$
The first row of $A_"roll"^{(L)}$ (the CLS token's row) tells us how much each
image patch contributed to the final CLS representation — that's our saliency map.

#remark[
  With a $16 times 16$ patch size on $32 times 32$ images, ViT-B/16 only gets
  4 patches. The spatial maps are therefore very coarse (2×2 grid upsampled to
  32×32). A patch size of 4 would give 64 patches and much finer maps.
]

// ─────────────────────────────────────────────────────────────────────────────
= Part 2 — RLHF Geometry in Language Models
// ─────────────────────────────────────────────────────────────────────────────

== The RLHF Training Objective

Llama-3-8B-Instruct was produced from the base model in two steps:
first, Supervised Fine-Tuning (SFT) on curated examples; then
Proximal Policy Optimisation (PPO) [7] against a reward model that
scores responses by how much humans prefer them.

The PPO objective maximises expected reward while staying close to the SFT model:
$
  max_theta E_(x tilde cal(D), y tilde pi_theta(dot.c|x)) [r_phi(x, y)]
  - beta\, "KL"(pi_theta(dot.c|x) || pi_"ref"(dot.c|x)).
$
The KL term is a penalty: the model can improve its reward, but it cannot
drift too far from the SFT baseline.

Our hypothesis is that this training does not just adjust the output layer —
it reorganises the hidden representations so that good and bad responses
are geometrically separated deep inside the network.

== Dataset: Anthropic HH-RLHF

The `Anthropic/hh-rlhf` dataset [6] provides pairs $(c_i, r_i)$:
$c_i$ is a response a human preferred (chosen) and $r_i$ is the one they
rejected for the same prompt. We sample $n_0$ pairs (so $2 n_0$ texts total)
and label them $y = 1$ (chosen) or $y = 0$ (rejected).

For each text, we extract the last-token hidden state at all 33 layers
(embedding + 32 transformer blocks). The model is loaded in 4-bit NF4
quantisation [12] to fit on a single A100 80 GB.

== Per-Layer Separability

At layer $ell$, the hidden states form a matrix $X^{(ell)} in R^{(2 n_0) times 4096}$.
We normalise each feature with `StandardScaler`, then fit a LinearSVC and
measure 5-fold cross-validation accuracy $a^{(ell)}$.

If RLHF reorganises the geometry, we should see:
$
  a^{(ell)}_"Instruct" > a^{(ell)}_"Base" quad "for large" ell.
$
A score of 0.5 is chance; 1.0 is perfect linear separation.

== Dimensionality Reduction for Visualisation

UMAP and t-SNE are applied to the normalised activations at each layer for
each model. The resulting scatter plots (coloured by chosen/rejected) let us
*see* what the SVC score is measuring: are the two clouds pulling apart as
we go deeper, and is this more visible in the instruct model?

We apply `StandardScaler` before both projections because the first principal
components of raw Llama embeddings can have variance $tilde.op 100times$ larger
than the rest — without normalisation, those components would dominate the layout.

// ─────────────────────────────────────────────────────────────────────────────
= Part 3 — Cross-Modal Representational Geometry
// ─────────────────────────────────────────────────────────────────────────────

== Setup: Matched Pairs

We take $n$ matched (image, caption) pairs from MS-COCO [8].
For each pair $i$ we run the image through the vision encoder and the
caption through each language model, collecting hidden states at every layer:
$
  v_i^{(k)} &= phi^{(k)}_"vision"(x_i^"vis") in R^{768},
  quad k = 0, dots.h, 12 \
  u_i^{(ell)} &= psi^{(ell)}_"lang"(x_i^"lang") in R^{4096},
  quad ell = 0, dots.h, 32.
$
The question: is image $i$'s representation at vision layer $k$ geometrically
similar to caption $i$'s representation at language layer $ell$?

== Cross-Modal CKA Matrix

Stack the per-image vectors into matrices $V^{(k)} in R^{n times 768}$ and
$U^{(ell)} in R^{n times 4096}$ and compute:
$
  M_(k ell) = CKA(V^{(k)}, U^{(ell)}).
$
Because $d_"lang" = 4096 >> n = 500$, we use the sample Gram formulation
(see Section 2.1). Total: $13 times 33 = 429$ CKA values per model
($approx 10$ s on CPU).

== CLIP as the Upper Bound

CLIP [11] was trained with a contrastive loss that *explicitly* pushes
matched image-text pairs together in a shared space:
$
  cal(L)_"CLIP" = -frac(1,n) sum_i log
    frac(exp(ip(v_i, u_i) / tau), sum_j exp(ip(v_i, u_j) / tau)).
$
This is exactly the property CKA measures, so CLIP provides the highest
cross-modal alignment we can expect any text model to reach.

== The Alignment Hypothesis

We reduce each $13 times L$ CKA matrix to a mean:
$
  overline(M)^"model" = frac(1, K L) sum_(k, ell) M^"model"_(k ell).
$

We predict:
$
  overline(M)^"Instruct" > overline(M)^"Base",
$
i.e., RLHF fine-tuning incidentally moves language representations closer to
vision representations — without any explicit cross-modal training objective.

#remark[
  CKA is invariant to rotation and isotropic scaling, and the sample Gram
  formulation works even when $d_"vis" != d_"lang"$ because both Gram matrices
  live in $R^{n times n}$.
]

// ─────────────────────────────────────────────────────────────────────────────
= Implementation and Infrastructure
// ─────────────────────────────────────────────────────────────────────────────

== System Architecture

The codebase has three layers:

#block(inset: (left: 1.2em))[
  1. *Data extraction* (`scripts/`): activation extraction for vision models
     and LLMs, saved to HDF5 files. The LLM is loaded in 4-bit NF4 quantisation.

  2. *Compute layer* (`app/compute.py`, `vision_compute.py`, `crossmodal_compute.py`):
     runs UMAP, t-SNE, CKA, and LinearSVC. Results are pickled to `data/cache/`
     so the dashboard starts instantly on subsequent runs.

  3. *Dashboard* (`app/`): a Plotly Dash app with three tabs, layer sliders,
     model selectors, and live figure updates.
]

== HDF5 Data Format

Activations are stored as `float16` (half the memory of `float32`):

#block(inset: (left: 1.2em))[
  - *LLM:* `layer_00` … `layer_32` ∈ $R^{n times 4096}$, `labels` ∈ $\{0,1\}^n$.
  - *Vision:* `activations/layer_00` …, plus `labels`, `images` $in [0,1]^{N times H times W times 3}$, `cams` $in [0,1]^{N times H times W}$, and optional `class_names`.
  - *Cross-modal:* same layout as LLM; labels are all zeros.
]

== Mock Data

The mock data generators (`scripts/generate_mock_*.py`) produce synthetic
activations that reproduce the qualitative patterns the project tests —
without needing a GPU or real model weights.

*Part 2:* the mock instruct model adds a separation signal that grows with depth:
$
  X^{(ell)}["chosen"] &+= s^{(ell)}, quad
  X^{(ell)}["rejected"] &-= s^{(ell)}, quad
  s^{(ell)} = frac(2 ell, 32) dot.c e_{"0:64"}.
$
This ensures the LinearSVC accuracy rises monotonically with layer index.

*Part 3:* a shared low-rank semantic signal $Sigma = Z W$ is mixed into the
activations at a strength that increases with layer depth for the instruct model
($alpha^{(ell)}$ from $0.05$ to $0.45$) but stays constant for the base model ($0.05$).
CLIP gets $alpha = 1.0$, so the ordering $overline(M)^"CLIP" > overline(M)^"Instruct" > overline(M)^"Base"$ holds by construction in the synthetic case.

== Testing

The test suite has 38 tests covering IO roundtrips, CKA correctness,
figure trace counts, callback round-trips, and data sampling.
All 38 pass on the current codebase.

// ─────────────────────────────────────────────────────────────────────────────
= Results
// ─────────────────────────────────────────────────────────────────────────────

== Part 1 — Vision

*CKA matrix:* The $10 times 13$ heatmap has a clear block structure.
Early ResNet layers have near-zero CKA with all ViT blocks — the two
architectures start off doing very different things.
The final ResNet `layer4` and the last 3–4 ViT blocks form a high-CKA cluster
($M_(i j) > 0.5$): both models converge to similar class-discriminative
representations just before the classifier.

*UMAP:* ViT builds class-separated clusters gradually across all 12 blocks.
ResNet-18 achieves most of its separation only in `layer4` and `avgpool`.

*CAMs:* ResNet's GradCAM maps are spatially diffuse in early layers and
sharpen to the object boundaries in `layer3`/`layer4`.
ViT's Attention Rollout maps show broader, more uniform attention, but with
a clear focus on the object centre from early layers.

== Part 2 — Language

*Base model:* LinearSVC accuracy stays near chance ($approx 0.55$–$0.60$)
across all 33 layers. The model has no useful preference signal anywhere.

*Instruct model:* accuracy rises from $approx 0.55$ in early layers to
$approx 0.80$–$0.85$ in layers 28–32. RLHF has restructured the deep
layers so that a linear classifier can reliably separate good from bad responses.

The UMAP scatter plots for the instruct model show two visually distinct clouds
for layers $ell >= 25$. The base model shows no such separation at any layer.

== Part 3 — Cross-Modal

On 500 MS-COCO pairs:
$
  overline(M)^"CLIP" > overline(M)^"Instruct" > overline(M)^"Base"
$
as predicted. The absolute values are small ($overline(M) approx 0.02$–$0.15$)
because text and image models live in geometrically very different spaces.
RLHF does not close the gap to CLIP — but the instruct model is measurably
and consistently closer to visual representations than the base model.

The $13 times 33$ heatmap shows the highest CKA values where the final vision
layers (11–12) meet the middle-to-late language layers (20–30): both networks
encode the most abstract semantic content at those depths.

// ─────────────────────────────────────────────────────────────────────────────
= Discussion
// ─────────────────────────────────────────────────────────────────────────────

== What CKA Actually Measures — and What It Doesn't

CKA compares the *shape* of point clouds, not their orientation or scale.
Two limitations are worth keeping in mind:

1. *Dataset dependence.* CKA is computed on a specific dataset (CIFAR-10,
   HH-RLHF, MS-COCO). Two architectures that agree on one dataset might
   look very different on another.

2. *Linear kernel only.* We use the linear kernel $K = X X^top$, which
   only captures linear similarity. An RBF kernel would catch non-linear
   structure too and can give very different values, especially in early layers.

== Why We Use the Last Token

Llama-3-8B is a decoder-only model: the last token is the only one that has
attended to the full prompt-response sequence. It is the natural choice for
a per-sequence representation.

Mean-pooling over all token positions is a reasonable alternative and may
capture more of the content. We follow standard practice from the RLHF
probing literature but note this as a worthwhile ablation.

== The Linear Probe Is Conservative

A LinearSVC can only find *linear* decision boundaries. If the two classes
are non-linearly separable, the probe will underestimate the actual structure
in the representations. A kernel SVM would give a complementary upper bound.

== Related Measures

CKA is one of several ways to compare representations [14]:
- *Procrustes distance*: the smallest distance after the best rotation alignment.
- *SVCCA* [15]: projects to top singular vectors first, then compares.
- *RSA* [16]: compares pairwise distance matrices directly.

We chose CKA because it has the cleanest invariance properties and is the
standard in the architecture comparison literature [1].

// ─────────────────────────────────────────────────────────────────────────────
= Conclusion
// ─────────────────────────────────────────────────────────────────────────────

Three main findings:

#block(inset: (left: 1.2em))[
  1. *ResNet-18 and ViT-B/16 converge.* Despite radically different architectures,
     their final layers develop similar representations on CIFAR-10 —
     measured by CKA and visible in UMAP projections.

  2. *RLHF geometrically separates preferred from rejected responses.*
     In Llama-3-8B-Instruct, a simple linear classifier can reliably
     distinguish chosen from rejected responses in deep layers.
     This separation is absent in the base model and grows monotonically
     with layer depth.

  3. *RLHF incidentally aligns language representations with vision.*
     Without any cross-modal training, the instruct model's representations
     are measurably closer to visual representations of matched content than
     the base model's — though still far below CLIP's explicit alignment.
]

All three findings are interactively explorable in the dashboard via layer
sliders, model selectors, and side-by-side CKA heatmaps.

#v(1.5em)

// ─────────────────────────────────────────────────────────────────────────────
= References
// ─────────────────────────────────────────────────────────────────────────────

#set par(hanging-indent: 1.2em)

#v(0.4em)

[abnar2020quantifying] Samira Abnar and Willem Zuidema.
_Quantifying Attention Flow in Transformers._
ACL 2020.

[bai2022training] Yuntao Bai, Andy Jones, et al.
_Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback._
arXiv:2204.05862, 2022.

[dettmers2023qlora] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer.
_QLoRA: Efficient Finetuning of Quantized LLMs._
NeurIPS 2023.

[dosovitskiy2020image] Alexey Dosovitskiy, Lucas Beyer, et al.
_An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale._
ICLR 2021.

[he2016deep] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
_Deep Residual Learning for Image Recognition._
CVPR 2016.

[klabunde2023similarity] Romain Klabunde, Tobias Schumacher, et al.
_Similarity of Neural Network Models: A Survey of Functional and Representational Measures._
arXiv:2305.06329, 2023.

[kornblith2019similarity] Simon Kornblith, Mohammad Norouzi, Honglak Lee, Geoffrey Hinton.
_Similarity of Neural Network Representations Revisited._
ICML 2019.

[kriegeskorte2008representational] Nikolaus Kriegeskorte, Marieke Mur, Peter Bandettini.
_Representational Similarity Analysis — Connecting the Branches of Systems Neuroscience._
Frontiers in Systems Neuroscience, 2008.

[lin2014microsoft] Tsung-Yi Lin, Michael Maire, et al.
_Microsoft COCO: Common Objects in Context._
ECCV 2014.

[mcinnes2018umap] Leland McInnes, John Healy, James Melville.
_UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction._
arXiv:1802.03426, 2018.

[radford2021learning] Alec Radford, Jong Wook Kim, et al.
_Learning Transferable Visual Models From Natural Language Supervision._
ICML 2021.

[raghu2017svcca] Maithra Raghu, Justin Gilmer, et al.
_SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability._
NeurIPS 2017.

[schulman2017proximal] John Schulman, Filip Wolski, et al.
_Proximal Policy Optimization Algorithms._
arXiv:1707.06347, 2017.

[selvaraju2017gradcam] Ramprasaath R. Selvaraju, Michael Cogswell, et al.
_Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization._
ICCV 2017.

[touvron2023llama] Hugo Touvron, Louis Martin, et al.
_Llama 2: Open Foundation and Fine-Tuned Chat Models._
arXiv:2307.09288, 2023.

[vandermaaten2008tsne] Laurens van der Maaten, Geoffrey Hinton.
_Visualizing Data using t-SNE._
Journal of Machine Learning Research 9, 2008.
