#import "../helpers.typ": *

// --------------------------------------------------------------------------
== Part 2 — Mathematical Foundations

#v(0.2em)
#grid(
  columns: (1fr, 1fr),
  rows: (1fr, 1fr),
  gutter: 0.7em,
  definition([Linear CKA], [
    #text(size: 0.82em)[
      For column-centered $X, Y$:
      $ "CKA"(X,Y) = frac(norm(X^top Y)_F^2, norm(X^top X)_F dot.c norm(Y^top Y)_F) $
      1 = proportional representations (no drift), 0 = orthogonal.
    ]
  ]),
  definition([Cohen's $d$], [
    #text(size: 0.82em)[
      Direction $v = (mu_"chosen" - mu_"rejected") / norm(mu_"chosen" - mu_"rejected")$:
      $ d = frac((mu_"chosen" - mu_"rejected") dot v, sigma_"pooled") $
      Effect size along the best linear separating direction.
    ]
  ]),
  definition([Anisotropy], [
    #text(size: 0.82em)[
      Over randomly sampled pairs $i eq.not j$:
      $ "aniso"(X) = EE_(i eq.not j) [cos(x_i, x_j)] $
      Near 1 = representations collapse to a narrow cone.
    ]
  ]),
  definition([Effective rank], [
    #text(size: 0.82em)[
      Participation ratio of covariance eigenvalues $lambda_i$:
      $ "erank"(X) = frac((sum_i lambda_i)^2, sum_i lambda_i^2) $
      How many dimensions are effectively in use.
    ]
  ]),
)
