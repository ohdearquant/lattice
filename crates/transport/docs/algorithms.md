# lattice-transport algorithms

This page collects operational background for transport APIs whose source-level
documentation is deliberately brief.

## Debiased divergence

Entropy regularization gives optimal transport a non-zero self-cost. Sinkhorn
divergence removes that bias by subtracting half of each self-interaction from
the cross cost: `S(a, b) = W_eps(a, b) - 0.5 W_eps(a, a) - 0.5 W_eps(b, b)`.

## Sparse plans

The solver retains dual variables rather than a dense coupling. Sparse-plan
extraction reconstructs only entries above a caller-selected mass threshold for
inspection and per-record drift reporting.

## Barycenters and unbalanced transport

Fixed-support barycenters optimize weights over a common support; the
free-support routine also relocates support points. Unbalanced Sinkhorn relaxes
the marginal constraints with KL penalties when corpus weights need not match.

## Online drift

`OnlineDriftDetector` freezes its first full window as the reference, slides a
current window over later observations, and evaluates divergence at the chosen
sample interval. Its window-size cap limits the three square solver workspaces
allocated for each detector.
