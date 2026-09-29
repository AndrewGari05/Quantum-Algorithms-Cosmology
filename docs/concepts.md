# Concepts

## The kernel is owned, so its parts can be swapped

A Metropolis-Hastings step has two replaceable parts:

1. a **proposal** draws a candidate `x'` from `q(x' | x)` and reports the
   Hastings term `log q(x | x') - log q(x' | x)`;
2. an **acceptance rule** turns `log_alpha = log p(x') - log p(x) + Hastings`
   into accept/reject.

The chain samples `p` when the acceptance probability `h` satisfies
`h(a) / h(-a) = e^a` (detailed balance). Metropolis, `h = min(1, e^a)`,
does; so does any unbiased stochastic estimate of it (shot noise is fine).
An affine distortion such as readout noise, `h = (1 - 2p) min(1, e^a) + p`,
does not: every candidate is accepted with probability at least `p`, and the
stationary distribution acquires heavier tails. qablate's acceptance rules
declare `preserves_detailed_balance`, and `MetropolisHastings` warns when it
is `False`.

A proposal that is not symmetric must report its Hastings term. The
random-circuit proposal is symmetric **by construction**: its increments are
scaled by a constant and multiplied by an independent random sign. Estimating
and subtracting a mean instead leaves a residual drift that no finite
calibration removes, and a drifting random walk under Metropolis samples a
tilted target; this was the thesis code's largest bug (ERRATA.md, QM-1).

## Ladders and equivalence checks

A `Study` names components with a classical and a quantum factory, builds a
*ladder* that switches them one at a time, and runs every configuration from
the same seed. Components that make the same random draws therefore share
their random streams, so paired configurations differ only by the component
that changed.

Some pairs are identical by construction, for example the amplitude-encoded
acceptance on an ideal simulator versus classical Metropolis: it computes
`A` classically, loads it into a qubit and reads the same number back.
`Study.check_equivalences` verifies such pairs bit for bit. They test the
plumbing of the quantum code path; they are not evidence about quantum
hardware and should not be reported as results.

## Backends and randomness

Every stochastic object takes a `numpy.random.Generator`. Backends draw a
simulator seed from it only when they sample shots; exact probabilities make
no draws. Rerunning with the same seed reproduces every number.

`AerBackend("FakeBrisbane")` places circuits on a connected path of physical
qubits of the calibrated device model and routes them against its coupling
map, so every two-qubit gate carries its calibrated error; outputs are
returned in logical qubit order.

## Variational inference on a grid

`BornMachineVI` discretizes the posterior on `2**n` points per parameter and
trains a circuit's output distribution `Q` to minimize `KL(Q || P)`. Reverse
KL is mode-seeking: when `Q` does not represent the correlation between
parameters it returns marginal widths close to the conditional widths
`sigma * sqrt(1 - rho^2)`. `VIResult.correlation` is reported so this can be
checked; for ΛCDM on CC+BAO+Pantheon (rho ≈ -0.77) the mean-field floor is
0.64 of the true width.
