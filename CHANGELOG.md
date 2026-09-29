# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] — unreleased

First release as the `qablate` library. The thesis scripts (v0.8) are
replaced by a package with a small public API and a separate `thesis/`
workflow built on it. Results change where the v0.8 code was wrong; see
ERRATA.md for the list and the measured effect of each fix.

### Added
- `MetropolisHastings` with pluggable `Proposal` (returns its Hastings term)
  and `Acceptance` (declares whether it preserves detailed balance; the
  sampler warns when it does not). Method names follow emcee's.
- `RandomCircuitProposal` (counts or statevector route), `AmplitudeEncodedMetropolis`
  (optional readout mitigation), `BornMachineVI` on a `Grid` with COBYLA or
  exact parameter-shift training and a circuit-evaluation counter.
- Backends: `AerBackend` (ideal, synthetic readout/depolarizing noise, or a
  fake IBM device with placement on connected physical qubits),
  `IBMRuntimeBackend` (SamplerV2). All randomness comes from the caller's
  `numpy.random.Generator`; exact runs draw nothing.
- `Study`: component ladders from one seed with declared-equivalence checks.
- `diagnostics`: ArviZ rank R-hat (bulk and tail) and bulk ESS, emcee-matched
  autocorrelation time; frozen parameters count as not converged.
- `qablate.cosmology`: models, datasets (`CC+BAO`, `CC`, `Pantheon`,
  `Pantheon+sys`), vectorized posterior, `fit_statistics`.
- `qablate.experimental.GeneticAlgorithm` with grid genome and classical twins
  of every circuit-sampled operator.
- `thesis/`: ladders, tidy results schema with provenance, campaign runner,
  analysis, tools for v0.8 campaign folders.
- Statistical-validity tests (stationarity of every proposal × acceptance
  combination), golden tests against the errata code, import-contract tests.

### Changed (results differ from v0.8)
- The circuit proposal is symmetric by construction (scale-only calibration
  and a random sign). v0.8 subtracted an estimated mean and drifted (QM-1).
- Fake-device noise applies two-qubit gate errors (HPC-3).
- The best-fit search reaches the chi2 minimum, including on the prior boundary (CO-3/4).
- Circuit-sampled genetic operators change only the selected genes and
  crossing pairs, and initialization uses the same box as the classical one (GA-2, GA-5).
- The default circuit-proposal route is `counts` (realizable on hardware).
- The VI "normalization" rung is removed: its circuit output was never used.
- Dataset name `CC` now means the 31 cosmic-chronometer points; the 51-point
  table is `CC+BAO` (the v0.8 alias `CC` is mapped by the legacy tools).

### Removed
- Spanish-language study notes and phase logs (kept at tag `v0.8.1-thesis`).
- Pantheon+ (2022) loader, whose data were never available to the project.

## [0.8.x] — thesis errata (branch `thesis-errata`)
One commit per fix on top of `v0.8.1-thesis`; see ERRATA.md.
