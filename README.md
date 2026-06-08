# Quantum-Classical Cosmological Parameter Estimation

> **Undergraduate Thesis — Engineering Physics**  
> Universidad Iberoamericana Ciudad de México (IBERO)   
> Advisor: Dr. García Aspeitia

This repository implements and compares **classical and quantum Bayesian parameter estimation** methods applied to the flat ΛCDM cosmological model. The parameters of interest are the matter density parameter Ω_m and the Hubble constant H₀, constrained against two observational datasets: Cosmic Chronometers (CC) and Pantheon+ Type Ia Supernovae.

---

## Repository Structure

```
.
├── lcdm_quantum_samplers_personal.py   # Main pipeline: 4 inference methods, interactive menu
├── lcdm_modular_quantum_1.py           # Modular pipeline: component-level quantum control
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Physics Background

The model assumes a spatially flat ΛCDM universe. The Hubble parameter is:

```
E²(z) = Ω_m(1+z)³ + Ω_r(1+z)⁴ + Ω_Λ
```

where `Ω_Λ = 1 − Ω_m − Ω_r` (flat geometry), and `Ω_r = 9.4×10⁻⁵` is fixed from Planck 2018.

The free parameters are **θ = (Ω_m, H₀)**.

### Datasets

| Dataset | Points | Observable | Reference |
|---|---|---|---|
| Cosmic Chronometers (CC) | 51 | H(z) [km/s/Mpc] | Jimenez & Loeb 2002 |
| Pantheon+ SNe Ia | 1048 | Distance modulus μ(z) | Brout et al. 2022 |

The Pantheon+ likelihood uses **analytical marginalization over M_abs** (the absolute magnitude nuisance parameter) via the Goliath et al. (2001) estimator:

```
χ²_eff = A − B²/C
```

where A, B, C are weighted sums of the residuals. Luminosity-distance integrals are precomputed on a grid of Ω_m values for speed.

---

## Algorithms

### 1. Classical MCMC (`ClassicalMCMC`)
Standard Metropolis-Hastings sampler with a Gaussian proposal distribution. Runs 8 parallel chains with automatic 10% burn-in and Gelman-Rubin convergence diagnostics (R̂).

### 2. Classical Variational Inference (`ClassicalVI`)
Fits a 2D Gaussian to the posterior by minimizing the KL divergence over a discrete grid. The grid resolution is set by `nqpp` (qubits per parameter) for fair comparison with QVMC. Optimization via L-BFGS-B.

### 3. QMCMC — Quantum Markov Chain Monte Carlo (`QMCMC`)
Implements the quantum proposal scheme from **Sarracino et al. (2025)**:
- **Quantum component:** proposes the Metropolis step via a parameterized quantum circuit (statevector simulation). The circuit encodes a rotation angle proportional to the log-posterior ratio, and the proposal is sampled from the resulting quantum state.
- **Classical component:** standard Metropolis acceptance/rejection, convergence diagnostics.

The Markov chain is classical; only the proposal mechanism is quantum.

### 4. QVMC — Quantum Variational Monte Carlo (`QVMC`)
Encodes the posterior as a quantum state |ψ(φ)⟩ via a hardware-efficient ansatz:
- **Classical component:** builds the target posterior on a discrete 2D grid; trains the circuit parameters by minimizing KL divergence (COBYLA optimizer).
- **Quantum component:** each circuit shot is a posterior sample. The circuit uses `2 × nqpp` qubits (default 6) and alternating Ry-Rz-CNOT layers.

---

## Quantumness Score (Modular Pipeline)

`lcdm_modular_quantum_1.py` breaks the inference pipeline into 5 components and assigns each a weight:

| Component | Weight | Description |
|---|---|---|
| Proposal | 20% | QMCMC proposal via statevector circuit |
| Acceptance | 25% | Metropolis-Hastings via Hadamard test |
| Training | 20% | QVMC gradient via parameter-shift rule |
| Sampling | 25% | QVMC samples from quantum shots |
| Normalization | 10% | Posterior normalization via QAE |

The **quantumness score** (0–100%) is a weighted sum of active quantum components. Six presets are available (0%, 20%, 45%, 70%, 90%, 100%).

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/quantum-lcdm-cosmology.git
cd quantum-lcdm-cosmology

# Install dependencies
pip install -r requirements.txt
```

### Optional: Pantheon+ data

For SNe Ia constraints, place the Pantheon+ data file in the same directory:
```
pantheon_full_parameters.txt   (Scolnic et al. 2022 / Brout et al. 2022)
```

Format (space-separated, `#` comments):
```
#name  zcmb  zhel  dz  mb  dmb
```

If the file is absent, the pipeline runs on CC data only.

---

## Usage

### Main pipeline (`lcdm_quantum_samplers_personal.py`)

```bash
python lcdm_quantum_samplers_personal.py
```

The interactive menu asks for:
1. **Hardware** — CPU (statevector simulation) or IBM Quantum
2. **Dataset** — CC only / Pantheon+ only / CC + Pantheon+
3. **Prior** — Flat or Planck 2018 Gaussian
4. **Methods** — any combination of Classical MCMC, Classical VI, QMCMC, QVMC
5. **nqpp** — qubits per parameter (grid resolution for VI and QVMC)
6. **Steps / iterations** per method

### Modular pipeline (`lcdm_modular_quantum_1.py`)

```bash
# Interactive component selection
python lcdm_modular_quantum_1.py --interactive

# Use a preset quantumness level (0, 20, 45, 70, 90, 100)
python lcdm_modular_quantum_1.py --preset 45

# Run all presets and compare performance
python lcdm_modular_quantum_1.py --benchmark

# Pass a JSON configuration
python lcdm_modular_quantum_1.py --config '{"proposal":true,"acceptance":false,"training":true,"sampling":true,"normalization":false}'
```

---

## Outputs

| File | Description |
|---|---|
| `lcdm_results_*.pdf` | Posterior plots (H(z) fit, Ω_m, H₀ marginals, KL convergence) |
| `corner_*.pdf` | 2D corner plots with 1σ/2σ contours |
| `results_log.csv` | Summary table: Ω_m, H₀ estimates, ESS, acceptance rate, runtime |
| `summary_table.pdf` | LaTeX-rendered summary table |
| `lcdm_q*.pdf` | Modular pipeline: one figure per quantumness preset |

---

## Diagnostics

The pipeline computes the following convergence and quality metrics:

| Metric | Symbol | Description |
|---|---|---|
| Effective Sample Size | ESS | Corrects for autocorrelation in MCMC chains |
| Integrated autocorr. time | τ | Mean steps between independent samples |
| Gelman-Rubin statistic | R̂ | Multi-chain convergence: R̂ < 1.05 is good |
| KL Divergence | D_KL | Variational methods: training curve |
| Wasserstein-1 distance | W₁ | Compares quantum vs classical posteriors |
| KS statistic | KS | Two-sample Kolmogorov-Smirnov test |
| Reduced chi-squared | χ²_red | Goodness of fit to data |

---

## Key Findings

- Posterior means for (Ω_m, H₀) are **consistent across all four methods**, confirming that the quantum circuits correctly encode the target distribution.
- ESS and acceptance rates differ between QMCMC and Classical MCMC, reflecting the different proposal geometries.
- The true quantum ceiling is ~60–70% (QMCMC) and ~75–85% (QVMC) due to classical data ingestion and H(z) evaluation — a fundamental boundary of current quantum-classical hybrid inference.

---

## Prior Values (Planck 2018)

| Parameter | Mean | σ |
|---|---|---|
| Ω_m | 0.3111 | 0.0056 |
| H₀ [km/s/Mpc] | 67.66 | 0.42 |

---

## Requirements

- Python ≥ 3.9
- Qiskit ≥ 1.0
- qiskit-aer ≥ 0.13
- NumPy, SciPy, Matplotlib, tqdm
- `corner` (optional, for corner plots)

See `requirements.txt` for pinned versions.

---

## References

1. Sarracino, G. et al. (2025). *Quantum Markov Chain Monte Carlo for cosmological parameter inference.* [arXiv:2509.09395](https://arxiv.org/abs/2509.09395)
2. Goliath, M. et al. (2001). *Supernovae and the nature of the dark energy.* A&A, 380, 6–18.
3. Brout, D. et al. (2022). *The Pantheon+ Analysis: Cosmological Constraints.* ApJ, 938, 110. [arXiv:2202.04077](https://arxiv.org/abs/2202.04077)
4. Planck Collaboration (2018). *Planck 2018 results. VI. Cosmological parameters.* A&A, 641, A6.
5. Jimenez, R. & Loeb, A. (2002). *Constraining Cosmological Parameters Based on Relative Galaxy Ages.* ApJ, 573, 37.

---

## License

This code is developed as part of an undergraduate thesis at IBERO. If you use or adapt it, please cite this repository and the associated thesis.

---

*Last updated: June 2026*
