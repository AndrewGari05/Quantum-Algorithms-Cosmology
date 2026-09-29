"""Regression tests for the thesis-errata fixes (branch ``thesis-errata``).

Each test is named after the finding ID in ERRATA.md and failed on the frozen
tag ``v0.8.1-thesis``.
"""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# --------------------------------------------------------------------------- #
# HPC-1: a model that fails inside --sweep-all must not be reported as OK.
# --------------------------------------------------------------------------- #
def test_hpc1_log_failure_detection(tmp_path):
    import cosmo_hpc_runner as hr
    ok = tmp_path / "ok.log"
    ok.write_text("[1/1] lcdm: DONE -> x/\n")
    bad = tmp_path / "bad.log"
    bad.write_text("[1/1] lcdm: FAILED — FileNotFoundError: x\n")
    assert not hr._log_reports_failure(str(ok))
    assert hr._log_reports_failure(str(bad))
    assert not hr._log_reports_failure(str(tmp_path / "missing.log"))


@pytest.mark.parametrize("script,extra", [
    ("cosmo_modular_quantum.py", ["--steps", "20", "--qvmc-iter", "2"]),
    ("cosmo_genetic_optimizers.py", ["--generations", "1", "--population-size", "8"]),
])
def test_hpc1_sweep_exit_code_nonzero_on_model_failure(tmp_path, script, extra):
    # Pantheon+ files are not shipped, so this dataset fails at load time.
    cmd = [sys.executable, os.path.join(ROOT, script), "--sweep-all",
           "--sweep-models", "lcdm", "--dataset", "CC+BAO+Pantheon+",
           "--no-plot", "--outdir", str(tmp_path)] + extra
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       timeout=600, env={**os.environ, "MPLBACKEND": "Agg"})
    out = r.stdout + r.stderr
    assert "failed" in out.lower(), out[-2000:]
    assert r.returncode != 0, out[-2000:]


# --------------------------------------------------------------------------- #
# HPC-2: noise channel strengths must reach the child processes.
# --------------------------------------------------------------------------- #
def test_hpc2_noise_strengths_forwarded(tmp_path):
    cmd = [sys.executable, os.path.join(ROOT, "cosmo_hpc_runner.py"), "--dry-run",
           "--models", "lcdm", "--noise", "full", "--noise-readout-p", "0.2",
           "--noise-gate-p1", "0.05", "--noise-gate-p2", "0.1",
           "--outdir", str(tmp_path / "dry")]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    child_cmds = [l for l in r.stdout.splitlines() if "--sweep-all" in l]
    assert len(child_cmds) == 2, r.stdout[-2000:]
    for line in child_cmds:
        assert "--noise-readout-p 0.2" in line
        assert "--noise-gate-p1 0.05" in line
        assert "--noise-gate-p2 0.1" in line


# --------------------------------------------------------------------------- #
# Campaign reading: QPU-6 (double counting), HPC-8/QPU-10 (optional folder
# tags), QPU-8 (cross-campaign pairing), QPU-9 (GA shift in population units).
# --------------------------------------------------------------------------- #
import csv as _csv

_COLS = ["Method", "model", "Om_mean", "Om_std", "H0_mean", "H0_std", "nqpp",
         "chi2", "AIC", "BIC", "n_data", "chi2_red", "ESS", "Time_s",
         "acceptance", "dataset", "prior", "noise", "seed"]


def _row(method, om, om_sd=0.02, nqpp="3", noise="none", dataset="CC+BAO", seed="42",
         chi2="27.5", n="51"):
    return dict(Method=method, model="lcdm", Om_mean=om, Om_std=om_sd, H0_mean=70,
                H0_std=1.2, nqpp=nqpp, chi2=chi2, AIC=float(chi2) + 4,
                BIC=float(chi2) + 7.9, n_data=n, chi2_red=0.56, ESS=100, Time_s=1.0,
                acceptance=0.5, dataset=dataset, prior="flat", noise=noise, seed=seed)


def _task(root, name, rows, cumulative=True):
    d = root / name / "model_lcdm"
    d.mkdir(parents=True)
    for path in [d / "resultados_config.csv"] + (
            [root / name / "resultados_TODOS_los_modelos.csv"] if cumulative else []):
        with open(path, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=_COLS)
            w.writeheader()
            w.writerows(rows)


def test_qpu6_no_double_counting(tmp_path):
    import campaign_io
    _task(tmp_path, "samplers_lcdm", [_row("Classical MCMC", 0.30)])
    assert len(campaign_io.read_campaign(str(tmp_path))) == 1
    assert len(campaign_io.result_csvs_any_layout(str(tmp_path))) == 1
    import cosmo_hpc_runner as hr
    assert len(hr._find_result_csvs(str(tmp_path))) == 1


def test_hpc8_task_names_without_tags_are_read(tmp_path):
    import campaign_io
    _task(tmp_path, "samplers_lcdm", [_row("QMCMC 100%", 0.30, nqpp="5")])
    _task(tmp_path, "samplers_lcdm_noise-readout",
          [_row("QMCMC 100%", 0.31, nqpp="5", noise="readout")])
    rows = campaign_io.read_campaign(str(tmp_path))
    assert sorted((r["_noi"], r["_g"]) for r in rows) == [("none", 5), ("readout", 5)]


def test_qpu8_rows_from_different_campaigns_never_pair(tmp_path):
    import graficas_ruido as gr
    a, b = tmp_path / "campA", tmp_path / "campB"
    _task(a, "samplers_lcdm_nqpp3_noise-none", [_row("QMCMC 100%", 0.30)])
    _task(a, "samplers_lcdm_nqpp3_noise-readout",
          [_row("QMCMC 100%", 0.30, noise="readout")])
    _task(b, "samplers_lcdm_nqpp3_noise-none", [_row("QMCMC 100%", 0.38)])
    rows = gr.leer_campana(str(a)) + gr.leer_campana(str(b))
    ideal = gr.indice_de_celdas_ideales(rows)
    noisy = [r for r in rows if r["_ruido"] == "readout"][0]
    assert gr.valor(noisy, "desplazamiento", ideal) == 0.0     # vs its own campaign


def test_qpu8_model_selection_refuses_mixed_datasets(tmp_path):
    import comparar_algoritmos as ca
    _task(tmp_path, "samplers_lcdm_nqpp3_noise-none", [
        _row("Classical MCMC", 0.30, dataset="CC+BAO", chi2="27.5", n="51"),
        _row("QMCMC 100%", 0.30, dataset="CC+BAO+Pantheon", chi2="1064.6", n="1099")])
    rows = ca.leer([str(tmp_path)])
    with pytest.raises(ValueError):
        ca.comparar_modelos(rows)
    one = ca.comparar_modelos(rows, dataset="CC+BAO", prior="flat")
    assert [m["n"] for m in one] == [51]


def test_qpu9_no_sigma_shift_for_genetic_rows(tmp_path):
    import graficas_ruido as gr
    fila = {"_familia": "genetic", "Method": "QGA (q=67%)", "Om_mean": 0.2398}
    assert gr.valor(fila, "desplazamiento", {}) is None
    assert "desplazamiento" not in gr.metricas_utiles([], "genetic")


# --------------------------------------------------------------------------- #
# QV-5: appending rows under a different header must not misalign columns.
# --------------------------------------------------------------------------- #
def test_qv5_csv_header_mismatch_rotates_file(tmp_path):
    import cosmo_modular_quantum as cmq
    path = str(tmp_path / "r.csv")
    cmq._write_csv_rows([{"a": 1, "b": 2}], ["a", "b"], path)
    cmq._write_csv_rows([{"a": 3, "c": 4, "b": 5}], ["a", "c", "b"], path)
    with open(path) as fh:
        rows = list(_csv.DictReader(fh))
    assert rows == [{"a": "3", "c": "4", "b": "5"}]
    moved = [p for p in os.listdir(tmp_path) if "old-schema" in p]
    assert len(moved) == 1
    cmq._write_csv_rows([{"a": 6, "c": 7, "b": 8}], ["a", "c", "b"], path)
    with open(path) as fh:
        assert len(list(_csv.DictReader(fh))) == 2          # same header: append


# --------------------------------------------------------------------------- #
# Provenance: every CSV row carries noise parameters, R-hat, a convergence
# flag and the code version (QM-3, HPC-17).
# --------------------------------------------------------------------------- #
def test_prov_fields_in_schema_and_noise_repr():
    import cosmo_modular_quantum as cmq
    import cosmo_noise as cn
    for f in ("noise_params", "rhat", "mc_converged", "code_version"):
        assert f in cmq.csv_fields_generic()
    spec = cn.NoiseSpec.from_level("readout", readout_p=0.07)
    assert spec.params == "readout_p=0.07"
    assert "gates=False" in repr(spec)                     # HPC-17: no AttributeError
    assert cn.NoiseSpec.from_level("full").metadata()["noise_has_gates"] is True


def test_prov_convergence_flag():
    import cosmo_modular_quantum as cmq
    good = cmq._convergence_fields({"rhat_final": 1.004, "ess": 900.0})
    bad = cmq._convergence_fields({"rhat_final": 1.004, "ess": 90.0})
    assert good == {"rhat": "1.0040", "mc_converged": "True"}
    assert bad["mc_converged"] == "False"
    assert cmq._convergence_fields({"kl_final": 0.1}) == {"rhat": "", "mc_converged": ""}


# --------------------------------------------------------------------------- #
# E-RUNGS: a rung run alone reproduces its rows from the full ladder.
# --------------------------------------------------------------------------- #
def test_rungs_subset_reproduces_full_ladder_rows(tmp_path):
    import cosmo_core as cc
    import cosmo_modular_quantum as cmq
    post = cc.Posterior(cc.MODELS["lcdm"], dataset="CC+BAO")
    kw = dict(n_steps_mcmc=80, max_iter_qvmc=2, nqpp=2, n_chains_mcmc=3,
              n_chains_qvmc=2, n_shots=256, outdir=str(tmp_path), seed=7,
              no_plot=True, log_every=10**9)
    full = cmq.run_quantumness_ladder(post, **kw)
    part = cmq.run_quantumness_ladder(post, rungs=["QMCMC100"], **kw)
    assert [r["pct"] for r in part["qmcmc"]] == [100.0] and part["qvmc"] == []
    ref = [r for r in full["qmcmc"] if r["pct"] == 100.0][0]
    assert (part["qmcmc"][0]["chains"] == ref["chains"]).all()
    with pytest.raises(ValueError):
        cmq.run_quantumness_ladder(post, rungs=["QMCMC75"], **kw)


def test_runner_forwards_rungs_and_qga_levels(tmp_path):
    cmd = [sys.executable, os.path.join(ROOT, "cosmo_hpc_runner.py"), "--dry-run",
           "--models", "cpl", "--rungs", "QMCMC50", "QMCMC100",
           "--qga-levels", "67", "100", "--outdir", str(tmp_path / "dry")]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                         timeout=300).stdout
    lines = [l for l in out.splitlines() if "--sweep-all" in l]
    assert any("--rungs QMCMC50 QMCMC100" in l for l in lines)
    assert any("--sweep-qga-levels 67 100" in l for l in lines)


# --------------------------------------------------------------------------- #
# QM-1: the quantum proposal must be symmetric (zero mean) for EVERY seed.
# Seed 7 with d=2 on the amplitude route had a calibration offset of -0.067
# step units in coordinate 0 (z ~ -6 over 8192 draws); the chain then sampled
# a shifted target (toy Gaussian mean -0.61, KS p = 2e-22).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("route", ["amplitude", "counts"])
def test_qm1_proposal_displacements_are_symmetric(route):
    import numpy as np
    import cosmo_modular_quantum as cmq
    import cosmo_noise as cn
    cmq.set_noise(cn.NoiseSpec.from_level("none"), proposal_route=route)
    try:
        cmq._reseed(7)
        eng = cmq.QuantumProposalEngine(2, n_layers=3, batch=512)
        assert np.all(eng._mu == 0.0)
        draws = np.array([eng.next() for _ in range(8192)])
        z = draws.mean(axis=0) / (draws.std(axis=0) / np.sqrt(len(draws)))
        assert np.all(np.abs(z) < 4.0), z
        assert np.allclose((draws ** 2).mean(axis=0), 1.0, atol=0.08)
    finally:
        cmq.set_noise(cn.NoiseSpec.from_level("none"), proposal_route="auto")


def test_qpu2_hardware_engine_has_no_calibration_drift():
    import numpy as np
    import qpu_cosmo_samplers as qpu

    class _Conn:                               # asymmetric fake device
        rng = np.random.default_rng(11)

        def transpile_isa(self, qc):
            return qc

        def run_pub(self, isa, phis, shots):
            rng = self.rng
            out = []
            for _ in phis:                     # biased <Z>: P(1) = 0.4 per qubit
                bits = rng.random((shots, 2)) < 0.4
                c = {}
                for b in bits:
                    k = "".join("1" if x else "0" for x in b[::-1])
                    c[k] = c.get(k, 0) + 1
                out.append(c)
            return out

    qpu._reseed(3)
    eng = qpu.QPUProposalEngine(_Conn(), n_phys=2, block=64, shots_per_proposal=64)
    draws = np.array([eng.next() for _ in range(8192)])
    z = draws.mean(axis=0) / (draws.std(axis=0) / np.sqrt(len(draws)))
    assert np.all(np.abs(z) < 4.0), z        # sign flip removes device bias


# --------------------------------------------------------------------------- #
# HPC-3: on a fake-backend level every two-qubit gate must carry noise, and
# routing must keep the outputs in logical qubit order.
# --------------------------------------------------------------------------- #
def test_hpc3_fake_backend_two_qubit_noise_applies():
    import numpy as np
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import ParameterVector
    from qiskit_aer import AerSimulator
    import cosmo_noise as cn
    from cosmo_core import make_simulator
    spec = cn.NoiseSpec.from_level("fake_brisbane")
    sim = make_simulator(**spec.simulator_kwargs(counts_route=False))
    n = 4
    th = ParameterVector("t", n)
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.ry(th[q], q)
    for q in range(n - 1):
        qc.cx(q, q + 1)
    qc.cx(n - 1, 0)
    qc.save_probabilities()
    t = spec.transpile(qc, sim)
    vals = np.random.default_rng(0).uniform(0, np.pi, n)
    bound = t.assign_parameters(vals)
    ideal = AerSimulator(method="statevector").run(
        transpile(qc.assign_parameters(vals), AerSimulator())).result().data(0)["probabilities"]
    routed_clean = AerSimulator(method="density_matrix").run(bound).result().data(0)["probabilities"]
    assert np.allclose(routed_clean, ideal, atol=1e-10)          # logical order kept
    noisy = sim.run(bound).result().data(0)["probabilities"]
    old = sim.run(transpile(qc, sim).assign_parameters(vals)).result().data(0)["probabilities"]
    tv = lambda a, b: 0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum()
    assert tv(noisy, ideal) > 3 * tv(old, ideal)                # 2q errors now act
    for ci in t.data:                                          # every ECR on a
        if ci.operation.num_qubits == 2 and ci.operation.name != "barrier":
            pair = tuple(t.find_bit(q).index for q in ci.qubits)
            assert pair in spec.backend.coupling_map.get_edges()   # device pair


def test_qpu5_noisy_twin_jobs_are_not_replayed():
    import numpy as np
    from qiskit import QuantumCircuit
    import cosmo_noise as cn
    import qpu_noisy_simulation as qns
    conn = qns.LocalNoisyConnection(cn.NoiseSpec.from_level("readout"), shots=256, seed=5)
    qc = QuantumCircuit(3)
    qc.h(range(3))
    qc.measure_all()
    isa = conn.transpile_isa(qc)
    a = conn.run_pub(isa, np.zeros((1, 0)))[0]
    b = conn.run_pub(isa, np.zeros((1, 0)))[0]
    assert a != b
    conn2 = qns.LocalNoisyConnection(cn.NoiseSpec.from_level("readout"), shots=256, seed=5)
    assert conn2.run_pub(conn2.transpile_isa(qc), np.zeros((1, 0)))[0] == a  # reproducible


# --------------------------------------------------------------------------- #
# CO-3/4: the best fit must reach the chi2 minimum inside the prior box,
# including a minimum on the boundary (CPL, Om = 0.18 on CC+BAO) and badly
# scaled parameters (GEDE from the fiducial start stopped 1.7 too high).
# Checked against an independent multi-start optimum.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model,dataset,start", [
    ("cpl", "CC+BAO", "fiducial"),
    ("gede", "CC+BAO+Pantheon", "fiducial"),
    ("cpl", "CC+BAO", "posterior-like"),
])
def test_co34_fit_statistics_reaches_minimum(model, dataset, start):
    import contextlib
    import io
    import numpy as np
    from scipy.optimize import minimize
    import cosmo_core as cc
    m = cc.MODELS[model]
    with contextlib.redirect_stdout(io.StringIO()):
        post = cc.Posterior(m, dataset=dataset)
    lo = np.array([b[0] for b in m.bounds])
    hi = np.array([b[1] for b in m.bounds])
    theta0 = (np.asarray(m.fiducial, float) if start == "fiducial"
              else np.array([0.20, 69.0, -0.9, 0.5]))
    st = cc.fit_statistics(post, theta0)
    assert np.isfinite(post.log_prior(st["theta_best"]))
    rng = np.random.default_rng(0)
    best = np.inf
    for _ in range(8):
        u0 = rng.uniform(0.05, 0.95, len(lo))
        r = minimize(lambda u: post.chi2(np.clip(lo + u * (hi - lo), lo, hi))[0], u0,
                     method="Nelder-Mead", bounds=[(0, 1)] * len(lo),
                     options={"maxiter": 6000, "xatol": 1e-10, "fatol": 1e-10})
        best = min(best, r.fun)
    assert st["chi2"] <= best + 1e-6, (st["chi2"], best)


# --------------------------------------------------------------------------- #
# GA-1: the QGA grid floor (value from the bug report: 28.3992 at n_bits = 4).
# --------------------------------------------------------------------------- #
def test_ga1_grid_floor_value():
    import errata_tools as et
    g = et.grid_floor("lcdm", "CC+BAO", 4)
    assert abs(g["chi2_grid_floor"] - 28.3992) < 1e-3
    assert abs(g["floor_minus_continuous"] - 0.930) < 2e-3


def test_refit_tool_reads_campaign(tmp_path):
    import errata_tools as et
    _task(tmp_path, "samplers_lcdm_nqpp3_noise-none",
          [_row("Classical MCMC", 0.2574, chi2="27.9", n="51")])
    os.chdir(tmp_path)
    out = et.cmd_refit([str(tmp_path)], quiet=True)
    assert len(out) == 1 and abs(out[0]["chi2_new"] - 27.469109) < 1e-4


def test_rungs_qmcmc_only_rerun_is_not_clamped_by_the_qvmc_grid(tmp_path):
    cmd = [sys.executable, os.path.join(ROOT, "cosmo_hpc_runner.py"), "--dry-run",
           "--only-samplers", "--models", "cpl", "--nqpp", "3", "--noise-sweep", "none,readout",
           "--rungs", "QMCMC50", "QMCMC100", "--outdir", str(tmp_path / "dry")]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300).stdout
    lines = [l for l in out.splitlines() if "--sweep-all" in l]
    assert len(lines) >= 2 and all("--nqpp 3" in l for l in lines)
