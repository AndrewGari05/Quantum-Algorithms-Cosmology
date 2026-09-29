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
