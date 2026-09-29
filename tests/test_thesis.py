"""Thesis workflow: ladders, results schema, campaign planning, legacy reader."""
import csv
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qablate.cosmology import Posterior  # noqa: E402
from thesis import campaign, ladders, results  # noqa: E402
from thesis.legacy import campaign_io, errata_tools  # noqa: E402


@pytest.fixture(scope="module")
def post():
    return Posterior("lcdm", "CC+BAO")


def test_mcmc_ladder_rows_and_plumbing_check(post):
    rows = ladders.mcmc_ladder(post, n_steps=150, n_chains=3, seed=3)
    assert [r["rung"] for r in rows] == list(ladders.MCMC_RUNGS)
    assert rows[1]["mean_Om"] == rows[2]["mean_Om"]        # identical by construction
    assert all(r["detailed_balance"] for r in rows)


def test_rungs_run_alone_equal_full_ladder(post):
    full = ladders.mcmc_ladder(post, n_steps=100, n_chains=3, seed=4)
    alone = ladders.mcmc_ladder(post, n_steps=100, n_chains=3, seed=4, rungs=["qmcmc-proposal"])
    assert alone[0]["mean_H0"] == full[1]["mean_H0"]


def test_results_schema_rejects_unknown_fields_and_other_headers(tmp_path, post):
    path = str(tmp_path / "results.csv")
    results.append(path, [{"model": "lcdm", "rung": "mcmc"}])
    with pytest.raises(KeyError):
        results.append(path, [{"not_a_field": 1}])
    with open(path) as fh:
        text = fh.read()
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        results.append(str(bad), [{"model": "lcdm"}])
    assert text.splitlines()[0].split(",") == results.FIELDS


def test_genetic_ladder_reports_grid_floor(post):
    rows = ladders.genetic_ladder(post, grid_bits=4, pop_size=30, n_generations=5, seed=1,
                                  rungs=["ga-grid"])
    floor = rows[0]["chi2_grid_floor"]
    assert abs(floor - 28.3992) < 1e-3                       # GA-1 value
    assert rows[0]["chi2_estimate"] >= floor - 1e-9


def test_campaign_plan_and_resources():
    class A:
        campaign, models, families, noise = "/tmp/x", ["lcdm", "cpl"], ["mcmc", "vi"], ["none", "readout"]
        dataset, prior, seed, grids, bits = "CC+BAO", "flat", 1, [3, 4], [4]
        steps, chains, iters, samples, population, generations = 10, 2, 2, 10, 10, 2
    tasks = campaign.plan(A)
    assert len(tasks) == 2 * 2 * (1 + 2)
    assert len({t.name for t in tasks}) == len(tasks)          # no shared folders
    noisy = [t for t in tasks if "cpl_g4_noise-readout" in t.name][0]
    assert noisy.mem_mb > 4 ** 16 * 16 / 1e6                   # density matrix of 16 qubits
    assert campaign.available_cores() >= 1 and campaign.available_memory_mb() > 0


def test_cli_run_writes_results(tmp_path):
    out = tmp_path / "t"
    r = subprocess.run([sys.executable, "-m", "thesis", "run", "genetic", "--model", "lcdm",
                        "--dataset", "CC+BAO", "--grid", "3", "--population", "10",
                        "--generations", "2", "--out", str(out), "--rungs", "ga-grid"],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    rows = results.read([str(out / "results.csv")])
    assert rows[0]["rung"] == "ga-grid" and rows[0]["code_version"]


# -- legacy campaigns (written by the v0.8 thesis code) ------------------------ #
def _legacy_task(root, name, rows, cumulative=True):
    cols = ["Method", "Om_mean", "Om_std", "H0_mean", "H0_std", "nqpp", "chi2", "AIC",
            "BIC", "n_data", "ESS", "dataset", "prior", "noise", "seed"]
    d = root / name / "model_lcdm"
    d.mkdir(parents=True)
    for p in [d / "resultados_config.csv"] + (
            [root / name / "resultados_TODOS_los_modelos.csv"] if cumulative else []):
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


def _lrow(method, om, noise="none", dataset="CC+BAO", nqpp="3", chi2="27.9"):
    return dict(Method=method, Om_mean=om, Om_std=0.02, H0_mean=70.734, H0_std=1.2,
                nqpp=nqpp, chi2=chi2, AIC=31.9, BIC=35.8, n_data=51, ESS=100,
                dataset=dataset, prior="flat", noise=noise, seed=42)


def test_legacy_reader_no_double_counting_and_optional_tags(tmp_path):
    _legacy_task(tmp_path, "samplers_lcdm", [_lrow("Classical MCMC", 0.2574, nqpp="—"),
                                             _lrow("QMCMC 50%", 0.2574, nqpp="5")])
    _legacy_task(tmp_path, "samplers_lcdm_noise-readout", [_lrow("QMCMC 50%", 0.26, "readout")])
    rows = campaign_io.read_campaign(str(tmp_path))
    assert len(rows) == 3
    assert sorted((r["_noi"], r["_g"]) for r in rows) == [("none", 5), ("none", 5), ("readout", 3)]


def test_legacy_refit_uses_legacy_dataset_names(tmp_path):
    _legacy_task(tmp_path, "samplers_lcdm", [_lrow("Classical MCMC", 0.2574, dataset="CC")])
    os.chdir(tmp_path)
    out = errata_tools.cmd_refit([str(tmp_path)], quiet=True)
    assert out[0]["dataset"] == "CC+BAO"
    assert abs(out[0]["chi2_new"] - 27.469109) < 1e-4


def test_legacy_grid_floor():
    g = errata_tools.grid_floor("lcdm", "CC+BAO", 4)
    assert abs(g["chi2_grid_floor"] - 28.3992) < 1e-3
    assert np.isfinite(g["floor_minus_continuous"])
