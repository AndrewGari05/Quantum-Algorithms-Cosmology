"""Redraw the four single-seed thesis figures with English text (reads CSVs only; runs nothing).

Usage:
    python scripts/thesis_single_seed_figures.py NOISE_CAMPAIGN CONVERGENCE_CAMPAIGN --out FOLDER

The figures keep the file names the thesis LaTeX already uses:

  convergence_lcdm               Fig 5.4   from CONVERGENCE_CAMPAIGN
  ruido_fijo_samplers_pede_nqpp4 Fig 5.5   from NOISE_CAMPAIGN
  sigma_contra_rejilla           Fig 5.9   from NOISE_CAMPAIGN
  ruido_fijo_genetic_lcdm_n_bits4 Fig 5.12 from NOISE_CAMPAIGN

Both arguments are copies of the single-seed campaign hpc_20260907_130425
(seed 42) taken at the moment each figure was first drawn: NOISE_CAMPAIGN is
the 2026-09-21 snapshot (nicte_min.tgz, 649 rows in 5 noise levels) and
CONVERGENCE_CAMPAIGN the 2026-09-20 one (csvs_campanas.tgz). No data is
selected here beyond what the plotting functions already do; every figure
gets the note 'single-seed campaign (seed 42)'. PNG at 200 dpi plus PDF.
"""
import argparse
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import compare_algorithms as ca          # noqa: E402
import cosmo_hpc_runner as runner        # noqa: E402
import noise_plots as npl                # noqa: E402

NOTE = 'single-seed campaign (seed 42)'
NOTE_ERRATA = 'single-seed campaign (seed 42), with the errata re-runs (QM-1, HPC-3)'
QMCMC_LADDER = ('Classical MCMC', 'QMCMC 50%', 'QMCMC 100%')
QVMC_LADDER = ('Classical VI', 'QVMC 33%', 'QVMC 67%')


def convergence(root, out, note=NOTE):
    """Fig 5.4: posterior means of the LCDM ladders against nqpp."""
    with tempfile.TemporaryDirectory() as tmp:
        runner.generate_convergence_plots(root, tmp, xlabel='nqpp',
                                          family='samplers', note=note)
        made = []
        for ext in ('png', 'pdf'):
            src = os.path.join(tmp, f'convergence_lcdm.{ext}')
            shutil.copy(src, out)
            made.append(os.path.join(out, os.path.basename(src)))
    return made[0]


def noise_fixed(rows, ideal, family, model, grid, name, out, note=NOTE, only=None,
                title=None):
    """Figs 5.5 and 5.12: every noise level at one fixed grid."""
    return npl.figure_at_fixed_grid(rows, family, model, grid,
                                    npl.useful_metrics(rows, family), ideal,
                                    out, name=name, note=note, only=only, title=title)


def errata_figures(root, out):
    """The same figures from the corrected view (scripts/errata_view.py), with the
    sampler noise figure split into a QMCMC and a QVMC figure."""
    made = [convergence(root, out, note=NOTE_ERRATA)]
    rows = npl.load_campaign(root)
    ideal = npl.ideal_cell_index(rows)
    for tag, ladder, ttl in (('qmcmc', QMCMC_LADDER, 'QMCMC'), ('qvmc', QVMC_LADDER, 'QVMC')):
        made.append(noise_fixed(rows, ideal, 'samplers', 'pede', 4,
                                f'ruido_fijo_samplers_pede_nqpp4_{tag}.png', out,
                                note=NOTE_ERRATA, only=ladder,
                                title=f'{ttl}, PEDE, nqpp = 4: effect of each noise level'))
    made.append(noise_fixed(rows, ideal, 'genetic', 'lcdm', 4,
                            'ruido_fijo_genetic_lcdm_n_bits4.png', out, note=NOTE_ERRATA,
                            title='QGA, LCDM, n_bits = 4: effect of each noise level'))
    return made


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('noise_campaign')
    ap.add_argument('convergence_campaign', nargs='?')
    ap.add_argument('--out', required=True)
    ap.add_argument('--errata', action='store_true',
                    help='NOISE_CAMPAIGN is the corrected view of scripts/errata_view.py: '
                         'draw Fig 5.4, the split sampler noise figures and Fig 5.12 from it')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.errata:
        made = errata_figures(a.noise_campaign, a.out)
        for m in made:
            print('  written', m)
        return 0 if all(made) else 1

    made = [convergence(a.convergence_campaign, a.out)]
    rows = npl.load_campaign(a.noise_campaign)
    print(f'  {os.path.basename(a.noise_campaign)}: {len(rows)} rows in '
          f'{len({r["_noise"] for r in rows})} noise levels')
    ideal = npl.ideal_cell_index(rows)
    made.append(noise_fixed(rows, ideal, 'samplers', 'pede', 4,
                            'ruido_fijo_samplers_pede_nqpp4.png', a.out))
    made.append(noise_fixed(rows, ideal, 'genetic', 'lcdm', 4,
                            'ruido_fijo_genetic_lcdm_n_bits4.png', a.out))
    series = ca.sigma_vs_grid(ca.read_rows([a.noise_campaign]))
    made.append(ca.sigma_vs_grid_figure(series, a.out,
                                        name='sigma_contra_rejilla.png',
                                        note=NOTE))
    for m in made:
        print('  written', m)
    return 0 if all(made) else 1


if __name__ == '__main__':
    sys.exit(main())
