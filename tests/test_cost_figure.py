"""Regression test for [B-NBITS2] and [B-PDF] in the cost figure."""
import os
import csv as _csv
import pytest


def _campaign(tmp_path):
    """Minimal campaign with BOTH families and different grids."""
    rows_s = [
        {'Method': 'Classical MCMC', 'Om_mean': '0.27', 'Om_std': '0.01',
         'H0_mean': '69.6', 'H0_std': '0.8', 'Time_s': '92.5', 'nqpp': '3'},
        {'Method': 'QMCMC 100%', 'Om_mean': '0.27', 'Om_std': '0.01',
         'H0_mean': '69.6', 'H0_std': '0.8', 'Time_s': '437.0', 'nqpp': '3'},
    ]
    rows_g = [
        {'Method': 'QGA (q=0%)', 'Om_mean': '0.276', 'Om_std': '0.002',
         'H0_mean': '69.6', 'H0_std': '0.1', 'Time_s': '3.2', 'nqpp': '4'},
        {'Method': 'QGA (q=100%)', 'Om_mean': '0.276', 'Om_std': '0.002',
         'H0_mean': '69.6', 'H0_std': '0.1', 'Time_s': '47.2', 'nqpp': '4'},
    ]
    root = tmp_path / 'hpc_x'
    for task, rows, grids in (('samplers_lcdm_nqpp{}_noise-none', rows_s, (3, 5)),
                              ('genetic_lcdm_nb{}_noise-none', rows_g, (4, 8))):
        for g in grids:
            d = root / task.format(g) / 'model_lcdm'
            d.mkdir(parents=True)
            with open(d / 'results_config.csv', 'w', newline='') as fh:
                w = _csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                for f in rows:
                    w.writerow({**f, 'nqpp': str(g)})
    return str(root)


def test_b_nbits2_cost_figure_does_not_mix_families(tmp_path):
    """The cost figure of one family must NOT carry curves from the other.

    It is checked by looking at the legend labels of the axes the function
    actually created, not by rebuilding the series from outside: what
    matters is what was actually DRAWN.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cosmo_hpc_runner as runner

    created = []
    original = plt.subplots

    def spy(*a, **k):
        fig, ax = original(*a, **k)
        created.append((fig, ax))
        return fig, ax

    root = _campaign(tmp_path)
    plt.subplots = spy
    try:
        labels = []
        runner.generate_convergence_plots(root, xlabel='nqpp',
                                          family='samplers')
        for fig, ax in created:
            for axis in (ax.flatten() if hasattr(ax, 'flatten') else [ax]):
                labels += axis.get_legend_handles_labels()[1]
    finally:
        plt.subplots = original

    assert labels, 'nothing was drawn'
    intruders = [e for e in labels
                 if 'GA' in e.upper().replace('CLASSICAL', '')]
    assert not intruders, f"genetic methods in the samplers figure: {intruders}"


def test_b_pdf_cost_figure_is_also_written_as_pdf(tmp_path):
    """[B-PDF] cost_<model>.pdf must exist next to the .png."""
    import matplotlib
    matplotlib.use('Agg')
    import cosmo_hpc_runner as runner

    root = _campaign(tmp_path)
    runner.generate_convergence_plots(root, xlabel='nqpp', family='samplers')
    png = os.path.join(root, 'cost_lcdm.png')
    assert os.path.exists(png), 'cost_lcdm.png was not generated'
    assert os.path.exists(png.replace('.png', '.pdf')), \
        'cost_lcdm.pdf is missing: the paper needs vector graphics'
