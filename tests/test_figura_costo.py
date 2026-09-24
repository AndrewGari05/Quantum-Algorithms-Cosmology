"""Prueba de regresion de [B-NBITS2] y [B-PDF] en la figura de costo."""
import os
import csv as _csv
import pytest


def _campana(tmp_path):
    """Campana minima con las DOS familias y rejillas distintas."""
    filas_s = [
        {'Method': 'Classical MCMC', 'Om_mean': '0.27', 'Om_std': '0.01',
         'H0_mean': '69.6', 'H0_std': '0.8', 'Time_s': '92.5', 'nqpp': '3'},
        {'Method': 'QMCMC 100%', 'Om_mean': '0.27', 'Om_std': '0.01',
         'H0_mean': '69.6', 'H0_std': '0.8', 'Time_s': '437.0', 'nqpp': '3'},
    ]
    filas_g = [
        {'Method': 'QGA (q=0%)', 'Om_mean': '0.276', 'Om_std': '0.002',
         'H0_mean': '69.6', 'H0_std': '0.1', 'Time_s': '3.2', 'nqpp': '4'},
        {'Method': 'QGA (q=100%)', 'Om_mean': '0.276', 'Om_std': '0.002',
         'H0_mean': '69.6', 'H0_std': '0.1', 'Time_s': '47.2', 'nqpp': '4'},
    ]
    raiz = tmp_path / 'hpc_x'
    for tarea, filas, rejillas in (('samplers_lcdm_nqpp{}_noise-none', filas_s, (3, 5)),
                                   ('genetic_lcdm_nb{}_noise-none', filas_g, (4, 8))):
        for g in rejillas:
            d = raiz / tarea.format(g) / 'model_lcdm'
            d.mkdir(parents=True)
            with open(d / 'resultados_config.csv', 'w', newline='') as fh:
                w = _csv.DictWriter(fh, fieldnames=list(filas[0]))
                w.writeheader()
                for f in filas:
                    w.writerow({**f, 'nqpp': str(g)})
    return str(raiz)


def test_b_nbits2_la_figura_de_costo_no_mezcla_familias(tmp_path):
    """La figura de costo de una familia NO debe traer curvas de la otra.

    Se comprueba mirando las etiquetas de la leyenda de los ejes que la
    funcion realmente creo, no reconstruyendo la serie por fuera: lo que
    importa es lo que quedo DIBUJADO.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cosmo_hpc_runner as runner

    creados = []
    original = plt.subplots

    def espia(*a, **k):
        fig, ax = original(*a, **k)
        creados.append((fig, ax))
        return fig, ax

    raiz = _campana(tmp_path)
    plt.subplots = espia
    try:
        etiquetas = []
        runner.generate_convergence_plots(raiz, xlabel='nqpp',
                                          familia='samplers')
        for fig, ax in creados:
            for eje in (ax.flatten() if hasattr(ax, 'flatten') else [ax]):
                etiquetas += eje.get_legend_handles_labels()[1]
    finally:
        plt.subplots = original

    assert etiquetas, 'no se dibujo nada'
    intrusos = [e for e in etiquetas
                if 'GA' in e.upper().replace('CLASSICAL', '')]
    assert not intrusos, f"metodos geneticos en la figura de samplers: {intrusos}"


def test_b_pdf_la_figura_de_costo_tambien_sale_en_pdf(tmp_path):
    """[B-PDF] cost_<modelo>.pdf debe existir junto al .png."""
    import matplotlib
    matplotlib.use('Agg')
    import cosmo_hpc_runner as runner

    raiz = _campana(tmp_path)
    runner.generate_convergence_plots(raiz, xlabel='nqpp', familia='samplers')
    png = os.path.join(raiz, 'cost_lcdm.png')
    assert os.path.exists(png), 'no se genero cost_lcdm.png'
    assert os.path.exists(png.replace('.png', '.pdf')), \
        'falta cost_lcdm.pdf: el paper necesita vectorial'
