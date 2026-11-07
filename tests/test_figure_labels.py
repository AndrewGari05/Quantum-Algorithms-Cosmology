"""Figure text is English and uses the display names of the QVMC rungs."""
import importlib.util
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import campaign_io  # noqa: E402
import compare_algorithms  # noqa: E402
import noise_plots  # noqa: E402

# Spanish words that used to appear in figures, plus any accented letter.
SPANISH = re.compile(
    r'[áéíóúñ¿¡]|\b(desplazamiento|tiempo|rejilla|pelda(n|ñ)o|semillas?|ruido|'
    r'sesgo|dispositivo|cl[aá]sico|fracci[oó]n|aceptaci[oó]n|mejor|refinar|'
    r'ancho|ubicaci[oó]n|arranque|caliente|m[ií]nimo|facturada|costo|piso|'
    r'malla|distancia|desviaci[oó]n|est[aá]ndar|simulador|acuerdo|exacta|'
    r'antes|despu[eé]s|media|semilla)\b', re.IGNORECASE)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _texts():
    hw = os.path.join(ROOT, 'scripts', 'hardware')
    seeds = _load(os.path.join(hw, 'hw_seeds.py'), 'hw_seeds')
    summary = _load(os.path.join(hw, 'hw_summary.py'), 'hw_summary')
    out = [label for label, _ in noise_plots.METRICS.values()]
    out += list(noise_plots.NOISE_LABELS.values())
    out += [noise_plots.NOISE_LEGEND_TITLE, compare_algorithms.REFERENCE]
    out += list(campaign_io._QVMC_DISPLAY.values())
    out += list(seeds.LOC_LABEL.values()) + [seeds.loc_label('device', 'ibm_fez')]
    out += [seeds.rule_text(n) for n in (1, 2, 3)]
    out += list(summary.LABEL.values())
    return out


def test_no_spanish_in_label_dictionaries():
    hits = [t for t in _texts() if SPANISH.search(t)]
    assert not hits, hits


def test_spanish_pattern_catches_the_old_labels():
    for old in ('desplazamiento maximo', 'peldano de ruido', 'simulador\ncon ruido',
                'clásico', 'QPU facturada [s]'):
        assert SPANISH.search(old), old


def test_code_names_of_the_qvmc_rungs_never_reach_a_figure():
    for code in ('QVMC 33%', 'QVMC 67%'):
        assert campaign_io.display_method(code) != code
        assert 'QVMC 33' not in campaign_io.display_method(code)
        assert 'QVMC 67' not in campaign_io.display_method(code)
    assert campaign_io.is_hidden_method('QVMC 100%')
