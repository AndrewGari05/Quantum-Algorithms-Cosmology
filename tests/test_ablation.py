"""
test_ablation.py — Tests for the classical-quantum ablation framework (B6).

These verify the uniform-weight ablation index and the FAITHFUL/ALGORITHMIC
classification WITHOUT importing the full sampler module (which pulls in
Qiskit). We parse the relevant definitions out of cosmo_modular_quantum.py and
cosmo_genetic_optimizers.py at the source level, so the test runs in any
environment.

The ablation index must satisfy:
  * uniform weighting: index = 100·(#quantum components)/(#components);
  * monotonic per-method ladders (each rung flips exactly one point);
  * QMCMC ladder = [0, 50, 100]; QVMC ladder = [0, 33.3, 67(.7 rounded), 100];
  * the legacy weighted index is still reproducible (CSV continuity);
  * every component is tagged FAITHFUL or ALGORITHMIC, and the tags match the
    design (proposal/training = ALGORITHMIC; acceptance/sampling/normalization
    = FAITHFUL; all three QGA operators = ALGORITHMIC).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _exec_block(path, names):
    """Exec a source file in a stubbed namespace and return selected names.

    We stub out the heavy imports (numpy, qiskit, matplotlib, ...) so the
    module-level ablation definitions evaluate without their dependencies.
    Only the pure-Python ablation helpers are exercised.
    """
    import types

    src = open(os.path.join(ROOT, path)).read()
    # Keep only up to the first class/sampler definition to avoid executing
    # qiskit-dependent code: we slice at a sentinel that follows all the
    # ablation helpers in both files.
    ns = {}
    # Provide harmless stubs for typing imports used in annotations.
    from typing import Dict, List
    ns.update({'Dict': Dict, 'List': List})
    # Execute line ranges that define the ablation helpers only.
    # Simpler and robust: compile the whole file but guard imports.
    stub = types.ModuleType('stub')

    class _Any:
        def __getattr__(self, k):
            return _Any()

        def __call__(self, *a, **k):
            return _Any()

    for mod in ('numpy', 'matplotlib', 'matplotlib.pyplot',
                'matplotlib.gridspec', 'matplotlib.lines', 'scipy',
                'scipy.optimize', 'tqdm', 'corner', 'qiskit',
                'qiskit.circuit', 'qiskit_aer', 'cosmo_core',
                'cosmo_modular_quantum'):
        sys.modules.setdefault(mod, _Any())
    # Execute only the lines before the first class/decorated-class to stay in
    # pure-config territory (both files put all ablation helpers above their
    # classes). Cut at the earliest of '\nclass ' or '\n@dataclass'.
    cut = len(src)
    for sentinel in ("\nclass ", "\n@dataclass"):
        i = src.find(sentinel)
        if i != -1:
            cut = min(cut, i)
    head = src[:cut]
    exec(compile(head, path, 'exec'), ns)
    return {n: ns[n] for n in names if n in ns}


def test_quantum_ablation_uniform_index():
    g = _exec_block('cosmo_modular_quantum.py',
                    ['compute_quantumness', 'quantumness_qmcmc',
                     'quantumness_qvmc', 'qmcmc_ladder', 'qvmc_ladder',
                     'QUANTUM_COMPONENTS', 'component_kinds',
                     'legacy_weighted_index', 'FAITHFUL', 'ALGORITHMIC'])
    ci = g['compute_quantumness']
    assert ci({}) == 0.0
    assert ci({'proposal': True}) == 20.0           # 1/5
    assert ci(dict(proposal=True, acceptance=True, training=True,
                   sampling=True, normalization=True)) == 100.0

    # per-method ladders, uniform
    qm = [g['quantumness_qmcmc'](c) for c in g['qmcmc_ladder']()]
    qv = [g['quantumness_qvmc'](c) for c in g['qvmc_ladder']()]
    assert qm == [0.0, 50.0, 100.0]
    assert qv[0] == 0.0 and qv[-1] == 100.0
    assert abs(qv[1] - 33.3) < 0.1 and abs(qv[2] - 66.7) < 0.1

    # ladders are monotonic
    assert all(b >= a for a, b in zip(qm, qm[1:]))
    assert all(b >= a for a, b in zip(qv, qv[1:]))


def test_quantum_component_kinds():
    g = _exec_block('cosmo_modular_quantum.py',
                    ['QUANTUM_COMPONENTS', 'component_kinds',
                     'FAITHFUL', 'ALGORITHMIC'])
    comps = g['QUANTUM_COMPONENTS']
    F, A = g['FAITHFUL'], g['ALGORITHMIC']
    assert comps['proposal']['kind'] == A
    assert comps['training']['kind'] == A
    assert comps['acceptance']['kind'] == F
    assert comps['sampling']['kind'] == F
    assert comps['normalization']['kind'] == F
    # active-kind map
    active = g['component_kinds'](dict(proposal=True, acceptance=True))
    assert active == {'proposal': A, 'acceptance': F}


def test_legacy_weighted_index_reproduces_old_numbers():
    g = _exec_block('cosmo_modular_quantum.py', ['legacy_weighted_index'])
    lw = g['legacy_weighted_index']
    # old global weighted scale: proposal20 accept25 train20 sample25 norm10
    assert lw({}) == 0.0
    assert lw(dict(proposal=True)) == 20.0
    assert lw(dict(proposal=True, acceptance=True, training=True,
                   sampling=True, normalization=True)) == 100.0
    # the historical QMCMC 44%-equivalent (proposal+sampling weighted global)
    assert lw(dict(proposal=True, sampling=True)) == 45.0


def test_qga_ablation_uniform_and_kinds():
    g = _exec_block('cosmo_genetic_optimizers.py',
                    ['compute_qga_quantumness', 'QGA_COMPONENTS',
                     'legacy_qga_weighted_index', 'FAITHFUL', 'ALGORITHMIC'])
    ci = g['compute_qga_quantumness']
    assert ci({}) == 0.0
    assert ci(dict(q_init=True)) == round(100 / 3, 1)
    assert ci(dict(q_init=True, q_mutation=True)) == round(200 / 3, 1)
    assert ci(dict(q_init=True, q_mutation=True, q_crossover=True)) == 100.0
    A = g['ALGORITHMIC']
    for k in ('q_init', 'q_mutation', 'q_crossover'):
        assert g['QGA_COMPONENTS'][k]['kind'] == A
    # legacy weights still reproducible
    lw = g['legacy_qga_weighted_index']
    assert lw(dict(q_init=True)) == 25.0
    assert lw(dict(q_crossover=True)) == 40.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
