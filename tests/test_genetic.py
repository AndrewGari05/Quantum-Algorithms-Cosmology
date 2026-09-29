"""Experimental genetic algorithm: operator scoping and classical/circuit twins."""
import numpy as np
import pytest
from scipy import stats

from qablate.experimental import GAConfig, GeneticAlgorithm

BOUNDS = [(0.0, 1.0), (-2.0, 2.0)]
BOX = [(0.2, 0.6), (-1.0, 1.0)]


def lp(x):
    return -0.5 * np.sum(((x - [0.4, 0.3]) / [0.1, 0.5]) ** 2, axis=1)


def make(quantum=(), bits=4, seed=0, **cfg):
    return GeneticAlgorithm(lp, BOUNDS, GAConfig(pop_size=40, n_generations=5, **cfg),
                            init_box=BOX, genome_bits=bits, quantum=quantum, rng=seed)


@pytest.mark.parametrize("bad", [dict(elite_frac=1.0), dict(tournament_k=0),
                                 dict(mutation_rate=1.5), dict(pop_size=1)])
def test_config_validation(bad):
    with pytest.raises(ValueError):
        GAConfig(**bad)


def test_circuit_operators_need_grid_genome():
    with pytest.raises(ValueError):
        GeneticAlgorithm(lp, BOUNDS, quantum=("mutation",))


@pytest.mark.parametrize("quantum", [(), ("init",)])
def test_init_stays_in_init_box(quantum):
    pop = make(quantum).init_population()
    lo, hi = np.array(BOX).T
    assert np.all((pop >= lo) & (pop <= hi))                       # GA-5


@pytest.mark.parametrize("quantum", [(), ("mutation",)])
def test_mutation_only_changes_selected_genes(quantum):
    ga = make(quantum, mutation_rate=0.2)
    pop = ga.init_population()
    state = ga.rng.bit_generator.state
    sel = ga.rng.random(pop.shape) < 0.2          # the mask mutate() will draw
    ga.rng.bit_generator.state = state
    out = ga.mutate(pop)
    assert np.array_equal(out[~sel], pop[~sel])                      # GA-2


@pytest.mark.parametrize("quantum", [(), ("crossover",)])
def test_crossover_only_changes_crossing_rows(quantum):
    ga = make(quantum, crossover_rate=0.5)
    a, b = ga.init_population(), ga.init_population()
    state = ga.rng.bit_generator.state
    do = ga.rng.random(len(a)) < 0.5
    ga.rng.bit_generator.state = state
    child = ga.crossover(a, b)
    assert np.array_equal(child[~do], a[~do])                        # GA-2


def _gene_hist(ga, op, n_rep=30):
    vals = []
    for _ in range(n_rep):
        pop = ga.init_population()
        out = ga.mutate(pop) if op == "mutation" else ga.crossover(pop, ga.init_population())
        vals.append(ga._encode(out)[:, 0])
    return np.bincount(np.concatenate(vals), minlength=16)


@pytest.mark.parametrize("op", ["mutation", "crossover"])
def test_circuit_operator_has_the_classical_distribution(op):
    kw = dict(mutation_rate=1.0) if op == "mutation" else dict(crossover_rate=1.0)
    c = _gene_hist(make((), seed=1, **kw), op)
    q = _gene_hist(make((op,), seed=2, **kw), op)
    keep = (c + q) > 0
    p = stats.chi2_contingency(np.vstack([c[keep], q[keep]]))[1]
    assert p > 1e-3, p


def test_run_improves_and_reports_population_spread_not_uncertainty():
    r = make(("init", "mutation", "crossover"), seed=3).run()
    assert r.log_prob_best == pytest.approx(np.max(r.log_prob))
    assert r.history[-1]["best_log_prob"] >= r.history[0]["best_log_prob"]
    assert r.population_spread.shape == (2,)
    assert r.config == {"init": True, "mutation": True, "crossover": True}


def test_continuous_children_stay_inside_the_open_box():
    ga = GeneticAlgorithm(lp, BOUNDS, GAConfig(pop_size=40, n_generations=3, mutation_rate=1.0,
                                               mutation_scale=2.0), rng=0)
    out = ga.mutate(ga.init_population())
    lo, hi = np.array(BOUNDS).T
    assert np.all((out > lo) & (out < hi))                           # G11
