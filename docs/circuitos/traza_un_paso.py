#!/usr/bin/env python3
"""traza_un_paso.py — que ENTRA y que SALE de cada circuito, con numeros.

Corre UN paso de cada uno de los tres algoritmos sobre el posterior real y
va imprimiendo, en cada frontera, que objeto cruza y de que tipo es. La
pregunta que contesta es la que no se ve en un diagrama de circuito:
¿como se pasa de un circuito a un numero?

No modifica nada ni escribe archivos. Uso:

    python docs/circuitos/traza_un_paso.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..'))
os.environ.setdefault('MPLBACKEND', 'Agg')

import cosmo_modular_quantum as cmq          # noqa: E402
import cosmo_noise as cnoise                 # noqa: E402
from cosmo_core import Posterior, MODELS     # noqa: E402


RAYA = '=' * 74
sub = lambda t: print(f"\n{'-'*74}\n  {t}\n{'-'*74}")   # noqa: E731


def marco(que, tipo, valor):
    """Una linea de la traza: que cruza, de que tipo, y su valor."""
    print(f"   {que:26s} | {tipo:26s} | {valor}")


def cabecera():
    print(f"   {'QUE CRUZA':26s} | {'TIPO':26s} | VALOR")
    print(f"   {'-'*26}-+-{'-'*26}-+-{'-'*30}")


# ═════════════════════════════════════════════════════════════════════
def traza_qmcmc(post):
    print(RAYA)
    print("  QMCMC — un paso de Metropolis completo")
    print(RAYA)
    d = post.model.n_params

    theta = np.array([0.28, 69.5])
    sub("1. CLASICO — el punto actual y su verosimilitud")
    cabecera()
    marco('theta (punto actual)', f'float[{d}]  clasico',
          np.array2string(theta, precision=4))
    lp = float(post.log_prob(theta))
    marco('log p(theta)', 'float  clasico',
          f'{lp:.4f}   <- aqui entran los {post.n_data} datos')

    sub("2. CUANTICO — la propuesta")
    eng = cmq.QuantumProposalEngine(n_phys=d, n_layers=3, batch=4)
    cabecera()
    marco('ancho del circuito', 'int', f'{eng.n_qubits} qubits  (= d, NO n_data)')
    marco('angulos de entrada', f'float[{eng.n_phi}]  aleatorios',
          'sorteados en [0, 2pi) — NADA del problema')
    # Se toma la primera propuesta que va CUESTA ABAJO. Aproximadamente la
    # mitad lo hacen (la propuesta es simetrica por construccion), y es el
    # caso interesante: cuesta arriba se acepta siempre y el circuito de
    # aceptacion queda con A = 1, que no ensena nada.
    for _ in range(200):
        bloque = eng._raw_block(1)
        delta = bloque[0] * np.array([0.05, 5.0])
        theta_p = theta + delta
        lp_p = float(post.log_prob(theta_p))
        if lp_p < lp:
            break
    marco('lo que sale de Aer', f'complex[{2**eng.n_qubits}]  amplitudes',
          'el vector de estado completo')
    marco('lectura: Re(psi)*sign(Im)', f'float[{d}]',
          np.array2string(bloque[0], precision=4))
    marco('theta\' = theta + s*delta', f'float[{d}]  clasico',
          np.array2string(theta_p, precision=4))

    sub("3. CLASICO — evaluar el punto propuesto")
    cabecera()
    marco('log p(theta\')', 'float  clasico',
          f'{lp_p:.4f}   <- otra vez los {post.n_data} datos')
    Delta = lp_p - lp
    marco('Delta', 'float', f'{Delta:+.6f}')
    A = min(1.0, float(np.exp(np.clip(Delta, -700, 700))))
    marco('A = min(1, e^Delta)', 'float en [0,1]', f'{A:.6f}')

    sub("4. CUANTICO — la aceptacion  (1 qubit)")
    cabecera()
    ang = 2.0 * np.arccos(np.sqrt(A))
    marco('angulo de entrada', 'float  UN solo numero',
          f'theta_A = 2*arccos(sqrt(A)) = {ang:.6f} rad')
    logp0 = cmq.hadamard_accept_log(lp, lp_p)
    p0 = float(np.exp(logp0))
    marco('lo que sale', 'float  P(medir 0)', f'{p0:.6f}')
    marco('comprobacion', 'P(0) deberia ser A',
          f'|P(0) - A| = {abs(p0 - A):.2e}   <- por eso es FIEL')
    u = 0.5
    marco('decision', 'bool  clasico',
          f'u={u} < P(0)={p0:.4f}  ->  {"ACEPTA" if u < p0 else "RECHAZA"}')

    print("\n   Resumen: los 1099 datos entraron DOS veces, las dos de forma")
    print("   clasica, y de ellos solo sobrevivio UN numero (A) hasta tocar")
    print("   un qubit. El circuito de propuesta ni siquiera vio el problema.")


# ═════════════════════════════════════════════════════════════════════
def traza_qvmc(post):
    print('\n' + RAYA)
    print("  QVMC — una evaluacion de la KL")
    print(RAYA)
    cfg = {'sampling': False, 'training': False, 'normalization': False}
    q = cmq.QVMCModular(post, cfg, n_qubits_per_param=3, n_layers=1,
                        n_shots=500, adaptive_grid=False)

    sub("1. CLASICO — la rejilla y el objetivo")
    cabecera()
    marco('ancho del circuito', 'int',
          f'{q.n_qubits} qubits = d({q.d}) x nqpp({q.nqpp})')
    marco('puntos de la rejilla', 'int', f'2^{q.n_qubits} = {q.n_states}')
    P = q.build_target()
    marco('P (objetivo)', f'float[{q.n_states}]  clasico',
          f'suma={P.sum():.6f}  <- construido con los {post.n_data} datos')
    marco('theta_table', f'float[{q.n_states}, {q.d}]',
          'cadena de bits -> punto del espacio de parametros')

    sub("2. CUANTICO — el ansatz")
    qc, n_phi = q._build_ansatz()
    cabecera()
    marco('angulos de entrada', f'float[{n_phi}]  los parametros phi',
          f'n_phi = n(2L+1) = {q.n_qubits}*(2*1+1) = {n_phi}')
    phi = np.full(n_phi, 0.3)
    from qiskit import transpile
    qc_t = transpile(qc.remove_final_measurements(inplace=False), q.sim)
    kl, Qs = q._kl_batch(phi, qc_t, P, return_q=True)
    Q = Qs[0]
    marco('lo que sale de Aer', f'float[{q.n_states}]  |psi|^2',
          f'suma={Q.sum():.10f}')
    marco('Q (la distribucion)', f'float[{q.n_states}]',
          f'max={Q.max():.5f} en la celda {int(np.argmax(Q))}')

    sub("3. CLASICO — comparar Q contra P")
    cabecera()
    marco('KL(Q || P)', 'float  UN numero', f'{float(kl[0]):.6f}')
    marco('estimador', f'float[{q.d}]  E_Q[theta]',
          np.array2string(Q @ q.theta_table, precision=4))

    print("\n   El circuito NUNCA vio los datos. Produjo Q; los datos vivian")
    print("   en P, y las dos se comparan clasicamente. El optimizador mueve")
    print("   los phi para bajar ese unico numero.")

    sub("4. El TERCER circuito del QVMC no es un circuito nuevo")
    cabecera()
    j = 0
    phi_mas, phi_menos = phi.copy(), phi.copy()
    phi_mas[j] += np.pi / 2
    phi_menos[j] -= np.pi / 2
    k_mas = float(q._kl_batch(phi_mas, qc_t, P)[0])
    k_menos = float(q._kl_batch(phi_menos, qc_t, P)[0])
    grad = (k_mas - k_menos) / 2.0
    marco('mismo circuito, phi[0]+pi/2', 'float  KL', f'{k_mas:.6f}')
    marco('mismo circuito, phi[0]-pi/2', 'float  KL', f'{k_menos:.6f}')
    marco('dKL/dphi[0]', 'float  derivada EXACTA',
          f'({k_mas:.6f} - {k_menos:.6f})/2 = {grad:+.6f}')
    marco('costo por iteracion', 'int  circuitos',
          f'1 + 2*{n_phi} = {1 + 2*n_phi}')
    print("\n   Por eso son TRES componentes y DOS dibujos: el entrenamiento")
    print("   reusa el ansatz con los angulos corridos pi/2.")


# ═════════════════════════════════════════════════════════════════════
def traza_qga(post):
    print('\n' + RAYA)
    print("  QGA — un gen mutado")
    print(RAYA)
    import cosmo_genetic_optimizers as G
    ga = G.GAConfig(pop_size=8, n_generations=2, seed=42)
    qga = G.QGA(post, ga, {'q_init': True, 'q_mutation': True,
                           'q_crossover': False},
                n_bits=3, rng=np.random.default_rng(42))

    sub("1. CLASICO — el individuo, como bits")
    cabecera()
    theta = np.array([0.28, 69.5])
    marco('theta (un individuo)', f'float[{qga.d}]',
          np.array2string(theta, precision=4))
    genes = qga._encode(theta[None, :])
    marco('genes (enteros)', f'int[{qga.d}]  rejilla 2^{qga.n_bits}',
          np.array2string(genes[0]))
    bits = qga._int_to_bits(genes[0], qga.n_bits)
    marco('bits', f'int[{qga.d}, {qga.n_bits}]',
          ' '.join(''.join(map(str, b)) for b in bits))

    sub("2. CUANTICO — la mutacion")
    cabecera()
    p = 0.2
    b0 = bits[0]
    P1 = b0 * (1 - p) + (1 - b0) * p
    ang = 2 * np.arcsin(np.sqrt(P1))
    marco('ancho del circuito', 'int', f'{qga.n_bits} qubits (= n_bits)')
    marco('angulos de entrada', f'float[{qga.n_bits}]',
          np.array2string(ang, precision=4))
    marco('  (de donde salen)', 'P1 = b(1-p) + (1-b)p',
          np.array2string(P1, precision=3))
    marco('lo que sale', 'cadena de bits medida',
          'una por disparo — ese ES el gen mutado')

    sub("3. CLASICO — decodificar y evaluar")
    cabecera()
    nuevos = qga._decode(genes)
    marco('theta decodificado', f'float[{qga.d}]',
          np.array2string(nuevos[0], precision=4))
    chi2, n = post.chi2(nuevos[0])
    marco('fitness (chi2)', 'float  clasico',
          f'{chi2:.4f}  sobre {n} datos   <- aqui entran los datos')

    print("\n   El circuito solo movio bits. Que tan bueno es el individuo lo")
    print("   decide el chi2, que es clasico de principio a fin.")


# ═════════════════════════════════════════════════════════════════════
def main():
    cmq.set_noise(cnoise.NoiseSpec.from_level('none'), 'auto')
    import cosmo_genetic_optimizers as G
    G.set_noise(cnoise.NoiseSpec.from_level('none'))
    post = Posterior(MODELS['lcdm'], 'CC+BAO+Pantheon')
    print(f"\nmodelo: {post.model.label} | parametros: {post.model.n_params} "
          f"| datos: {post.n_data}\n")
    traza_qmcmc(post)
    traza_qvmc(post)
    traza_qga(post)
    print('\n' + RAYA)
    print("  En los tres: los datos entran SOLO por la evaluacion clasica.")
    print("  Los qubits cuentan parametros, no datos.")
    print(RAYA + '\n')


if __name__ == '__main__':
    main()
