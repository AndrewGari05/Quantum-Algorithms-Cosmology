"""
test_noise_axis.py — Regresiones del segundo eje de ablacion (ruido NISQ).

Cubre `cosmo_noise` y la guarda que este eje anade a
`cosmo_core.make_simulator`. Cada test que corresponde a un bug encontrado
lleva su etiqueta ([N1], [B-RO]) para que el motivo del test no se pierda.

Los tests marcados `qiskit` se saltan sin Qiskit/Aer, igual que el resto de la
suite.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cosmo_noise as cn                                     # noqa: E402


# =============================================================================
# Nombres y resolucion de niveles (no necesitan Aer)
# =============================================================================

def test_canonical_level_normaliza_variantes_de_backend():
    """Las tres grafias de un backend deben colapsar a UNA etiqueta de CSV.

    Si no colapsan, el eje de ruido genera filas duplicadas que parecen
    peldanos distintos.
    """
    assert cn.canonical_level('FakeBrisbane') == 'fakebrisbane'
    assert cn.canonical_level('fake_brisbane') == 'fakebrisbane'
    assert cn.canonical_level('  FAKE-BRISBANE ') == 'fakebrisbane'
    assert cn.canonical_level(None) == 'none'
    assert cn.canonical_level('') == 'none'
    for lvl in cn.NAMED_LEVELS:
        assert cn.canonical_level(lvl.upper()) == lvl


def test_nivel_desconocido_levanta():
    """Un nivel mal escrito debe fallar ruidosamente, no caer a 'none'."""
    with pytest.raises(ValueError):
        cn.NoiseSpec.from_level('no_existe_este_backend')


def test_techo_de_qubits_se_deriva_de_la_ram():
    """[REV] El techo con ruido escala con la RAM, no es una constante.

    La primera version del eje lo fijaba en 13 duro, argumentando un limite de
    tiempo. Era una mala generalizacion sacada de una maquina de 7 GB: en un
    nodo grande y sin prisa el limitante vuelve a ser la memoria, y esa si se
    relaja. Este test fija que ahora crece de forma monotona con la RAM.
    """
    prev = 0
    for gb in (7, 64, 256, 1024):
        c = cn.noisy_qubit_ceiling(mem_mb=gb * 1024)
        assert c >= prev, "el techo no puede bajar al dar mas RAM"
        prev = c
    assert cn.noisy_qubit_ceiling(mem_mb=7 * 1024) < \
        cn.noisy_qubit_ceiling(mem_mb=1024 * 1024)

    # sin cifra de RAM se cae al respaldo conservador
    assert cn.noisy_qubit_ceiling() == cn.DEFAULT_NOISY_QUBITS
    # el tope del usuario sigue siendo cota superior
    assert cn.noisy_qubit_ceiling(8, mem_mb=1024 * 1024) == 8
    assert cn.noisy_density_bytes(13) == 2 ** 26 * 16


def test_el_lote_de_parameter_shift_es_lo_que_manda():
    """Con entrenamiento cuantico el techo lo fija el lote, no rho suelta.

    `2 * n_phi` matrices de densidad a la vez: a 12 qubits son ~42 GB frente a
    los 256 MB de una sola. Ignorarlo daria un techo demasiado optimista y la
    tarea moriria por OOM justo en los rungs 67% y 100%.
    """
    assert cn.param_shift_batch_factor(12) > 100
    solo = cn.noisy_density_bytes(12)
    lote = cn.noisy_density_bytes(12, cn.param_shift_batch_factor(12))
    assert lote == solo * cn.param_shift_batch_factor(12)
    assert lote / 2 ** 30 > 40          # ~42 GB

    con = cn.noisy_qubit_ceiling(mem_mb=95 * 1024, quantum_training=True)
    sin = cn.noisy_qubit_ceiling(mem_mb=95 * 1024, quantum_training=False)
    assert con < sin, "el lote tiene que recortar mas que rho suelta"


# =============================================================================
# Peldano ideal: debe ser EXACTAMENTE la ruta previa al eje de ruido
# =============================================================================

def test_none_es_ideal_y_no_toca_nada():
    """`--noise none` tiene que reproducir la ruta ideal bit a bit.

    Es la condicion que hace de la columna ideal una linea base y no una
    aproximacion: mismo metodo de simulacion, sin modelo de ruido, y un
    `apply_readout` que devuelve el MISMO objeto sin copiar ni redondear.
    """
    spec = cn.NoiseSpec.from_level('none')
    assert spec.is_ideal
    assert not spec.has_readout
    assert spec.full_model() is None
    assert spec.simulator_kwargs(counts_route=False) == {
        'method': 'statevector'}
    assert spec.simulator_kwargs(counts_route=True) == {
        'method': 'statevector'}

    p = np.array([0.1, 0.2, 0.3, 0.4])
    assert spec.apply_readout(p, 2) is p


# =============================================================================
# [N1] La guarda de make_simulator
# =============================================================================

@pytest.mark.qiskit
def test_n1_statevector_con_ruido_levanta():
    """[N1] statevector + noise_model devuelve el resultado IDEAL en silencio.

    Aer no avisa: la corrida completa, reporta exito y entrega numeros sin
    ruido. Una campana ruidosa entera volveria limpia y plausible. La
    combinacion tiene que ser imposible de construir.
    """
    from cosmo_core import make_simulator

    spec = cn.NoiseSpec.from_level('full')
    with pytest.raises(ValueError, match=r'\[N1\]'):
        make_simulator(method='statevector',
                       noise_model=spec.full_model())


@pytest.mark.qiskit
def test_simulator_kwargs_nunca_produce_la_combinacion_prohibida():
    """Ningun peldano puede generar por si mismo la pareja que [N1] prohibe."""
    from cosmo_core import make_simulator

    for level in ('none', 'readout', 'full'):
        spec = cn.NoiseSpec.from_level(level)
        for counts_route in (False, True):
            kwargs = spec.simulator_kwargs(counts_route)
            if kwargs.get('noise_model') is not None:
                assert kwargs['method'] == 'density_matrix'
            make_simulator(**kwargs)          # no debe levantar


# =============================================================================
# [B-RO] El mapa analitico de lectura
# =============================================================================

def _diag_rho(qc, spec, n):
    """diag(rho) del circuito bajo el ruido de COMPUERTA del peldano."""
    from qiskit import transpile
    from cosmo_core import make_simulator

    sim = make_simulator(**spec.simulator_kwargs(counts_route=False))
    q = qc.copy()
    q.save_density_matrix()
    rho = np.asarray(
        sim.run(transpile(q, sim, optimization_level=0)).result()
        .data(0)['density_matrix'])
    p = np.real(np.diag(rho))
    return p / p.sum()


def _counts_probs(qc, spec, n, shots, seed=5):
    """Probabilidades medidas con el modelo COMPLETO (Aer aplica la lectura)."""
    from qiskit import transpile
    from cosmo_core import make_simulator

    sim = make_simulator(seed_simulator=seed,
                         **spec.simulator_kwargs(counts_route=True))
    q = qc.copy()
    q.measure_all()
    counts = sim.run(transpile(q, sim, optimization_level=0),
                     shots=shots).result().get_counts()
    out = np.zeros(2 ** n)
    for key, val in counts.items():
        out[int(key, 2)] = val / shots
    return out


def _sample_circuit(n, seed):
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(float(rng.uniform(0, np.pi)), i)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


@pytest.mark.qiskit
@pytest.mark.parametrize('n', [1, 2, 3])
def test_b_ro_lectura_analitica_coincide_con_conteos(n):
    """[B-RO] El canal de lectura debe alcanzar TODOS los qubits, no solo el 0.

    Un `add_all_qubit_readout_error` aparece en `to_dict()` SIN clave
    `gate_qubits`. Interpretar esa ausencia como `[[0]]` degradaba el canal a
    un solo qubit: el mapa analitico ruidificaba n qubits y Aer solo uno. El
    sintoma era silencioso — coincidencia exacta en n=1 y divergencia
    creciente con n y con p — asi que el test barre n>1 a proposito.
    """
    shots = 400_000
    spec = cn.NoiseSpec.from_level('readout', readout_p=0.05)
    qc = _sample_circuit(n, seed=100 + n)

    predicted = spec.apply_readout(_diag_rho(qc, spec, n), n)
    observed = _counts_probs(qc, spec, n, shots)

    # 5 sigma de ruido de disparo, con margen por el maximo sobre 2^n celdas.
    tol = 5 * 0.5 / np.sqrt(shots) + 1e-3
    assert np.max(np.abs(predicted - observed)) < tol


@pytest.mark.qiskit
def test_lectura_analitica_con_canal_no_uniforme():
    """Un backend real trae una matriz por qubit; el mapa debe respetar cual.

    Las probabilidades de volteo de FakeBrisbane difieren entre qubits, de
    modo que cualquier permutacion de qubits o transposicion de las matrices
    rompe este test (a diferencia del canal uniforme, que es ciego a ambas).
    """
    n, shots = 3, 400_000
    spec = cn.NoiseSpec.from_level('FakeBrisbane')
    mats = [spec.readout_matrix(q) for q in range(n)]
    assert all(m is not None for m in mats)
    assert len({round(float(m[1, 0]), 6) for m in mats}) > 1, \
        "el canal debe ser no uniforme para que el test tenga poder"

    qc = _sample_circuit(n, seed=777)
    predicted = spec.apply_readout(_diag_rho(qc, spec, n), n)
    observed = _counts_probs(qc, spec, n, shots)
    tol = 5 * 0.5 / np.sqrt(shots) + 1e-3
    assert np.max(np.abs(predicted - observed)) < tol


def test_apply_readout_conserva_la_normalizacion():
    """Un canal estocastico no crea ni destruye probabilidad."""
    rng = np.random.default_rng(3)
    spec = cn.NoiseSpec.from_level('readout', readout_p=0.07)
    for n in (1, 2, 4):
        p = rng.random(2 ** n)
        p /= p.sum()
        out = spec.apply_readout(p, n)
        assert out.shape == p.shape
        assert np.isclose(out.sum(), 1.0)
        assert np.all(out >= 0.0)


def test_apply_readout_acepta_lotes():
    """La ruta por lotes debe coincidir fila a fila con la ruta individual."""
    rng = np.random.default_rng(4)
    spec = cn.NoiseSpec.from_level('readout', readout_p=0.04)
    n = 3
    batch = rng.random((5, 2 ** n))
    batch /= batch.sum(axis=1, keepdims=True)
    out = spec.apply_readout(batch, n)
    assert out.shape == batch.shape
    for k in range(batch.shape[0]):
        assert np.allclose(out[k], spec.apply_readout(batch[k], n))


def test_apply_readout_rechaza_longitud_incorrecta():
    """Un desajuste de ancho debe fallar, no reinterpretar los bits."""
    spec = cn.NoiseSpec.from_level('readout')
    with pytest.raises(ValueError):
        spec.apply_readout(np.ones(8) / 8, 2)


# =============================================================================
# La identidad FAITHFUL sobrevive al cambio de lectura
# =============================================================================

@pytest.mark.qiskit
def test_rho00_reproduce_la_aceptacion_metropolis():
    """rho[0,0] == |psi_0|^2 == min(1, e^Delta), sin ruido.

    Es la afirmacion de fidelidad del proyecto. El eje de ruido cambia COMO se
    lee la aceptacion (de amplitud a rho), asi que hay que demostrar que el
    cambio de lectura no introduce error propio antes de atribuir al ruido
    cualquier desviacion posterior.
    """
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import ParameterVector
    from cosmo_core import make_simulator

    deltas = np.array([-6.0, -3.0, -1.0, -0.3, -0.05, 0.0, 0.5, 2.0])
    a_exact = np.minimum(1.0, np.exp(deltas))
    thetas = 2.0 * np.arccos(np.sqrt(np.clip(a_exact, 1e-12, 1.0)))

    par = ParameterVector('t', 1)
    qc = QuantumCircuit(1)
    qc.ry(par[0], 0)
    qc.save_density_matrix()

    spec = cn.NoiseSpec.from_level('none')
    sim = make_simulator(method='density_matrix')
    isa = transpile(qc, sim)
    res = sim.run([isa.assign_parameters({par[0]: float(t)})
                   for t in thetas]).result()
    rho00 = np.array([
        float(np.real(np.asarray(res.data(k)['density_matrix'])[0, 0]))
        for k in range(len(thetas))])

    assert np.max(np.abs(rho00 - a_exact)) < 1e-12
    assert spec.apply_readout(rho00, 1) is rho00


@pytest.mark.qiskit
def test_la_lectura_sola_no_mueve_rho():
    """El error de lectura es un canal CLASICO: no puede tocar el estado.

    Este test fija la razon de ser del mapa analitico. Si algun dia rho SI se
    moviera con el peldano `readout`, significaria que el ruido de lectura se
    esta aplicando dos veces (una en Aer y otra en `apply_readout`).
    """
    from qiskit import QuantumCircuit, transpile
    from cosmo_core import make_simulator

    qc = _sample_circuit(2, seed=11)
    qc.save_density_matrix()

    ideal = cn.NoiseSpec.from_level('none')
    ro = cn.NoiseSpec.from_level('readout', readout_p=0.05)

    out = []
    for spec in (ideal, ro):
        kwargs = dict(spec.simulator_kwargs(counts_route=False))
        kwargs['method'] = 'density_matrix'      # el ideal usa statevector
        sim = make_simulator(**kwargs)
        rho = np.asarray(sim.run(transpile(qc, sim, optimization_level=0))
                         .result().data(0)['density_matrix'])
        out.append(np.real(np.diag(rho)))
    assert np.max(np.abs(out[0] - out[1])) < 1e-12


@pytest.mark.qiskit
def test_b_recon_no_se_reconstruye_el_modelo():
    """[B-RECON] El modelo calibrado debe llegar a Aer SIN round-trip.

    Reconstruirlo con `NoiseModel.from_dict()` (ademas de estar deprecado
    desde qiskit-aer 0.15) es lossy: contra FakeBrisbane la rho reconstruida
    difiere de la original en ~6e-4 a 3 qubits. Este test fija que el objeto
    que se entrega al simulador es EL MISMO que produjo Aer.
    """
    spec = cn.NoiseSpec.from_level('FakeBrisbane')
    assert spec.full_model() is spec.source_model
    for counts_route in (False, True):
        kwargs = spec.simulator_kwargs(counts_route)
        assert kwargs['noise_model'] is spec.source_model


# =============================================================================
# Rutas de lectura de cosmo_modular_quantum bajo el eje
# =============================================================================

@pytest.mark.qiskit
def test_peldano_none_es_bit_identico_y_reversible():
    """`--noise none` debe seguir dando EXACTAMENTE los numeros publicados.

    Cubre ademas la invalidacion de la caché `_HAD`: tras pasar por peldanos
    ruidosos, volver a `none` tiene que reconstruir el simulador ideal. Si la
    caché no se invalidara, la vuelta seguiria usando el simulador ruidoso y
    los resultados serian del nivel equivocado sin ningun aviso.
    """
    import cosmo_modular_quantum as mq

    lp_cur = np.array([-10.0, -10.0, -10.0, -10.0])
    lp_prop = np.array([-16.0, -10.5, -10.0, -8.0])

    mq.set_noise(cn.NoiseSpec.from_level('none'))
    ref = mq.hadamard_accept_log_batch(lp_cur, lp_prop)

    exact = np.minimum(1.0, np.exp(lp_prop - lp_cur))
    assert np.max(np.abs(np.exp(ref) - exact)) < 1e-11

    # la version por lotes y la individual comparten caché: deben coincidir
    single = np.array([mq.hadamard_accept_log(a, b)
                       for a, b in zip(lp_cur, lp_prop)])
    assert np.array_equal(single, ref)

    for level in ('readout', 'full'):
        mq.set_noise(cn.NoiseSpec.from_level(level))
        mq.hadamard_accept_log_batch(lp_cur, lp_prop)

    mq.set_noise(cn.NoiseSpec.from_level('none'))
    assert np.array_equal(mq.hadamard_accept_log_batch(lp_cur, lp_prop), ref)


@pytest.mark.qiskit
def test_la_lectura_si_mueve_la_aceptacion():
    """La columna `readout` del eje NO puede salir igual que la ideal.

    Es el sintoma exacto del bug [B-RO] y de leer `rho[0,0]` sin aplicar el
    canal en cerrado. Para la aceptacion hay respuesta analitica: con volteo
    simetrico p, P'(0) = (1-p)A + p(1-A), asi que una aceptacion de A=1 debe
    caer a 1-p exactamente.
    """
    import cosmo_modular_quantum as mq

    p = 0.03
    lp_cur = np.array([-10.0])
    lp_prop = np.array([-8.0])            # Delta>0 => A=1 exacta

    mq.set_noise(cn.NoiseSpec.from_level('readout', readout_p=p))
    a_noisy = float(np.exp(mq.hadamard_accept_log_batch(lp_cur, lp_prop)[0]))
    mq.set_noise(cn.NoiseSpec.from_level('none'))

    assert abs(a_noisy - (1.0 - p)) < 1e-9


@pytest.mark.qiskit
def test_invariancia_de_la_propuesta_ante_lectura_uniforme():
    """[INVARIANCIA] La calibracion absorbe el 100% de la lectura uniforme.

    Un canal de lectura simetrico y uniforme reescala <Z_q> por (1-2p), y la
    calibracion a std unitaria divide ese escalar, de modo que el
    desplazamiento calibrado es EXACTAMENTE el mismo. Este test fija esa
    invariancia para que la columna `readout` de la fila de la propuesta,
    que sale identica a la ideal, no se confunda nunca con una regresion.

    El ruido de COMPUERTA si debe sobrevivir: es lo que separa esta
    invariancia legitima del bug [B-RO].
    """
    import cosmo_modular_quantum as mq

    def calibrated_block(spec, n=32):
        mq.set_noise(spec, 'counts')
        mq._reseed(11)
        eng = mq.QuantumProposalEngine(3, batch=n, n_calib=n)
        mq._reseed(99)
        return np.array([eng.next() for _ in range(n)])

    ideal = calibrated_block(cn.NoiseSpec.from_level('none'))
    for p in (0.01, 0.20):
        noisy = calibrated_block(cn.NoiseSpec.from_level('readout',
                                                         readout_p=p))
        assert np.max(np.abs(noisy - ideal)) < 1e-12, \
            f"la lectura uniforme p={p} deberia ser absorbida por completo"

    gates = calibrated_block(cn.NoiseSpec.from_level('full'))
    assert np.max(np.abs(gates - ideal)) > 1e-3, \
        "el ruido de compuerta NO debe ser absorbido por la calibracion"
    mq.set_noise(cn.NoiseSpec.from_level('none'))


@pytest.mark.qiskit
def test_regla_de_conteos_coincide_con_la_del_pipeline_qpu():
    """La ruta por conteos debe ser la MISMA regla que corre en hardware.

    `qpu_cosmo_samplers.py` se mantiene intacto para las corridas de QPU, asi
    que la regla <Z_q> = 1 - 2 P(q=1) esta implementada dos veces. Este test
    las confronta sobre los mismos conteos para que no puedan divergir en
    silencio: una divergencia aqui romperia la comparabilidad simulador-QPU,
    que es el punto de todo el modulo.
    """
    import cosmo_modular_quantum as mq
    from qpu_cosmo_samplers import QPUProposalEngine

    rng = np.random.default_rng(2)
    n_qubits, d, shots = 4, 3, 100_000
    probs = rng.random(2 ** n_qubits)
    probs /= probs.sum()

    counts = {format(i, f'0{n_qubits}b'): int(round(p * shots))
              for i, p in enumerate(probs) if p * shots >= 1}
    total = sum(counts.values())

    # --- regla del modulo QPU (sin construir la clase: no hay conexion) ---
    qpu = QPUProposalEngine.__new__(QPUProposalEngine)
    qpu.n_qubits, qpu.d = n_qubits, d
    z_qpu = qpu._counts_to_shift(counts)

    # --- regla del simulador, sobre las MISMAS frecuencias ---
    freq = np.zeros(2 ** n_qubits)
    for bits, c in counts.items():
        freq[int(bits, 2)] = c / total
    idx = np.arange(2 ** n_qubits)
    z_sim = np.array([1.0 - 2.0 * float(freq @ ((idx >> q) & 1))
                      for q in range(d)])

    assert np.allclose(z_qpu, z_sim, atol=1e-12), (
        f"las dos implementaciones de <Z_q> divergieron: "
        f"QPU={z_qpu}, simulador={z_sim}")


# =============================================================================
# El eje de ruido como dimension de tarea del runner HPC
# =============================================================================

def _runner_args(extra):
    """Namespace del runner con el atributo `profile` ya resuelto."""
    import cosmo_hpc_runner as r
    args = r.build_parser().parse_args(['--models', 'lcdm', 'cpl'] + extra)
    args.profile = not getattr(args, 'no_profile', False)
    return args


def test_runner_sin_ruido_conserva_las_rutas_de_salida():
    """Una corrida ideal debe producir EXACTAMENTE las carpetas de siempre.

    Si el eje anadiera un sufijo 'noise-none' a todas las tareas, los CSV y
    figuras previos dejarian de alinear con los nuevos y toda comparacion
    historica exigiria renombrar a mano. El peldano solo entra en el nombre
    cuando de verdad hay mas de uno.
    """
    import cosmo_hpc_runner as r

    args = _runner_args(['--nqpp', '3'])
    tasks = r.build_tasks(args, '/tmp/m', 18, [], q_ceiling_genetic=24,
                          noise_levels=['none'])
    assert tasks, "deberia generar tareas"
    for t in tasks:
        assert 'noise' not in t.outdir
        assert 'noise' not in t.name
        assert t.noise == 'none'
        assert '--noise' not in t.argv


def test_runner_genera_la_matriz_bidimensional():
    """--noise-sweep debe multiplicar tareas como una dimension mas.

    Es el requisito del diseno: el nivel de ruido se trata igual que nqpp y
    n_bits, de modo que una sola invocacion produzca la matriz completa.
    """
    import cosmo_hpc_runner as r

    levels = ['none', 'readout', 'full']
    # --no-noise-control para aislar la matriz; la columna de control tiene
    # su propio test.
    args = _runner_args(['--nqpp', '3', '--no-noise-control'])
    base = r.build_tasks(args, '/tmp/m', 18, [], q_ceiling_genetic=24,
                         noise_levels=['none'])
    grid = r.build_tasks(args, '/tmp/m', 18, [], q_ceiling_genetic=24,
                         noise_levels=levels, noisy_q_ceiling=13,
                         noisy_q_ceiling_genetic=13)
    assert len(grid) == len(base) * len(levels)
    assert {t.noise for t in grid} == set(levels)
    for t in grid:
        if t.noise != 'none':
            assert '--noise' in t.argv
            assert t.argv[t.argv.index('--noise') + 1] == t.noise
    # nombres unicos: dos peldanos no pueden escribir en la misma carpeta
    assert len({t.outdir for t in grid}) == len(grid)


def test_runner_aplica_el_techo_de_ruido_por_modelo():
    """El techo con ruido debe recortar los modelos pesados, no los ligeros.

    CPL (d=4) con nqpp=4 son 16 qubits: por encima del techo de 13, asi que
    debe bajar. LCDM (d=2) con nqpp=4 son 8 y debe quedarse igual. Es el mismo
    recorte por modelo que ya hacian los otros dos techos.
    """
    import cosmo_hpc_runner as r

    args = _runner_args(['--nqpp', '4'])
    tasks = r.build_tasks(args, '/tmp/m', 18, [], q_ceiling_genetic=24,
                          noise_levels=['full'], noisy_q_ceiling=13,
                          noisy_q_ceiling_genetic=13)
    by_model = {t.model: t for t in tasks if t.grid_kind == 'nqpp'}
    assert by_model['lcdm'].total_qubits == 8      # 2*4, cabe
    assert by_model['cpl'].total_qubits <= 13      # 4*4=16 -> recortado


def test_runner_modelo_de_memoria_ruidoso_es_aditivo():
    """La matriz de densidad se SUMA al modelo de la tarea, no lo sustituye.

    Una tarea de samplers con ruido sigue construyendo su grid de
    verosimilitud ademas de rho; estimar solo rho la subestimaria.
    """
    import cosmo_hpc_runner as r
    import cosmo_noise as cnz

    for q in (8, 10, 12):
        ideal = r.estimate_qubits_and_mem(q, 'nqpp', noisy=False)
        noisy = r.estimate_qubits_and_mem(q, 'nqpp', noisy=True)
        assert noisy > ideal
        factor = cnz.param_shift_batch_factor(q)
        assert abs((noisy - ideal)
                   - cnz.noisy_density_bytes(q, factor) / 1e6) < 1e-6


def test_runner_techo_de_ruido_escala_con_ram_pero_recorta():
    """[REV] El techo ruidoso crece con la RAM, y sigue muy por debajo del ideal.

    Dos cosas a la vez: que dar mas RAM conceda mas qubits (lo que la version
    anterior negaba), y que aun asi el eje de ruido recorte fuerte respecto a
    la corrida ideal — a 95 GB, 12 qubits contra los 18+ que el statevector
    permite comodamente.
    """
    import cosmo_hpc_runner as r

    c_small = r.qubit_ceiling(None, 7 * 1024, 'nqpp', noisy=True)
    c_big = r.qubit_ceiling(None, 512 * 1024, 'nqpp', noisy=True)
    assert c_big > c_small, "mas RAM debe conceder mas qubits"

    mem95 = 95 * 1024
    assert r.qubit_ceiling(None, mem95, 'nqpp', noisy=True) < \
        r.qubit_ceiling(None, mem95, 'nqpp'), \
        "el eje de ruido debe seguir recortando respecto al ideal"

    # el genetico afloja mas: no tiene lote de parameter-shift
    assert r.qubit_ceiling(None, mem95, 'n_bits', noisy=True) > \
        r.qubit_ceiling(None, mem95, 'nqpp', noisy=True)


# =============================================================================
# Gemelo ruidoso del pipeline QPU
# =============================================================================

@pytest.mark.qiskit
def test_gemelo_no_modifica_el_modulo_de_hardware():
    """`qpu_cosmo_samplers` debe quedar INTACTO tras usar el gemelo.

    Ese modulo es el que se manda a hardware real. El gemelo lo parchea en
    caliente para sustituir la conexion, asi que lo unico que separa "reusar
    el pipeline" de "corromperlo para el resto del proceso" es que el parche
    se deshaga. Este test fija esa restauracion.
    """
    import qpu_cosmo_samplers as hw
    import qpu_noisy_simulation as twin

    original = hw.QPUConnection
    rc = twin.main(['--model', 'lcdm', '--method', 'qmcmc', '--steps', '20',
                    '--chains', '2', '--block', '8', '--shots', '256',
                    '--noise', 'none', '--outdir', '/tmp/_twin_restore',
                    '--no-plot'])
    assert rc == 0
    assert hw.QPUConnection is original


@pytest.mark.qiskit
def test_gemelo_respeta_la_interfaz_de_run_pub():
    """La conexion local debe cumplir el contrato de la de hardware.

    B bindings -> B diccionarios de conteos, en orden, con las frecuencias
    sumando los disparos pedidos. Si este contrato se rompe, el pipeline lo
    consumiria igual y produciria desplazamientos silenciosamente mal
    ordenados.
    """
    import qpu_cosmo_samplers as hw
    import qpu_noisy_simulation as twin

    spec = cn.NoiseSpec.from_level('full')
    conn = twin.LocalNoisyConnection(noise=spec, shots=512, seed=3)

    qc = hw.build_proposal_circuit(3, n_layers=2)
    isa = conn.transpile_isa(qc)
    b = 5
    rng = np.random.default_rng(0)
    pv = rng.uniform(0, 2 * np.pi, size=(b, isa.num_parameters))

    out = conn.run_pub(isa, pv, shots=512)
    assert isinstance(out, list) and len(out) == b
    for counts in out:
        assert isinstance(counts, dict)
        assert sum(counts.values()) == 512
        assert all(len(k.replace(' ', '')) == 3 for k in counts)
    conn.close()


@pytest.mark.qiskit
def test_gemelo_rechaza_anchos_por_encima_del_techo():
    """Un ancho inviable debe fallar ANTES de empezar, no tras horas.

    CPL con nqpp=6 son 24 qubits: la matriz de densidad serian ~4 PB. El
    gemelo tiene que detectarlo y salir con codigo 1.
    """
    import qpu_noisy_simulation as twin

    rc = twin.main(['--model', 'cpl', '--method', 'qvmc', '--iters', '1',
                    '--nqpp', '6', '--noise', 'full',
                    '--outdir', '/tmp/_twin_cap', '--no-plot'])
    assert rc == 1


def test_gemelo_filtra_los_flags_del_eje_de_ruido():
    """El parser de hardware no conoce --noise*: hay que quitarlos del argv.

    Si se colaran, `qpu.main` abortaria con "unrecognized arguments" y el
    gemelo seria inusable con cualquier peldano explicito.
    """
    import qpu_noisy_simulation as twin

    argv = ['--model', 'lcdm', '--noise', 'full', '--steps', '10',
            '--noise-readout-p', '0.05', '--noise-seed', '7', '--dry-run']
    out = twin._strip_noise_flags(argv)
    assert out == ['--model', 'lcdm', '--steps', '10', '--dry-run']
    # forma --flag=valor
    assert twin._strip_noise_flags(['--noise=full', '--model', 'lcdm']) == \
        ['--model', 'lcdm']


@pytest.mark.parametrize('flag,expected', [([], True), (['--no-noise-control'], False)])
def test_runner_genera_la_columna_de_control(flag, expected):
    """[NOISE-CONTROL] Barrer ruido debe traer la columna ideal-por-conteos.

    Sin ella, el peldano ideal lee amplitudes y los ruidosos leen conteos, asi
    que la comparacion mezcla el ruido con el cambio de operador. En la
    primera corrida del eje eso hacia parecer que el ruido de lectura mejora
    el muestreo (sigma 0.0208 -> 0.0176, ESS 101 -> 141), cuando en realidad
    no toca la propuesta en absoluto.

    Por eso el control se genera SOLO, y desactivarlo tiene que ser explicito.
    """
    import cosmo_hpc_runner as r

    args = _runner_args(['--noise-sweep', 'none,readout,full'] + flag)
    tasks = r.build_tasks(args, '/tmp/m', 18, [], q_ceiling_genetic=24,
                          noise_levels=['none', 'readout', 'full'],
                          noisy_q_ceiling=13, noisy_q_ceiling_genetic=13)
    control = [t for t in tasks if t.noise == 'none-counts']
    assert bool(control) is expected
    if expected:
        # una columna de control POR MODELO, igual que cualquier otra columna
        assert {t.model for t in control} == {'lcdm', 'cpl'}
        assert len(control) == 2
        for t in control:
            assert t.script == 'cosmo_modular_quantum.py', \
                "el control es de la propuesta; el QGA no tiene ruta conmutable"
            assert '--proposal-route' in t.argv
            assert t.argv[t.argv.index('--proposal-route') + 1] == 'counts'
            assert t.argv[t.argv.index('--noise') + 1] == 'none'


def test_runner_control_no_contamina_una_corrida_ideal():
    """Sin barrido de ruido no debe aparecer ninguna columna extra.

    Una corrida ideal tiene que seguir produciendo exactamente las mismas
    tareas y rutas que antes del eje.
    """
    import cosmo_hpc_runner as r

    args = _runner_args([])
    tasks = r.build_tasks(args, '/tmp/m', 18, [], q_ceiling_genetic=24,
                          noise_levels=['none'])
    assert all(t.noise == 'none' for t in tasks)
    assert all('noise' not in t.outdir for t in tasks)
    assert all('--proposal-route' not in t.argv for t in tasks)


# =============================================================================
# [B-PLAN] Planificar sin qiskit
# =============================================================================

def test_b_plan_formula_del_ansatz_coincide_con_el_circuito():
    """[B-PLAN] El respaldo sin qiskit debe dar EXACTAMENTE lo mismo.

    `ansatz_n_params` mide el circuito real cuando qiskit esta, y usa
    `n*(2L+1)` cuando no — porque construir el circuito arrastraba qiskit al
    runner, que hasta entonces solo planificaba tareas y lanzaba subprocesos.
    Si las dos rutas divergieran, el runner planificaria con un techo de qubits
    distinto del que la corrida necesita de verdad, y la tarea moriria por OOM
    a mitad de campana.
    """
    pytest.importorskip('qiskit')
    from qpu_cosmo_samplers import build_ansatz

    for layers in (1, 2, 3, 4, 5):
        for n in (2, 3, 5, 6, 9, 13, 18):
            assert build_ansatz(n, layers).num_parameters == \
                n * (2 * layers + 1), \
                f"la formula de respaldo diverge en n={n}, L={layers}"
            assert cn.ansatz_n_params(n, layers) == n * (2 * layers + 1)


def test_b_plan_el_respaldo_no_necesita_qiskit(monkeypatch):
    """El respaldo debe activarse si el import falla, no propagar el error."""
    import builtins
    real_import = builtins.__import__

    def blocked(name, *a, **kw):
        if name.startswith(('qpu_cosmo_samplers', 'qiskit')):
            raise ImportError('simulado: nodo sin qiskit')
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, '__import__', blocked)
    assert cn.ansatz_n_params(12, 3) == 12 * 7
    assert cn.param_shift_batch_factor(12) == 2 * 12 * 7


@pytest.mark.qiskit
def test_b_gpu_el_diagnostico_no_revienta_sin_gpu():
    """[B-GPU] El diagnostico debe informar, nunca romper la corrida.

    Se invoca justo cuando algo ya va mal (se pidio GPU y no la hay); si el
    propio diagnostico lanzara, se llevaria por delante la corrida que iba a
    salvar.
    """
    from cosmo_core import gpu_diagnosis, resolve_device

    txt = gpu_diagnosis()
    assert isinstance(txt, str) and 'GPU' in txt
    assert 'qiskit-aer' in txt or 'no pude consultar' in txt
    # degradar sin GPU nunca debe levantar, con o sin aviso
    assert resolve_device(True, warn=False) in ('CPU', 'GPU')
    assert resolve_device(True, warn=True) in ('CPU', 'GPU')
    assert resolve_device(False) == 'CPU'


# =============================================================================
# [B-CGROUP] Limites del contenedor
# =============================================================================

def test_b_cgroup_lee_el_limite_del_contenedor(tmp_path, monkeypatch):
    """[B-CGROUP] En un contenedor hay que leer el cgroup, no el nodo.

    `psutil.virtual_memory()` y `os.cpu_count()` reportan los recursos del
    NODO. En un pod con "Maximum memory 63Gi" sobre un nodo de 512 GB, el
    runner creia tener 512 GB, admitia decenas de tareas a la vez y el runtime
    mataba el pod por OOM — un SIGKILL, sin traceback ni resultados parciales.
    """
    import cosmo_hpc_runner as r

    v2 = tmp_path / 'v2'
    v2.mkdir()
    (v2 / 'memory.max').write_text('67645734912\n')      # 63 GiB
    (v2 / 'cpu.max').write_text('1500000 100000\n')      # 15 nucleos

    real_open = open

    def fake_open(path, *a, **kw):
        p = str(path)
        if p.startswith('/sys/fs/cgroup/'):
            return real_open(str(v2 / os.path.basename(p)), *a, **kw)
        return real_open(path, *a, **kw)

    monkeypatch.setattr('builtins.open', fake_open)
    assert abs(r._cgroup_memory_limit_mb() - 67645.7) < 1.0
    assert r._cgroup_cpu_limit() == 15
    assert r.detected_cores() == 15
    assert abs(r.detected_memory_mb() - 67645.7) < 1.0


def test_b_cgroup_sin_limite_cae_al_nodo(monkeypatch):
    """Sin cgroup (o con 'max'), debe comportarse como antes del arreglo."""
    import cosmo_hpc_runner as r

    real_open = open

    def no_cgroup(path, *a, **kw):
        if str(path).startswith('/sys/fs/cgroup/'):
            raise FileNotFoundError(path)
        return real_open(path, *a, **kw)

    monkeypatch.setattr('builtins.open', no_cgroup)
    assert r._cgroup_memory_limit_mb() is None
    assert r._cgroup_cpu_limit() is None
    assert r.detected_cores() >= 1
    assert r.detected_memory_mb() > 0


def test_b_cgroup_centinela_de_ilimitado(monkeypatch, tmp_path):
    """El centinela gigante de cgroup v1 no debe leerse como un limite real.

    cgroup v1 escribe 2^63-1 para decir "sin limite". Tomarlo al pie de la
    letra daria un presupuesto de 9 millones de TB.
    """
    import cosmo_hpc_runner as r

    d = tmp_path / 'v1'
    d.mkdir()
    (d / 'memory.limit_in_bytes').write_text('9223372036854771712\n')
    real_open = open

    def fake_open(path, *a, **kw):
        p = str(path)
        if p == '/sys/fs/cgroup/memory.max':
            raise FileNotFoundError(p)
        if p.startswith('/sys/fs/cgroup/'):
            return real_open(str(d / os.path.basename(p)), *a, **kw)
        return real_open(path, *a, **kw)

    monkeypatch.setattr('builtins.open', fake_open)
    assert r._cgroup_memory_limit_mb() is None


# =============================================================================
# [B-GRID] La comparacion de ruido no puede mezclar resoluciones
# =============================================================================

def _fake_run(tmp_path, folders):
    """Arma un master_dir falso con un CSV minimo por carpeta."""
    hdr = ("Method,Om_mean,Om_std,H0_mean,H0_std,nqpp,chi2_red,final_KL,ESS\n")
    for name, kl in folders.items():
        d = tmp_path / name
        d.mkdir(parents=True)
        (d / 'resultados_config.csv').write_text(
            hdr
            + f"Classical MCMC,0.26,0.016,70.8,1.09,—,0.56,,101\n"
            + f"QVMC 67%,0.26,0.017,70.8,1.10,3,0.56,{kl},91\n")
    return str(tmp_path)


def test_b_grid_no_mezcla_resoluciones(tmp_path):
    """[B-GRID] Con --nqpp-sweep y --noise-sweep juntos, una figura por nqpp.

    El techo con ruido recorta unas resoluciones y no otras, asi que la
    columna ideal puede quedarse con nqpp=5 mientras la ruidosa baja a nqpp=3.
    Comparandolas en el mismo eje, la figura atribuiria al ruido lo que es un
    cambio de resolucion — y la serie no avisaba: tomaba `sel[-1]`, una fila
    arbitraria entre las varias que compartian metodo y peldano.
    """
    import cosmo_hpc_runner as r

    master = _fake_run(tmp_path, {
        'samplers_lcdm_nqpp3_noise-none': 10.0,
        'samplers_lcdm_nqpp3_noise-full': 11.5,
        'samplers_lcdm_nqpp5_noise-none': 8.0,
        'samplers_lcdm_nqpp5_noise-full': 9.2,
    })
    made = r.generate_noise_comparison_plots(master)
    names = {os.path.basename(p) for p in made}
    assert names == {'noise_comparison_nqpp3_lcdm.png',
                     'noise_comparison_nqpp5_lcdm.png'}, names


def test_b_grid_una_sola_resolucion_conserva_el_nombre(tmp_path):
    """Sin barrido, el nombre y el titulo se quedan como siempre."""
    import cosmo_hpc_runner as r

    master = _fake_run(tmp_path, {
        'samplers_lcdm_noise-none': 10.0,
        'samplers_lcdm_noise-full': 11.5,
    })
    made = r.generate_noise_comparison_plots(master)
    assert [os.path.basename(p) for p in made] == \
        ['noise_comparison_lcdm.png']


def test_b_grid_una_carpeta_sin_etiqueta_es_UN_grupo(tmp_path):
    """Un CSV sin etiqueta de carpeta no debe partirse en dos grupos.

    Deducir la resolucion fila a fila de la columna `nqpp` lo hacia: el MCMC
    clasico la deja vacia, asi que el mismo CSV generaba un grupo '' y otro
    'nqpp3', y salian dos figuras con la mitad de los metodos cada una.
    """
    import cosmo_hpc_runner as r

    assert r._infer_grid('/x/samplers_lcdm_noise-none/r.csv',
                         {'nqpp': '3'}) == ''
    assert r._infer_grid('/x/samplers_lcdm_noise-none/r.csv',
                         {'nqpp': ''}) == ''
    assert r._infer_grid('/x/samplers_lcdm_nqpp5_noise-full/r.csv',
                         {}) == 'nqpp5'
    assert r._infer_grid('/x/genetic_cpl_nb4_noise-full/r.csv', {}) == 'nb4'
