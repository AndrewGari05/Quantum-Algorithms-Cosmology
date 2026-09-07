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
import inspect
import tempfile

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

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


# =============================================================================
# [B-OFFSET] El eje de chi2 no puede mentir con notacion de offset
# =============================================================================

@pytest.mark.qiskit
def test_b_offset_el_eje_de_chi2_muestra_valores_reales():
    """[B-OFFSET] Sin offset: un chi2 de 1100 debe leerse como 1100.

    Los rungs convergen a chi2 casi identicos, asi que el rango del eje es de
    decimas sobre un valor de cuatro cifras. Ante eso matplotlib etiqueta
    0.0, 0.1, 0.2... y esconde un '+1.1e3' en una esquina: la figura APARENTA
    un chi2 entre 0 y 1 y un lector razonable concluye que esta viendo el
    reducido, o que el ajuste es absurdamente bueno.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cosmo_genetic_optimizers as ge

    gens = np.arange(40)
    fig, ax = plt.subplots()
    for off in (0.42, 0.45, 0.51):
        ax.plot(gens, 1100.0 + off - 0.3 * np.exp(-gens / 5.0))

    class _R:
        stats = {'n_data': 1099, 'chi2_red': 1.001}

    ge._chi2_axis(ax, [_R()], plt)
    fig.canvas.draw()
    assert ax.get_yaxis().get_offset_text().get_text() == '', \
        "el eje sigue usando notacion de offset"
    ticks = [t.get_text().replace('−', '-')
             for t in ax.get_yticklabels() if t.get_text()]
    vals = []
    for t in ticks:
        try:
            vals.append(float(t))
        except ValueError:
            pass
    assert vals and min(vals) > 1000, \
        f"las etiquetas deberian rondar 1100, son {ticks}"
    # y el eje debe decir que es crudo, con cuantos datos
    lab = ax.get_ylabel()
    assert 'not reduced' in lab and '1099' in lab
    plt.close(fig)


@pytest.mark.qiskit
def test_b_overlap_las_etiquetas_de_valor_no_se_pisan():
    """[B-OVERLAP] Etiquetas superpuestas son peor que no ponerlas.

    Las curvas de chi2 de los rungs acaban a milesimas unas de otras, asi que
    la etiqueta directa del valor final es lo unico que las separa sin
    ambiguedad — pero si las etiquetas se solapan el numero queda ilegible y
    parece un valor que no es.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cosmo_genetic_optimizers as ge

    fig, ax = plt.subplots()
    ax.set_ylim(27.4, 29.0)
    vals = [27.469, 27.469, 27.475, 27.680, 28.079]
    items = []
    for v in vals:
        ann = ax.annotate(f"{v:.3f}", xy=(30, v), xytext=(4, 0),
                          textcoords='offset points', va='center')
        items.append((v, ann))
    fig.canvas.draw()
    ge._spread_labels(ax, items)
    fig.canvas.draw()

    # posiciones efectivas en datos, tras el desplazamiento
    lo, hi = ax.get_ylim()
    ys = sorted(v + ann.get_position()[1] / ax.bbox.height * (hi - lo)
                for v, ann in items)
    gaps = np.diff(ys)
    assert np.all(gaps > 0.02 * (hi - lo)), \
        f"quedaron etiquetas pisadas: separaciones {gaps}"
    # y el punto al que apuntan NO se movio: la figura no miente
    assert sorted(a.xy[1] for _, a in items) == sorted(vals)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# [B-MEM] El modelo de memoria de los samplers debe depender de N_data
# ─────────────────────────────────────────────────────────────────────────────
#
# Origen: campana CC+BAO+Pantheon del 2026-08-31. El planificador estimo
# 3.7 GB para 18 qubits y se midieron 17.9-19.6 GB. Tres tareas de 20 y 21
# qubits llegaron al OOM killer (rc=-9). La causa no era calibracion sino
# forma: la constante 1660*8 no dependia de N_data, cuando el termino que
# domina son arreglos de forma (n_states, N_data).

#: Mediciones reales de peak RSS (MB) de dos campanas, misma maquina, mismo
#: codigo, subproceso limpio. (dataset_n_data, total_qubits, peak_rss_mb)
MEDICIONES_RSS_SAMPLERS = [
    (1099, 12, 520.9), (1099, 14, 1156.5), (1099, 15, 2555.6),
    (1099, 16, 4822.1), (1099, 18, 19585.8),
    (51, 16, 999.8), (51, 18, 4035.5),
]


def test_b_mem_estimacion_cubre_todas_las_mediciones_reales():
    """[B-MEM] La estimacion nunca puede quedar por debajo de lo medido.

    Equivocarse por abajo es un SIGKILL sin traceback ni resultados
    parciales; por arriba es una tarea menos en paralelo. El test exige
    cobertura estricta y ademas acota la sobreestimacion, para que "seguro"
    no degenere en "inutilmente conservador".
    """
    r = pytest.importorskip('cosmo_hpc_runner')
    for n_data, q, medido in MEDICIONES_RSS_SAMPLERS:
        est = r.estimate_qubits_and_mem(q, 'nqpp', n_data=n_data)
        assert est >= medido, (
            f"SUBestimacion en N_data={n_data}, {q}q: estimado {est:.0f} MB "
            f"< medido {medido:.0f} MB -> riesgo de OOMKill")
        assert est <= 2.0 * medido, (
            f"sobreestimacion excesiva en N_data={n_data}, {q}q: "
            f"{est:.0f} MB frente a {medido:.0f} MB medidos")


def test_b_mem_la_constante_vieja_habria_fallado():
    """[B-MEM] Prueba de regresion: el modelo viejo SI subestimaba.

    Sin esto el test de arriba pasaria tambien con un modelo que acertara por
    casualidad. Aqui se fija que el fallo concreto que se corrigio existia.
    """
    r = pytest.importorskip('cosmo_hpc_runner')
    viejo_por_estado = 1660 * 8          # la constante original
    peores = [(n, q, m) for n, q, m in MEDICIONES_RSS_SAMPLERS if q >= 15]
    assert peores
    for n_data, q, medido in peores:
        viejo = (2 ** q) * viejo_por_estado / 1e6 + r.PROCESS_BASELINE_MB
        nuevo = r.estimate_qubits_and_mem(q, 'nqpp', n_data=n_data)
        if n_data >= 1000:
            assert viejo < medido, "la mediicon deberia exponer el modelo viejo"
        assert nuevo > viejo


def test_b_mem_depende_del_dataset_y_solo_en_samplers():
    """[B-MEM] N_data mueve el coste de los samplers, no el del genetico."""
    r = pytest.importorskip('cosmo_hpc_runner')
    chico = r.estimate_qubits_and_mem(16, 'nqpp', n_data=51)
    grande = r.estimate_qubits_and_mem(16, 'nqpp', n_data=1099)
    assert grande > 3 * chico, (
        "el coste por estado debe crecer con N_data; si no, el termino "
        "(n_states, N_data) no esta en el modelo")
    # El QGA evalua la verosimilitud sobre la POBLACION, no sobre la rejilla.
    g1 = r.estimate_qubits_and_mem(20, 'n_bits', n_data=51)
    g2 = r.estimate_qubits_and_mem(20, 'n_bits', n_data=1099)
    assert g1 == g2


def test_b_mem_dataset_desconocido_es_conservador():
    """[B-MEM] Un dataset que no esta en la tabla usa el mayor, no el menor."""
    r = pytest.importorskip('cosmo_hpc_runner')
    assert r.dataset_n_data('CC+BAO') == 51
    assert r.dataset_n_data(None) == r.DEFAULT_PLAN_N_DATA
    assert r.dataset_n_data('CC+BAO+DESI+Union3') == r.DEFAULT_PLAN_N_DATA
    assert r.DEFAULT_PLAN_N_DATA == max(r.DATASET_N_DATA.values())


def test_b_mem_las_tareas_que_murieron_ahora_se_rechazan():
    """[B-MEM] Las tres configuraciones que dieron rc=-9 no deben planificarse.

    cpl/nqpp5 = 20 q, gede/nqpp7 y wcdm/nqpp7 = 21 q, en un contenedor de
    63 GiB. Con el modelo viejo el techo las admitia.
    """
    r = pytest.importorskip('cosmo_hpc_runner')
    techo = r.qubit_ceiling(None, 63 * 1024, 'nqpp', n_data=1099)
    assert techo < 20, (
        f"el techo ({techo} q) sigue admitiendo las tareas que fueron "
        f"OOMKilled a 20 y 21 qubits")
    # y con CC+BAO, que es 20x mas chico, el mismo nodo concede mas
    assert r.qubit_ceiling(None, 63 * 1024, 'nqpp', n_data=51) > techo


# ─────────────────────────────────────────────────────────────────────────────
# [B-GAPLOT] La figura del genetico y el CSV deben reportar el MISMO numero
# ─────────────────────────────────────────────────────────────────────────────

def test_b_gaplot_barra_de_la_figura_coincide_con_el_csv():
    """[B-GAPLOT] Misma estadistica en la tabla y en la figura.

    La figura dibujaba `theta_map` con la desviacion SIN pesos de la poblacion
    final; el CSV reporta la media y la desviacion PONDERADAS por fitness. En
    lcdm/nb6 eso daba +-0.0165 en la figura contra +-0.0014 en la tabla, doce
    veces mas ancha, sin nada que avisara cual era cual.
    """
    ge = pytest.importorskip('cosmo_genetic_optimizers')
    rng = np.random.default_rng(0)
    pop = rng.normal(0.28, 0.02, size=(400, 2))
    pop[:, 1] = rng.normal(69.6, 1.5, size=400)
    # fitness que concentra el peso en un nucleo estrecho, como el elitismo
    fit = -0.5 * (((pop[:, 0] - 0.276) / 0.0015) ** 2
                  + ((pop[:, 1] - 69.60) / 0.11) ** 2)
    w = ge._fitness_weights(fit)

    class _R:
        method, quantumness, label = 'CGA', 0.0, 'CGA'
        theta_map = np.array([0.2763, 69.5949])
        chi2_map, stats, elapsed, config = 0.0, {}, 0.0, {}
        final_pop, final_fit, final_weights = pop, fit, w
        history = [{'gen': g, 'theta_best': np.array([0.2763, 69.5949]),
                    'best_chi2': 1.0, 'mean_chi2': 2.0} for g in range(5)]
        pop_history, fit_history = [], []

    mu_csv = np.average(pop, weights=w, axis=0)
    sd_csv = np.sqrt(np.average((pop - mu_csv) ** 2, weights=w, axis=0))
    sd_sin_pesos = pop.std(axis=0)
    # el escenario tiene que ser el del bug, si no el test no prueba nada
    assert (sd_sin_pesos > 5 * sd_csv).all()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    class _M:
        name, label = 'lcdm', 'Flat LCDM'
        n_params = 2
        param_names = ['Om', 'H0']
        param_latex = [r'$\Omega_m$', r'$H_0$']
        fiducial = np.array([0.3111, 67.66])

    with tempfile.TemporaryDirectory() as td:
        ge.plot_genetic_convergence([_R()], _M(), td)
        fig = plt.gcf()
    # recuperar las barras dibujadas de la figura ya cerrada no es posible, asi
    # que se comprueba la estadistica que el codigo de la figura calcula ahora,
    # replicandola: media y desviacion ponderadas, identicas a las del CSV.
    finite = np.isfinite(fit)
    mu_fig = np.average(pop[finite], weights=w[finite], axis=0)
    sd_fig = np.sqrt(np.average((pop[finite] - mu_fig) ** 2,
                                weights=w[finite], axis=0))
    assert np.allclose(mu_fig, mu_csv)
    assert np.allclose(sd_fig, sd_csv)
    plt.close('all')


def test_b_gaplot_el_pie_no_llama_intervalo_a_la_dispersion():
    """[B-GAPLOT] La barra del genetico NO es un intervalo de credibilidad.

    Es dispersion de convergencia del optimizador. Confundirlas es como se
    llega a "el QGA difiere del CGA en 9 sigma" cuando en unidades del sigma
    del MCMC la diferencia es 0.16.
    """
    ge = pytest.importorskip('cosmo_genetic_optimizers')
    src = inspect.getsource(ge.plot_genetic_convergence)
    assert 'NOT a credible interval' in src
    assert 'fitness-weighted' in src


# ─────────────────────────────────────────────────────────────────────────────
# [B-TIME] Al genetico con ruido lo acota el reloj, no la RAM
# ─────────────────────────────────────────────────────────────────────────────
#
# Origen: campana 2026-09-03. El techo del genetico ruidoso se derivaba solo
# de la memoria, que a 14 qubits concede sin problema (una rho son 4.3 GB).
# Pero a 14 qubits el QGA tarda ~7.7 h POR GENERACION, o sea ~4 meses las 500
# que se pidieron. La tarea quedo colgada y ademas bloqueo las figuras de
# resumen de la corrida entera, porque el runner no las escribe hasta que
# todas las tareas terminan.

#: Ritmo medido en los logs de esa campana (pop=500, readout, nodo de 63 GiB).
MEDICIONES_SEG_POR_GEN_QGA = [(10, 73.0), (12, 1425.0)]


def test_b_time_el_modelo_reproduce_las_mediciones():
    """[B-TIME] La extrapolacion tiene que pasar por los puntos medidos."""
    for n, seg in MEDICIONES_SEG_POR_GEN_QGA:
        est = cn.genetic_noisy_seconds_per_gen(n)
        assert abs(est - seg) / seg < 0.05, (
            f"{n}q: modelo {est:.0f} s/gen contra {seg:.0f} s medidos")


def test_b_time_extrapolacion_a_14q_coincide_con_lo_observado():
    """[B-TIME] Validacion fuera de la muestra.

    El modelo se ajusta con 10 y 12 qubits. La tercera observacion — a 14
    qubits no habia registrado ni la generacion 0 tras 7 h — no entra en el
    ajuste, asi que sirve para comprobarlo: el modelo debe predecir mas de 7 h
    por generacion, y no un valor absurdo.
    """
    s14 = cn.genetic_noisy_seconds_per_gen(14)
    assert s14 > 7 * 3600, f"predice {s14/3600:.1f} h/gen; se observo >7 h"
    assert s14 < 24 * 3600, f"predice {s14/3600:.1f} h/gen, implausible"


def test_b_time_el_techo_corta_la_celda_que_nunca_termino():
    """[B-TIME] 14 qubits con 500 generaciones no puede entrar en el plan."""
    techo = cn.genetic_noisy_time_ceiling(500, cn.DEFAULT_NOISY_TASK_HOURS)
    assert techo < 14, (
        f"el techo ({techo} q) sigue admitiendo la celda de ~4 meses")
    # y el presupuesto tiene que MANDAR sobre lo que la RAM concederia
    solo_ram = cn.noisy_qubit_ceiling(mem_mb=95 * 1024, quantum_training=False)
    con_tiempo = cn.noisy_qubit_ceiling(mem_mb=95 * 1024,
                                        quantum_training=False,
                                        generations=500)
    assert con_tiempo < solo_ram, (
        "con generaciones dadas, el techo temporal debe ser mas restrictivo "
        "que el de memoria en un nodo grande")


def test_b_time_es_un_presupuesto_no_una_constante():
    """[B-TIME] Mas presupuesto o menos generaciones conceden mas qubits.

    Esto es lo que lo distingue del MAX_NOISY_QUBITS=13 duro que se quito en
    [REV]: aquel no se movia con nada.
    """
    assert (cn.genetic_noisy_time_ceiling(60, 48)
            > cn.genetic_noisy_time_ceiling(500, 48))
    assert (cn.genetic_noisy_time_ceiling(60, 480)
            > cn.genetic_noisy_time_ceiling(60, 48))
    # y nunca devuelve 0, que produciria un plan vacio sin explicacion
    assert cn.genetic_noisy_time_ceiling(10 ** 6, 0.001) >= 1


def test_b_time_no_toca_ni_los_samplers_ni_las_tareas_ideales():
    """[B-TIME] El techo temporal es SOLO del genetico con ruido.

    Los samplers los acota el lote de parameter-shift (memoria) y las tareas
    sin ruido no construyen matriz de densidad en absoluto.
    """
    r = pytest.importorskip('cosmo_hpc_runner')
    mem = 63 * 1024
    # samplers: pasar generations no puede cambiar nada
    a = r.qubit_ceiling(None, mem, 'nqpp', noisy=True, n_data=1099)
    b = r.qubit_ceiling(None, mem, 'nqpp', noisy=True, n_data=1099,
                        generations=500)
    assert a == b
    # genetico ideal: tampoco
    c = r.qubit_ceiling(None, mem, 'n_bits', noisy=False)
    d = r.qubit_ceiling(None, mem, 'n_bits', noisy=False, generations=500)
    assert c == d
    # genetico ruidoso: si
    e = r.qubit_ceiling(None, mem, 'n_bits', noisy=True, generations=500)
    assert e < c


def test_b_time_flag_expuesto_en_el_cli():
    """[B-TIME] El presupuesto tiene que poder cambiarse sin editar codigo."""
    r = pytest.importorskip('cosmo_hpc_runner')
    p = r.build_parser()
    ns = p.parse_args(['--noisy-task-hours', '12'])
    assert ns.noisy_task_hours == 12.0
    assert p.parse_args([]).noisy_task_hours == cn.DEFAULT_NOISY_TASK_HOURS


# ─────────────────────────────────────────────────────────────────────────────
# [B-NOSTATE] El estado del genetico tiene que sobrevivir al proceso
# ─────────────────────────────────────────────────────────────────────────────
#
# Origen: al corregir [B-GAPLOT] se descubrio que las figuras del genetico no
# se pueden regenerar, porque la poblacion final solo vivia en memoria. Para
# arreglar el color de una curva habia que repetir la campana entera — dias de
# computo por un cambio cosmetico.

def _ga_result_sintetico(seed=0, n=200, d=2, gens=6):
    """Un GAResult con datos realistas, sin correr la optimizacion."""
    ge = pytest.importorskip('cosmo_genetic_optimizers')
    rng = np.random.default_rng(seed)
    pop = np.column_stack([rng.normal(0.28, 0.02, n), rng.normal(69.6, 1.5, n)])
    fit = -0.5 * (((pop[:, 0] - 0.276) / 0.0015) ** 2
                  + ((pop[:, 1] - 69.60) / 0.11) ** 2)
    hist = [{'gen': g, 'theta_best': np.array([0.2763 + 1e-5 * g, 69.59]),
             'best_chi2': 1064.6 - 0.01 * g, 'mean_chi2': 1100.0 - g}
            for g in range(gens)]
    return ge.GAResult(
        method='QGA', quantumness=67.0, theta_map=np.array([0.2763, 69.5949]),
        chi2_map=1064.61, stats={'chi2': 1064.61, 'n_data': 1099},
        final_pop=pop, final_fit=fit,
        final_weights=ge._fitness_weights(fit), history=hist,
        elapsed=12.5, config={'mutation': 'quantum'}, label='QGA (q=67%)',
        pop_history=[pop + 0.001 * g for g in range(gens)],
        fit_history=[fit for _ in range(gens)])


def test_b_nostate_ida_y_vuelta_exacta():
    """[B-NOSTATE] Lo que se guarda es lo que se lee, sin perdida donde importa.

    La poblacion final, su fitness y sus pesos alimentan las estadisticas que
    van al CSV, asi que tienen que volver EXACTAS (float64). La historia de
    poblaciones solo alimenta la animacion y se guarda en float32 a proposito.
    """
    ge = pytest.importorskip('cosmo_genetic_optimizers')
    core = pytest.importorskip('cosmo_core')
    orig = [_ga_result_sintetico(seed=s, ) for s in (0, 1)]
    with tempfile.TemporaryDirectory() as td:
        p = ge.save_ga_state(orig, core.MODELS['lcdm'], td)
        assert os.path.exists(p)
        back, meta = ge.load_ga_state(p)

    assert len(back) == len(orig)
    assert meta['model'] == 'lcdm'
    for a, b in zip(orig, back):
        assert b.label == a.label and b.method == a.method
        assert b.quantumness == a.quantumness
        # exactos: de aqui salen los numeros publicados
        assert np.array_equal(a.final_pop, b.final_pop)
        assert np.array_equal(a.final_fit, b.final_fit)
        assert np.array_equal(a.final_weights, b.final_weights)
        assert np.array_equal(a.theta_map, b.theta_map)
        # historia
        assert len(b.history) == len(a.history)
        for ha, hb in zip(a.history, b.history):
            assert ha['gen'] == hb['gen']
            assert np.allclose(ha['theta_best'], hb['theta_best'])
            assert ha['best_chi2'] == pytest.approx(hb['best_chi2'])


def test_b_nostate_la_estadistica_del_csv_se_reconstruye():
    """[B-NOSTATE] Tras el viaje por disco, el CSV saldria identico.

    Es la propiedad que de verdad importa: el estado sirve para redibujar, y
    la figura tiene que seguir coincidiendo con la tabla ([B-GAPLOT]).
    """
    ge = pytest.importorskip('cosmo_genetic_optimizers')
    core = pytest.importorskip('cosmo_core')
    orig = [_ga_result_sintetico(seed=3)]

    def stats(r):
        finite = np.isfinite(r.final_fit)
        pop, w = r.final_pop[finite], np.asarray(r.final_weights)[finite]
        mu = np.average(pop, weights=w, axis=0)
        sd = np.sqrt(np.average((pop - mu) ** 2, weights=w, axis=0))
        return mu, sd

    with tempfile.TemporaryDirectory() as td:
        p = ge.save_ga_state(orig, core.MODELS['lcdm'], td)
        back, _ = ge.load_ga_state(p)
    mu0, sd0 = stats(orig[0])
    mu1, sd1 = stats(back[0])
    assert np.array_equal(mu0, mu1), "la media ponderada cambio al guardar"
    assert np.array_equal(sd0, sd1), "la desviacion ponderada cambio al guardar"


def test_b_nostate_replot_no_necesita_la_optimizacion():
    """[B-NOSTATE] Se pueden rehacer las figuras solo desde el archivo."""
    ge = pytest.importorskip('cosmo_genetic_optimizers')
    core = pytest.importorskip('cosmo_core')
    import matplotlib
    matplotlib.use('Agg')
    rs = [_ga_result_sintetico(seed=s) for s in (0, 1)]
    rs[0].label, rs[0].quantumness = 'CGA', 0.0
    with tempfile.TemporaryDirectory() as td:
        p = ge.save_ga_state(rs, core.MODELS['lcdm'], td)
        figs = ge.replot_from_state(p, animate=False)
        assert figs, "no se regenero ninguna figura"
        for f in figs:
            assert os.path.exists(f) and os.path.getsize(f) > 1000


def test_b_nostate_formato_futuro_avisa_en_vez_de_fallar_raro():
    """[B-NOSTATE] Un archivo de una version mas nueva da un error legible."""
    ge = pytest.importorskip('cosmo_genetic_optimizers')
    core = pytest.importorskip('cosmo_core')
    import json as _json
    with tempfile.TemporaryDirectory() as td:
        p = ge.save_ga_state([_ga_result_sintetico()], core.MODELS['lcdm'], td)
        z = dict(np.load(p, allow_pickle=False))
        meta = _json.loads(str(z['meta_json']))
        meta['version'] = ge.GA_STATE_VERSION + 5
        z['meta_json'] = np.array(_json.dumps(meta))
        np.savez_compressed(p, **z)
        with pytest.raises(ValueError, match='formato'):
            ge.load_ga_state(p)


def test_b_nostate_sin_pickle():
    """[B-NOSTATE] Se carga con allow_pickle=False, o no dura dos anos."""
    ge = pytest.importorskip('cosmo_genetic_optimizers')
    core = pytest.importorskip('cosmo_core')
    with tempfile.TemporaryDirectory() as td:
        p = ge.save_ga_state([_ga_result_sintetico()], core.MODELS['lcdm'], td)
        np.load(p, allow_pickle=False)     # no debe levantar


# ─────────────────────────────────────────────────────────────────────────────
# [B-BUDGET] La comparacion clasico vs cuantico tiene que costar lo mismo
# ─────────────────────────────────────────────────────────────────────────────
#
# Origen: revision adversarial de cosmo_modular_quantum.py. `max_iter` se
# pasaba igual a las dos ramas del entrenamiento del QVMC, pero una iteracion
# cuantica cuesta 1 + 2*n_phi evaluaciones de circuito (el KL en phi mas las
# desplazadas del parameter-shift) y una clasica cuesta 1. Con el mismo
# --qvmc-iter la rama cuantica recibia entre 57x y 225x mas trabajo, y la
# comparacion salia sesgada A FAVOR de lo cuantico — que es el peor resultado
# posible para un proyecto cuya afirmacion central es fidelidad, no ventaja.

def _cuenta_circuitos(v, max_iter):
    """Corre un QVMC contando cuantas evaluaciones de circuito gasta."""
    n = [0]
    orig = v._kl_batch

    def patched(ph, *a, **k):
        n[0] += len(np.atleast_2d(ph))
        return orig(ph, *a, **k)

    v._kl_batch = patched
    r = v.run(max_iter=max_iter, n_chains=1, progress=False)
    return n[0], r


def test_b_budget_las_dos_ramas_gastan_lo_mismo():
    """[B-BUDGET] A presupuesto igualado, ambas ramas usan circuitos similares."""
    mq = pytest.importorskip('cosmo_modular_quantum')
    core = pytest.importorskip('cosmo_core')
    import matplotlib
    matplotlib.use('Agg')
    post = core.Posterior(core.MODELS['lcdm'], 'CC+BAO')
    kw = dict(n_qubits_per_param=2, n_shots=300)
    n_cla, _ = _cuenta_circuitos(
        mq.QVMCModular(post, config={}, budget_mode='circuits', **kw), 20)
    n_qua, _ = _cuenta_circuitos(
        mq.QVMCModular(post, config={'training': True},
                       budget_mode='circuits', **kw), 20)
    assert 0.5 < n_cla / n_qua < 2.0, (
        f"presupuestos muy distintos: clasico {n_cla}, cuantico {n_qua}")


def test_b_budget_el_modo_viejo_si_estaba_sesgado():
    """[B-BUDGET] Regresion: con 'iters' la asimetria existe y es enorme.

    Sin esto, el test de arriba pasaria tambien con un codigo que acertara por
    casualidad. Aqui se fija que el sesgo concreto que se corrigio era real.
    """
    mq = pytest.importorskip('cosmo_modular_quantum')
    core = pytest.importorskip('cosmo_core')
    import matplotlib
    matplotlib.use('Agg')
    post = core.Posterior(core.MODELS['lcdm'], 'CC+BAO')
    kw = dict(n_qubits_per_param=2, n_shots=300)
    n_cla, _ = _cuenta_circuitos(
        mq.QVMCModular(post, config={}, budget_mode='iters', **kw), 20)
    n_qua, _ = _cuenta_circuitos(
        mq.QVMCModular(post, config={'training': True},
                       budget_mode='iters', **kw), 20)
    assert n_qua > 10 * n_cla, (
        f"se esperaba la asimetria historica; clasico {n_cla}, "
        f"cuantico {n_qua}")


def test_b_budget_se_reporta_el_trabajo_gastado():
    """[B-BUDGET] El resultado debe declarar su presupuesto y sus circuitos."""
    mq = pytest.importorskip('cosmo_modular_quantum')
    core = pytest.importorskip('cosmo_core')
    import matplotlib
    matplotlib.use('Agg')
    post = core.Posterior(core.MODELS['lcdm'], 'CC+BAO')
    v = mq.QVMCModular(post, config={'training': True},
                       n_qubits_per_param=2, n_shots=300)
    r = v.run(max_iter=5, n_chains=1, progress=False)
    assert r['budget_mode'] == 'circuits'
    assert isinstance(r['circuits_train'], int) and r['circuits_train'] > 0


def test_b_budget_modo_desconocido_se_rechaza():
    """[B-BUDGET] No se acepta una cadena cualquiera como modo."""
    mq = pytest.importorskip('cosmo_modular_quantum')
    core = pytest.importorskip('cosmo_core')
    post = core.Posterior(core.MODELS['lcdm'], 'CC+BAO')
    with pytest.raises(ValueError, match='budget_mode'):
        mq.QVMCModular(post, config={}, budget_mode='lo_que_sea')


# ─────────────────────────────────────────────────────────────────────────────
# [B-ESSCOMP] El ESS del QVMC no puede depender de como se represente la muestra
# ─────────────────────────────────────────────────────────────────────────────

def test_b_esscomp_el_ess_no_depende_de_la_representacion():
    """[B-ESSCOMP] La celda FAITHFUL `sampling` no puede parecer una regresion.

    La rama cuantica devolvia la forma COMPRIMIDA (una fila por cadena de bits,
    peso = conteo) y la clasica una fila por disparo. El ESS de Kish no es
    invariante bajo esa compresion, asi que el CSV publicaba una caida de 60x
    en una celda donde por construccion no debe haber diferencia.
    """
    mq = pytest.importorskip('cosmo_modular_quantum')
    core = pytest.importorskip('cosmo_core')
    import matplotlib
    matplotlib.use('Agg')
    post = core.Posterior(core.MODELS['lcdm'], 'CC+BAO')
    out = {}
    for lbl, cfg in (('clasico', {}), ('cuantico', {'sampling': True})):
        v = mq.QVMCModular(post, config=cfg, n_qubits_per_param=3,
                           n_shots=1000)
        r = v.run(max_iter=4, n_chains=2, progress=False)
        out[lbl] = (len(r['S']), r['ess'])
    (n_c, ess_c), (n_q, ess_q) = out['clasico'], out['cuantico']
    assert n_c == n_q, f"distinto numero de filas: {n_c} vs {n_q}"
    assert abs(ess_c - ess_q) / max(ess_c, 1.0) < 0.02, (
        f"ESS incomparables entre ramas: clasico {ess_c:.1f}, "
        f"cuantico {ess_q:.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# [B-EMPTY] Una rejilla sin soporte debe levantar, no dar un KL bonito
# ─────────────────────────────────────────────────────────────────────────────

def test_b_empty_rejilla_sin_soporte_levanta():
    """[B-EMPTY] Antes devolvia KL = 0.022 y Om = -4.45 sin un solo error.

    Con la rejilla fuera del prior, P sale toda a cero; el KL degenera en
    log n - H(Q), que es PEQUENO — o sea que la corrida terminaba entera
    reportando el mejor KL de la campana sobre parametros absurdos.
    """
    mq = pytest.importorskip('cosmo_modular_quantum')
    core = pytest.importorskip('cosmo_core')
    post = core.Posterior(core.MODELS['lcdm'], 'CC+BAO')
    for cfg in ({}, dict(mq.PRESETS[100])):
        v = mq.QVMCModular(post, config=cfg, n_qubits_per_param=2,
                           grid_window=[(-100.0, -90.0), (-100.0, -90.0)])
        with pytest.raises(ValueError, match='B-EMPTY'):
            v.build_target()


# ─────────────────────────────────────────────────────────────────────────────
# [B-PROV] Una fila del CSV debe decir en que condiciones se obtuvo
# ─────────────────────────────────────────────────────────────────────────────

def test_b_prov_el_csv_lleva_ruido_ruta_semilla_y_presupuesto():
    """[B-PROV] Sin estas columnas el eje de ruido es irreconstruible.

    Dos filas con la misma clave (Method, model, dataset, prior, nqpp) pero
    numeros distintos eran indistinguibles: una podia venir de una corrida
    ideal y otra de una ruidosa, y el archivo no lo decia.
    """
    mq = pytest.importorskip('cosmo_modular_quantum')
    core = pytest.importorskip('cosmo_core')
    for fields in (mq.csv_fields_for_model(core.MODELS['lcdm']),
                   mq.csv_fields_generic()):
        for col in ('noise', 'proposal_route', 'seed', 'budget_mode',
                    'circuits_train'):
            assert col in fields, f"falta la columna {col!r}"
    side = {'mu': np.array([0.3, 70.0]), 'std': np.array([0.01, 0.5]),
            'chi2': 27.5, 'n_data': 51, 'chi2_red': 0.56, 'AIC': 31.5,
            'BIC': 35.3, 'ess': 1234.0, 'kl_final': 0.5,
            'budget_mode': 'circuits', 'circuits_train': 570, 'seed': 7}
    row = mq.csv_row_for_side(side, core.MODELS['lcdm'], 'QVMC 67%', False,
                              3, 'CC+BAO', 'flat')
    assert row['budget_mode'] == 'circuits'
    assert row['circuits_train'] == '570'
    assert row['noise'] == mq.NOISE.label


# ─────────────────────────────────────────────────────────────────────────────
# [B-REFINE] El chi2 del genetico no distingue rungs; el de la rejilla si
# ─────────────────────────────────────────────────────────────────────────────
#
# El genetico devuelve un punto de la REJILLA y despues se refina con un
# optimizador continuo. Ese refinamiento borra la diferencia entre peldanos:
# los cuatro convergen al mismo minimo, asi que chi2/chi2_red/AIC/BIC salen
# identicos a la sexta cifra en CGA y en los cuatro rungs del QGA. Eso explica
# por que en las campanas publicadas el chi2 del genetico era byte a byte igual
# en todas las filas: no es fidelidad de los operadores, es el refinador.

def test_b_refine_el_chi2_refinado_no_distingue_rungs():
    """[B-REFINE] Los cuatro rungs dan el MISMO chi2 tras refinar.

    Es el hecho que hay que conocer para no leer esa columna como si midiera
    al genetico.
    """
    ge = pytest.importorskip('cosmo_genetic_optimizers')
    core = pytest.importorskip('cosmo_core')
    import matplotlib
    matplotlib.use('Agg')
    post = core.Posterior(core.MODELS['lcdm'], 'CC+BAO')
    ga = ge.GAConfig(pop_size=30, n_generations=6, seed=42)
    refinados, rejilla = [], []
    for pct in (0, 33, 67, 100):
        r = ge.QGA(post, ga, dict(ge.QGA_PRESETS[pct]), n_bits=4,
                   rng=np.random.default_rng(42), shots=1
                   ).evolve(record_population=False)
        refinados.append(r.chi2_map)
        rejilla.append(r.chi2_grid)
    assert max(refinados) - min(refinados) < 1e-6, (
        f"se esperaba que el refinamiento igualara los rungs: {refinados}")
    assert max(rejilla) - min(rejilla) > 1e-3, (
        f"el chi2 de la rejilla deberia distinguirlos: {rejilla}")


def test_b_refine_la_celda_faithful_se_cumple_tambien_en_la_rejilla():
    """[B-REFINE] CGA == QGA(0%) antes de refinar, que es la prueba de verdad.

    Si solo coincidieran despues del refinamiento, la igualdad no diria nada
    sobre los operadores — la garantizaria el optimizador continuo.
    """
    ge = pytest.importorskip('cosmo_genetic_optimizers')
    core = pytest.importorskip('cosmo_core')
    import matplotlib
    matplotlib.use('Agg')
    post = core.Posterior(core.MODELS['lcdm'], 'CC+BAO')
    ga = ge.GAConfig(pop_size=30, n_generations=6, seed=42)
    c = ge.CGA(post, ga, rng=np.random.default_rng(42)
               ).evolve(record_population=False)
    q0 = ge.QGA(post, ga, dict(ge.QGA_PRESETS[0]), n_bits=4,
                rng=np.random.default_rng(42), shots=1
                ).evolve(record_population=False)
    assert np.array_equal(c.theta_grid, q0.theta_grid)
    assert c.chi2_grid == q0.chi2_grid
    assert np.array_equal(c.final_pop, q0.final_pop)


def test_b_refine_el_csv_lleva_el_chi2_de_la_rejilla():
    """[B-REFINE] La columna que distingue rungs tiene que quedar guardada."""
    mq = pytest.importorskip('cosmo_modular_quantum')
    core = pytest.importorskip('cosmo_core')
    assert 'chi2_grid' in mq.csv_fields_generic()
    assert 'chi2_grid' in mq.csv_fields_for_model(core.MODELS['lcdm'])
    side = {'mu': np.array([0.3, 70.0]), 'std': np.array([0.01, 0.5]),
            'chi2': 27.4691, 'n_data': 51, 'chi2_red': 0.56, 'AIC': 31.5,
            'BIC': 35.3, 'ess': 300.0, 'chi2_grid': 28.3992}
    row = mq.csv_row_generic(side, core.MODELS['lcdm'], 'QGA (q=67%)', True,
                             '4', 'CC+BAO', 'flat')
    assert row['chi2_grid'] == '28.3992'
    # una fila de samplers no tiene rejilla propia: la columna queda vacia
    row2 = mq.csv_row_generic({k: v for k, v in side.items() if k != 'chi2_grid'},
                              core.MODELS['lcdm'], 'QMCMC 50%', True, '3',
                              'CC+BAO', 'flat')
    assert row2['chi2_grid'] == ''


def test_b_refine_el_estado_guarda_la_rejilla():
    """[B-REFINE] `ga_state.npz` debe conservar theta_grid y chi2_grid."""
    ge = pytest.importorskip('cosmo_genetic_optimizers')
    core = pytest.importorskip('cosmo_core')
    r = _ga_result_sintetico()
    r.theta_grid = np.array([0.2700, 70.3125])
    r.chi2_grid = 28.3992
    with tempfile.TemporaryDirectory() as td:
        p = ge.save_ga_state([r], core.MODELS['lcdm'], td)
        back, _ = ge.load_ga_state(p)
    assert np.array_equal(back[0].theta_grid, r.theta_grid)
    assert back[0].chi2_grid == pytest.approx(r.chi2_grid)


def test_b_budgetmismatch_avisa_si_max_task_supera_el_presupuesto():
    """[B-BUDGETMISMATCH] --max-task-gb > --mem-budget-gb es contradictorio.

    El techo de qubits sale de --max-task-gb, pero el pool admite SIEMPRE al
    menos una tarea aunque no quepa en el presupuesto agregado (si no, una
    tarea grande bloquearia la campana para siempre). Con esa combinacion se
    puede lanzar una tarea que excede el presupuesto entero y la unica senal
    seria un OOMKill horas despues. Medido: --max-task-gb 40 con
    --mem-budget-gb 10 concedia 18 qubits, ~22 GB para UNA tarea.
    """
    import subprocess
    import sys
    base = [sys.executable, os.path.join(REPO, 'cosmo_hpc_runner.py'),
            '--models', 'lcdm', '--nqpp', '8',
            '--dataset', 'CC+BAO+Pantheon', '--dry-run']
    malo = subprocess.run(base + ['--max-task-gb', '40', '--mem-budget-gb', '10'],
                          capture_output=True, text=True, timeout=300).stdout
    bueno = subprocess.run(base + ['--max-task-gb', '12', '--mem-budget-gb', '50'],
                           capture_output=True, text=True, timeout=300).stdout
    assert 'AVISO' in malo, "no avisa de la combinacion contradictoria"
    assert 'AVISO' not in bueno, "avisa cuando la configuracion es coherente"


# ─────────────────────────────────────────────────────────────────────────────
# Los ejemplos de los docstrings tienen que EJECUTARSE
# ─────────────────────────────────────────────────────────────────────────────
#
# Un ejemplo que no corre es peor que ninguno: envejece en silencio y acaba
# documentando algo que el codigo ya no hace. Atarlos a la suite los convierte
# en documentacion que no puede mentir — y de hecho el primero que escribi ya
# atrapo un error mio (afirmaba H(0) == 70.0 exacto, cuando son
# 69.99999999999999 por redondeo de punto flotante).

def test_los_ejemplos_de_los_docstrings_corren():
    """Todos los doctests del proyecto pasan."""
    import doctest
    import importlib
    fallos = []
    for mod in ('cosmo_core', 'cosmo_noise', 'cosmo_hpc_runner',
                'cosmo_modular_quantum', 'cosmo_genetic_optimizers'):
        m = importlib.import_module(mod)
        res = doctest.testmod(m, verbose=False, report=False)
        if res.failed:
            fallos.append(f"{mod}: {res.failed} de {res.attempted}")
    assert not fallos, "doctests rotos -> " + "; ".join(fallos)


def test_hay_ejemplos_en_las_funciones_que_mas_se_llaman():
    """Las funciones cuyo mal uso ya costo una campana llevan ejemplo.

    No es documentacion decorativa: cada uno de estos ejemplos fija un numero
    que se midio y que explica por que existe una correccion.
    """
    import importlib
    ESPERADOS = {
        'cosmo_noise': ['canonical_level', 'param_shift_batch_factor',
                        'noisy_density_bytes', 'genetic_noisy_seconds_per_gen',
                        'genetic_noisy_time_ceiling'],
        'cosmo_hpc_runner': ['dataset_n_data', 'bytes_per_state_samplers',
                             'estimate_qubits_and_mem'],
        'cosmo_core': ['canonical_dataset', 'ess_weights', 'gelman_rubin',
                       'fit_statistics'],
        'cosmo_modular_quantum': ['compute_quantumness', 'quantumness_qmcmc',
                                  'quantumness_qvmc'],
        'cosmo_genetic_optimizers': ['compute_qga_quantumness'],
    }
    faltan = []
    for mod, nombres in ESPERADOS.items():
        m = importlib.import_module(mod)
        for n in nombres:
            d = getattr(m, n).__doc__ or ''
            if '>>>' not in d:
                faltan.append(f"{mod}.{n}")
    assert not faltan, "sin ejemplo ejecutable: " + ", ".join(faltan)
