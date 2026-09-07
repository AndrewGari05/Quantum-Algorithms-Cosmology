#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmo_noise.py — Segundo eje de ablacion: ruido NISQ.
================================================================================

Fuente unica de verdad del eje de ruido, del mismo modo que
`cosmo_core.make_simulator` lo es de la creacion de simuladores y
`cosmo_core.log_prob_batch` lo es de la fisica. Los tres modulos cuanticos
(`cosmo_modular_quantum`, `cosmo_genetic_optimizers` y el gemelo ruidoso del
pipeline QPU) obtienen su ruido de aqui y de ningun otro sitio.

El marco de ablacion pasa a ser bidimensional:

```
                        ruido  ->
                    none   readout   full   <backend>
    quantumness 0%    .        .        .        .
          |    33%    .        .        .        .
          v    67%    .        .        .        .
              100%    .        .        .        .
```

Los cuatro peldanos
-------------------
* `none`      — limite ideal. Es el de TODOS los resultados publicados hasta
                hoy. Reproduce bit a bit la ruta anterior a este modulo.
* `readout`   — solo error de lectura, simetrico, uniforme.
* `full`      — despolarizacion en compuertas de 1 y 2 qubits + lectura.
* `<backend>` — modelo calibrado de un backend real de IBM, por nombre
                (p. ej. `FakeBrisbane`, `fake_brisbane`).

Por que el error de lectura viaja SEPARADO del NoiseModel
---------------------------------------------------------
El error de lectura es un canal CLASICO posterior a la medicion: no toca el
estado. Aer solo lo aplica cuando el circuito mide. Pero dos de las tres
lecturas del simulador ideal (`hadamard_accept_log_batch` y `_kl_batch`) son
probabilidades que bajo ruido se obtienen de `rho` sin medir, y por tanto
serian CIEGAS al canal de lectura: la columna `readout` de la matriz de
ablacion saldria identica a la columna ideal — falsa, y falsa de la peor
manera, porque no falla ruidosamente sino que produce numeros limpios y
plausibles.

Verificado empiricamente (`noise_feasibility_probe.py`, sonda D): con
p = 0.01, 0.03 y 0.05 el valor de `rho[0,0]` no se mueve ni un digito.

La solucion no es medir, sino aplicar el canal en cerrado. Sobre un registro
de n qubits el error de lectura es un producto tensorial de matrices
estocasticas 2x2, de modo que actua exactamente sobre el vector de
probabilidades `diag(rho)`:

    P_ruidosa = (tensor_q M_q) @ P_ideal

Eso da el eje de ruido COMPLETO, exacto y sin ruido de disparo, al costo de
una contraccion O(n * 2^n). Importa porque la ruta por disparos no puede
resolver el sesgo buscado: con compuertas a 1e-3 el sesgo en la aceptacion es
~1.4e-4 y distinguirlo del ruido de muestreo exigiria ~1e7 disparos POR
evaluacion de aceptacion.

Por eso este modulo expone DOS modelos por cada peldano:

* `gate_model()`  — sin readout. Para las rutas que leen `rho` y aplican el
                    canal de lectura con `apply_readout`.
* `full_model()`  — con readout. Para las rutas que de verdad miden (el QGA,
                    el motor de propuesta, el gemelo QPU), donde Aer lo aplica.

Usar el modelo equivocado produce doble conteo del error de lectura o su
desaparicion silenciosa, asi que la eleccion NO se deja al llamador: se pide
por `simulator_kwargs(counts_route=...)`.

Techo de qubits
---------------
Simular con ruido exige matriz de densidad (`2^(2n) * 16 B`) o trayectorias
(`shots * 2^n` en tiempo). Medido sobre el ansatz real del proyecto, la matriz
de densidad DOMINA en tiempo en todo el rango donde cabe (1.5x-15x mas rapida),
porque evoluciona una vez y luego muestrea, mientras que las trayectorias
re-simulan el circuito completo una vez por disparo.

Los disparos NO compran qubits: las trayectorias caben en RAM pero su coste en
tiempo crece como `shots * 2^n`, de modo que convierten un OOM en una corrida
que no termina. La matriz de densidad es la ruta buena.

[REV] Cuanto se puede subir depende de la RAM del nodo, y ese techo SI se
relaja al tener mas — la version anterior de este modulo lo fijaba en 13 duro
argumentando que el limitante era el tiempo, y era una mala generalizacion
sacada de una maquina de 7 GB. Pero el coste no es un solo numero: depende de
cuantas matrices de densidad viven a la vez.

    ancho     rho (1x)     lote parameter-shift (2*n_phi * rho)
     12       256 MB                42 GB
     13       1.0 GB               182 GB
     16        64 GB                14 TB

El entrenamiento cuantico del QVMC materializa `2 * n_phi` estados en un solo
job. Sin ruido eso es barato (cada estado es un statevector de 16*2^n); con
ruido cada uno es una matriz de densidad y el lote pasa a ser el termino
dominante. En un nodo de 95 GB:

* QMCMC — su motor de propuesta usa `max(2, d)` qubits (2 a 4) y la aceptacion
  uno solo. Es GRATIS bajo ruido a cualquier nqpp; nqpp no toca sus circuitos.
* QVMC sin entrenamiento cuantico (rungs 0% y 33%) — una rho suelta: 16 qubits.
* QVMC CON entrenamiento cuantico (rungs 67% y 100%) — manda el lote: 12.
* QGA — una rho suelta por operador: 16 qubits.

Como un `--benchmark` recorre la escalera entera, el peldano del lote es el
que fija el techo de una tarea de samplers. Trocear ese lote (pendiente ya
identificado en el README) es lo que desbloquearia resoluciones mayores.

Medido (ansatz de 3 capas, B=2 bindings, 4096 disparos, FakeBrisbane):

    qubits   densidad   trayectorias   rho teorica
      6        1.66 s       5.31 s        0.1 MB
      8        2.33 s       6.77 s        1.0 MB
     10        3.77 s      22.41 s       16.0 MB
     12       34.40 s     100.85 s      256.0 MB
     13      216.53 s     333.25 s        1.0 GB
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

# =============================================================================
# Constantes del eje
# =============================================================================

#: Peldanos con nombre. Cualquier otro valor se interpreta como nombre de
#: backend de `qiskit_ibm_runtime.fake_provider`.
NAMED_LEVELS: Tuple[str, ...] = ('none', 'readout', 'full')

#: Techo por defecto de qubits con ruido cuando NO se conoce la RAM del nodo.
#:
#: [REV] Este valor fue en su origen un techo DURO, justificado como limite de
#: tiempo. Era una mala generalizacion: se calibro en una maquina de 7 GB, y en
#: un nodo con RAM de verdad y sin prisa el limitante vuelve a ser la memoria,
#: que SI se relaja al tener mas. El techo se deriva ahora de la RAM
#: disponible (`noisy_qubit_ceiling`) y esto queda solo como respaldo
#: conservador para cuando no hay una cifra de RAM que usar.
DEFAULT_NOISY_QUBITS: int = 13

#: Alias retrocompatible del nombre anterior.
MAX_NOISY_QUBITS: int = DEFAULT_NOISY_QUBITS

#: Capas del ansatz variacional, para dimensionar el lote de parameter-shift.
ANSATZ_LAYERS: int = 3

#: Parametros por defecto de los peldanos sinteticos.
DEFAULT_READOUT_P: float = 0.03
DEFAULT_GATE_P1: float = 1e-3
DEFAULT_GATE_P2: float = 1e-2

#: Compuertas a las que se adjunta despolarizacion en el peldano `full`.
_GATES_1Q = ('id', 'u', 'u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'x', 'y', 'z',
             'h', 's', 'sdg', 't', 'tdg', 'sx', 'sxdg', 'p')
_GATES_2Q = ('cx', 'cz', 'cy', 'ch', 'swap', 'ecr', 'rzz', 'rxx', 'ryy')


# =============================================================================
# Utilidades de nombres
# =============================================================================

def canonical_level(level: Optional[str]) -> str:
    """Normaliza el nombre de un nivel de ruido para logs y CSV.

    Los peldanos con nombre se dejan en minusculas; los nombres de backend se
    normalizan a minusculas sin guiones bajos, de modo que `FakeBrisbane`,
    `fake_brisbane` y `fakebrisbane` produzcan la MISMA etiqueta en el CSV y
    no se dupliquen filas del eje.

    Args:
        level: nivel tal como lo escribio el usuario, o None.

    Returns:
        Etiqueta canonica; `'none'` cuando `level` es None o vacio.
    Examples:
        Los tres peldanos con nombre se normalizan a minusculas, y cualquier
        otra cosa se interpreta como nombre de backend:

        >>> canonical_level('NONE'), canonical_level(None)
        ('none', 'none')
        >>> canonical_level('FakeBrisbane')
        'fakebrisbane'
    """
    if not level:
        return 'none'
    s = str(level).strip()
    if s.lower() in NAMED_LEVELS:
        return s.lower()
    return re.sub(r'[_\-\s]', '', s).lower()


def _resolve_fake_backend(level: str):
    """Devuelve la clase de fake backend cuyo nombre canonico es `level`.

    Args:
        level: etiqueta canonica (minusculas, sin guiones bajos).

    Returns:
        Instancia del fake backend.

    Raises:
        ValueError: si ningun backend del fake_provider coincide.
    """
    try:
        from qiskit_ibm_runtime import fake_provider
    except ImportError as exc:                              # pragma: no cover
        raise ValueError(
            f"El nivel de ruido '{level}' exige qiskit-ibm-runtime, que no "
            f"esta instalado. Usa none/readout/full o instala el paquete."
        ) from exc

    for name in dir(fake_provider):
        if not name.startswith('Fake'):
            continue
        if canonical_level(name) == level:
            return getattr(fake_provider, name)()

    available = sorted(n for n in dir(fake_provider) if n.startswith('Fake'))
    raise ValueError(
        f"Nivel de ruido desconocido: '{level}'. Se esperaba uno de "
        f"{NAMED_LEVELS} o un backend de fake_provider. "
        f"Disponibles (primeros 10): {available[:10]}"
    )


# =============================================================================
# Construccion de modelos de ruido
# =============================================================================

def _readout_error(p: float):
    """`ReadoutError` simetrico de probabilidad de volteo `p`.

    Args:
        p: probabilidad de que el bit reportado se voltee.
    """
    from qiskit_aer.noise import ReadoutError
    return ReadoutError([[1 - p, p], [p, 1 - p]])


def _readout_noise_model(p: float):
    """NoiseModel con SOLO error de lectura simetrico y uniforme.

    Args:
        p: probabilidad de que el bit reportado se voltee.
    """
    from qiskit_aer.noise import NoiseModel
    nm = NoiseModel()
    nm.add_all_qubit_readout_error(_readout_error(p))
    return nm


def _gate_noise_model(p1: float, p2: float):
    """NoiseModel con SOLO despolarizacion de compuerta (sin lectura).

    Args:
        p1: probabilidad de despolarizacion en compuertas de un qubit.
        p2: idem en compuertas de dos qubits.
    """
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    nm = NoiseModel()
    if p1 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1),
                                       list(_GATES_1Q))
    if p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2),
                                       list(_GATES_2Q))
    return nm


def _extract_readout(noise_model
                     ) -> Tuple[Dict[int, np.ndarray], Optional[np.ndarray]]:
    """Extrae las matrices de confusion de lectura de un NoiseModel.

    Solo LEE: no reconstruye ni modifica el modelo. Las matrices devueltas son
    las que el mapa analitico `NoiseSpec.apply_readout` aplica sobre
    `diag(rho)` en las rutas que no miden.

    [B-RO] Un error de lectura declarado con `add_all_qubit_readout_error`
    aparece en `to_dict()` SIN clave `gate_qubits` — significa "todos los
    qubits". Tratar esa ausencia como `[[0]]` lo degradaba a un error de un
    solo qubit: el mapa analitico ruidificaba n qubits mientras que Aer solo
    ruidificaba el qubit 0. El sintoma era silencioso — coincidencia exacta en
    n=1 y divergencia creciente con n y con p — asi que el caso "sin
    gate_qubits" se devuelve como matriz POR DEFECTO, no como entrada del
    qubit 0.

    [B-RECON] Una version anterior ademas RECONSTRUIA el modelo sin lectura
    con `NoiseModel.from_dict()` para alimentar la ruta de `rho`. Se elimino
    por dos razones: `from_dict` esta deprecado desde qiskit-aer 0.15, y el
    round-trip resulta LOSSY para los modelos calibrados — medido contra
    FakeBrisbane, la rho reconstruida difiere de la original en 6.2e-4 a 3
    qubits, muy por encima del epsilon de maquina. No hace falta: sobre un
    circuito que no mide, Aer no aplica el canal de lectura (verificado a
    2e-16 en los peldanos sinteticos), asi que el modelo COMPLETO sirve para
    ambas rutas.

    Args:
        noise_model: NoiseModel de Aer.

    Returns:
        Tupla `({qubit: M}, M_por_defecto_o_None)`, donde
        `M[a, b] = P(reportar a | verdadero b)`.
    """
    per_qubit: Dict[int, np.ndarray] = {}
    default: Optional[np.ndarray] = None
    for entry in noise_model.to_dict().get('errors', []):
        if entry.get('type') != 'roerror':
            continue
        # to_dict entrega P(reportar a | verdadero b) como fila b, columna a;
        # se transpone para que `M @ p` sea la accion sobre probabilidades.
        probs = np.asarray(entry['probabilities'], dtype=float).T
        qubit_groups = entry.get('gate_qubits')
        if not qubit_groups:                      # [B-RO] all-qubit
            default = probs.copy()
        else:
            for qubits in qubit_groups:
                per_qubit[int(qubits[0])] = probs.copy()
    return per_qubit, default


# =============================================================================
# NoiseSpec
# =============================================================================

@dataclass
class NoiseSpec:
    """Un peldano del eje de ruido, ya resuelto y listo para usar.

    Construir con `NoiseSpec.from_level(...)`, nunca a mano: el reparto entre
    ruido de compuerta y de lectura es lo que evita el doble conteo, y el
    constructor es quien lo garantiza.

    Attributes:
        label: etiqueta canonica para logs y CSV.
        source_model: el NoiseModel COMPLETO (compuertas + lectura), tal cual
            lo produjo Aer. Es el UNICO modelo que se usa, en ambas rutas: la
            reconstruccion de una variante sin lectura resulto deprecada y
            ademas lossy (ver [B-RECON] en `_extract_readout`).
        readout: matrices de confusion 2x2 por qubit; vacio si el canal de
            lectura es uniforme (entonces vive en `default_readout`).
        default_readout: matriz de confusion aplicable a cualquier qubit sin
            entrada propia, o None. Los peldanos sinteticos son uniformes y
            usan esta; los backends reales traen una matriz por qubit.
    """

    label: str
    source_model: Optional[object] = None
    readout: Dict[int, np.ndarray] = field(default_factory=dict)
    default_readout: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    @classmethod
    def from_level(cls, level: Optional[str],
                   readout_p: float = DEFAULT_READOUT_P,
                   gate_p1: float = DEFAULT_GATE_P1,
                   gate_p2: float = DEFAULT_GATE_P2) -> "NoiseSpec":
        """Resuelve un nivel del eje a un `NoiseSpec` utilizable.

        Args:
            level: `'none'`, `'readout'`, `'full'` o un nombre de fake backend.
            readout_p: probabilidad de lectura de los peldanos sinteticos.
            gate_p1: despolarizacion de 1 qubit del peldano `full`.
            gate_p2: despolarizacion de 2 qubits del peldano `full`.

        Returns:
            El `NoiseSpec` correspondiente.

        Raises:
            ValueError: si el nivel no es reconocible.
        """
        lab = canonical_level(level)

        if lab == 'none':
            return cls(label='none')

        if lab == 'readout':
            source = _readout_noise_model(readout_p)
        elif lab == 'full':
            source = _gate_noise_model(gate_p1, gate_p2)
            source.add_all_qubit_readout_error(
                _readout_error(readout_p))
        else:
            # Backend real: el modelo calibrado ya trae ambas partes juntas.
            from qiskit_aer.noise import NoiseModel
            source = NoiseModel.from_backend(_resolve_fake_backend(lab))

        per_qubit, default = _extract_readout(source)
        return cls(label=lab, source_model=source, readout=per_qubit,
                   default_readout=default)

    # ------------------------------------------------------------------ #
    @property
    def is_ideal(self) -> bool:
        """True si este peldano es el limite ideal (sin ruido de ningun tipo).

        Cuando es True, TODA ruta de este modulo debe reducirse exactamente a
        la del codigo previo al eje de ruido: mismo metodo de simulacion,
        mismas lecturas, mismos numeros bit a bit. Es la condicion que hace
        de `--noise none` una linea base valida y no una aproximacion.
        """
        return self.source_model is None

    @property
    def has_readout(self) -> bool:
        """True si el peldano incluye canal de lectura."""
        return bool(self.readout) or self.default_readout is not None

    # ------------------------------------------------------------------ #
    def full_model(self):
        """NoiseModel CON error de lectura, para rutas que miden de verdad.

        Es el que usan el QGA, el motor de propuesta y el gemelo QPU: ahi el
        circuito mide y Aer aplica el canal de lectura por si mismo, asi que
        aplicarlo ademas con `apply_readout` seria doble conteo.

        Devuelve el modelo ORIGINAL sin reconstruirlo: la reconstruccion es
        justo donde se perdio la semantica "todos los qubits" en [B-RO] y
        donde el modelo calibrado perdia precision en [B-RECON].

        Returns:
            NoiseModel, o None si el peldano es ideal.
        """
        return self.source_model

    # ------------------------------------------------------------------ #
    def simulator_kwargs(self, counts_route: bool) -> Dict[str, object]:
        """kwargs para `cosmo_core.make_simulator` en este peldano.

        La eleccion del modelo (con o sin lectura) NO se deja al llamador
        porque equivocarla produce doble conteo del error de lectura o su
        desaparicion silenciosa; se decide aqui a partir de `counts_route`.

        El modelo entregado es el mismo en ambas rutas; lo que cambia es quien
        aplica el canal de lectura. En `counts_route=True` el circuito mide y
        lo aplica Aer. En `counts_route=False` el circuito no mide, Aer ignora
        el canal de lectura por construccion (verificado a 2e-16), y el
        llamador DEBE aplicarlo despues con `apply_readout` — si no lo hace,
        la columna `readout` del eje sale identica a la ideal.

        Args:
            counts_route: True si el circuito MIDE y el resultado se lee de
                conteos (QGA, motor de propuesta, gemelo QPU). False si el
                resultado se lee del estado (`rho[0,0]`, `diag(rho)`).

        Returns:
            dict con `method` y, si procede, `noise_model`.
        """
        if self.is_ideal:
            return {'method': 'statevector'}
        return {'method': 'density_matrix', 'noise_model': self.source_model}

    # ------------------------------------------------------------------ #
    def readout_matrix(self, qubit: int) -> Optional[np.ndarray]:
        """Matriz de confusion 2x2 del qubit dado, o None si no hay lectura.

        Args:
            qubit: indice del qubit.

        Returns:
            `M` con `M[a, b] = P(reportar a | verdadero b)`, o None.
        """
        if qubit in self.readout:
            return self.readout[qubit]
        return self.default_readout

    # ------------------------------------------------------------------ #
    def apply_readout(self, probs: np.ndarray, n_qubits: int) -> np.ndarray:
        """Aplica el canal de lectura en cerrado a un vector de probabilidades.

        Sobre n qubits el error de lectura es un producto tensorial de mapas
        estocasticos 2x2, asi que actua EXACTAMENTE sobre `diag(rho)` sin
        necesidad de medir y sin introducir ruido de disparo. La contraccion
        se hace qubit a qubit en O(n * 2^n) en vez de construir la matriz
        2^n x 2^n.

        Convencion de bits: Qiskit es little-endian, el bit del qubit q del
        indice i es `(i >> q) & 1`. Al hacer `reshape([2]*n)` en orden C el
        eje 0 corresponde al bit MAS significativo, es decir al qubit n-1; de
        ahi el mapeo `eje = n - 1 - q`.

        Args:
            probs: vector de probabilidades de longitud `2**n_qubits`, o lote
                de forma `(B, 2**n_qubits)`.
            n_qubits: numero de qubits del registro.

        Returns:
            Vector (o lote) de probabilidades tras la lectura ruidosa. Si el
            peldano no tiene canal de lectura, devuelve `probs` sin copiar.
        """
        if not self.has_readout:
            return probs

        p = np.asarray(probs, dtype=float)
        batched = (p.ndim == 2)
        if not batched:
            p = p[None, :]
        b = p.shape[0]
        if p.shape[1] != 2 ** n_qubits:
            raise ValueError(
                f"apply_readout: se esperaban {2 ** n_qubits} probabilidades "
                f"para {n_qubits} qubits, llegaron {p.shape[1]}.")

        out = p.reshape((b,) + (2,) * n_qubits)
        for q in range(n_qubits):
            m = self.readout_matrix(q)
            if m is None:
                continue
            axis = n_qubits - q                      # +1 por el eje de lote
            out = np.moveaxis(out, axis, -1)
            out = out @ m.T
            out = np.moveaxis(out, -1, axis)
        out = out.reshape(b, 2 ** n_qubits)
        return out if batched else out[0]

    # ------------------------------------------------------------------ #
    def metadata(self) -> Dict[str, object]:
        """Metadatos del peldano para el CSV y las figuras.

        Returns:
            dict con la etiqueta, si es ideal, y los parametros efectivos.
        """
        return {
            'noise': self.label,
            'noise_ideal': self.is_ideal,
            'noise_has_readout': self.has_readout,
            'noise_has_gates': self.gate_model is not None,
        }

    def __repr__(self) -> str:                              # pragma: no cover
        """Representacion corta para depuracion."""
        return (f"NoiseSpec(label={self.label!r}, "
                f"gates={self.gate_model is not None}, "
                f"readout={self.has_readout})")


# =============================================================================
# Techo de qubits
# =============================================================================

def ansatz_n_params(n_qubits: int, n_layers: int = ANSATZ_LAYERS) -> int:
    """Numero de angulos del ansatz variacional de `n_qubits`.

    Se mide sobre el circuito REAL cuando qiskit esta disponible, para que no
    pueda desincronizarse si el ansatz cambia.

    [B-PLAN] Con respaldo de formula cerrada cuando qiskit NO se puede
    importar. Construir el circuito arrastra qiskit a `cosmo_hpc_runner`, que
    hasta ahora solo planificaba tareas y lanzaba subprocesos sin necesitar el
    stack cuantico. En una HPC eso importa: se planifica y se envia desde un
    nodo de login donde el entorno de computo no esta cargado, y el runner
    moria con ModuleNotFoundError antes de escribir una sola tarea.

    La formula `n * (2*L + 1)` sale de la estructura del ansatz — L capas de
    (RY, RZ) por qubit mas una capa final de RY — y esta verificada exacta
    contra el circuito real para L = 1..5 y n = 2..18 por
    `test_noise_axis.py`, de modo que el respaldo no puede divergir en
    silencio del ansatz que de verdad se ejecuta.

    Args:
        n_qubits: ancho del circuito.
        n_layers: capas del ansatz.

    Returns:
        Cantidad de parametros libres.
    """
    n, layers = int(n_qubits), int(n_layers)
    try:
        from qpu_cosmo_samplers import build_ansatz
        return int(build_ansatz(n, layers).num_parameters)
    except Exception:                              # [B-PLAN] nodo sin qiskit
        return n * (2 * layers + 1)


def param_shift_batch_factor(n_qubits: int,
                             n_layers: int = ANSATZ_LAYERS) -> int:
    """Multiplicador de memoria del lote de parameter-shift: 2 * n_phi.

    El entrenamiento cuantico del QVMC evalua el gradiente exacto materializando
    `2 * n_phi` estados en un solo job de Aer. Sin ruido eso es barato porque
    cada estado es un statevector (16 * 2^n); CON ruido cada uno es una matriz
    de densidad (16 * 4^n) y el multiplicador se vuelve el termino dominante:

        n=12 -> rho 256 MB, lote  42 GB
        n=13 -> rho 1.0 GB, lote 182 GB

    Es decir, en un nodo de 95 GB el QVMC con entrenamiento cuantico y ruido
    topa en 12 qubits, no en 13 ni en 16: lo que manda es el LOTE, no rho.
    Trocear ese lote (pendiente ya identificado en el README) es lo que
    desbloquearia resoluciones mayores.

    Args:
        n_qubits: ancho del circuito.
        n_layers: capas del ansatz.

    Returns:
        `2 * n_phi`.
    Examples:
        Este es el numero que fija el techo de qubits del QVMC con ruido: el
        entrenamiento cuantico materializa `2 * n_phi` matrices de densidad a
        la vez, no una.

        >>> param_shift_batch_factor(4)
        56
        >>> param_shift_batch_factor(12)
        168
    """
    return 2 * ansatz_n_params(n_qubits, n_layers)


def noisy_density_bytes(n_qubits: int, batch_factor: int = 1) -> int:
    """Bytes de matriz de densidad de `n_qubits` (complejo de 16 B).

    Args:
        n_qubits: ancho del circuito.
        batch_factor: cuantas matrices de densidad viven a la vez (1 para una
            evaluacion suelta; `param_shift_batch_factor(n)` para el lote de
            entrenamiento cuantico del QVMC).
    Examples:
        Una matriz de densidad suelta a 12 qubits son 256 MB...

        >>> noisy_density_bytes(12) / 1e6
        268.435456

        ...pero el lote de parameter-shift a esa anchura son 45 GB, y por eso
        el techo del QVMC ruidoso esta muy por debajo del que da la RAM:

        >>> round(noisy_density_bytes(12, param_shift_batch_factor(12)) / 1e9, 1)
        45.1
    """
    return (2 ** (2 * int(n_qubits))) * 16 * int(batch_factor)


def noisy_qubits_fitting_in(mem_mb: float, quantum_training: bool = True,
                            n_layers: int = ANSATZ_LAYERS) -> int:
    """Mayor ancho cuyo coste con ruido cabe en `mem_mb`.

    Args:
        mem_mb: memoria disponible para la tarea, en MB.
        quantum_training: si True (por defecto) reserva sitio para el lote de
            parameter-shift, que es el peldano mas caro de la escalera QVMC y
            por tanto el que manda en un `--benchmark` completo. Ponlo en
            False solo si la corrida no incluye rungs con entrenamiento
            cuantico.
        n_layers: capas del ansatz.

    Returns:
        Numero de qubits, 0 si no cabe ni el caso mas pequeno.
    """
    best = 0
    for n in range(1, 31):
        factor = (param_shift_batch_factor(n, n_layers)
                  if quantum_training else 1)
        if noisy_density_bytes(n, factor) / 1e6 <= mem_mb:
            best = n
        else:
            break
    return best


# ── [B-TIME] Techo por TIEMPO del genetico con ruido ────────────────────────
#
# La memoria NO es el limitante del QGA con ruido: una rho suelta a 14 qubits
# son 4.3 GB, que caben de sobra en cualquier nodo util. El limitante es el
# tiempo, y crece como ~4^n porque cada evaluacion propaga una matriz de
# densidad de 4^n elementos.
#
# Medido en la campana 2026-09-03 (pop=500, ruido readout, nodo de 63 GiB,
# leido de los logs por marca de tiempo entre generaciones):
#
#     10 qubits (n_bits=5, d=2)  ->     73 s/generacion
#     12 qubits (n_bits=6, d=2)  ->   1425 s/generacion   (23.8 min)
#     14 qubits (n_bits=7, d=2)  ->  no habia registrado ni la generacion 0
#                                    tras 7 h de reloj
#
# Los dos primeros puntos fijan un crecimiento de x4.42 por qubit, que
# extrapolado a 14 da 27810 s/generacion (7.7 h) — consistente con la tercera
# observacion, que asi queda de validacion y no de ajuste.
#
# Con ese ritmo, 500 generaciones a 14 qubits son ~4 MESES. El plan de esa
# campana las acepto porque el techo se derivaba solo de la RAM, y la tarea
# quedo colgada bloqueando ademas las figuras de resumen de toda la corrida
# (el runner no las escribe hasta que la lista entera termina).
#
# Esto NO reintroduce la constante dura que se quito en [REV]. Aquel techo era
# malo por dos razones distintas: estaba calibrado en una maquina de 7 GB, y se
# aplicaba a los SAMPLERS, donde el que manda de verdad es el lote de
# parameter-shift, o sea memoria. Aqui el limite es de tiempo — que no se
# relaja con mas RAM — esta calibrado con medidas del nodo real, y no es una
# constante: sale de un presupuesto de reloj por tarea que el usuario fija.
GENETIC_NOISY_REF_QUBITS = 10
GENETIC_NOISY_REF_SEC_PER_GEN = 73.0
GENETIC_NOISY_GROWTH_PER_QUBIT = 4.42

#: Presupuesto de reloj por tarea genetica ruidosa, en horas. Es el valor por
#: defecto de `--noisy-task-hours`; dos dias deja pasar 12 qubits y corta 13.
DEFAULT_NOISY_TASK_HOURS = 48.0


def genetic_noisy_seconds_per_gen(n_qubits: int) -> float:
    """Segundos por generacion del QGA con ruido, a `n_qubits`.

    Extrapolacion de las dos mediciones de la campana 2026-09-03 (ver el
    bloque [B-TIME] de arriba). Es un orden de magnitud, no una promesa: sirve
    para decidir si una celda tarda horas o meses, que es la unica pregunta que
    hay que contestar al planificar.

    AVISO de alcance: las mediciones son con `--noise readout`, que es el
    canal mas barato — se aplica analiticamente sobre diag(rho). Los niveles
    `full` y los backends calibrados (FakeBrisbane) meten errores de puerta,
    o sea mas operadores de Kraus por compuerta, y corren MAS LENTO que lo que
    predice esta funcion. Con esos niveles en la barrida, baja
    `--noisy-task-hours` para compensar, o cuenta con que el reloj real supere
    al presupuesto.

    Args:
        n_qubits: ancho del circuito en qubits.

    Returns:
        float
    Examples:
        Reproduce las dos mediciones de la campana 2026-09-03...

        >>> round(genetic_noisy_seconds_per_gen(10))
        73
        >>> round(genetic_noisy_seconds_per_gen(12))
        1426

        ...y extrapola a 14 qubits el valor que motivo [B-TIME]: 7.7 h POR
        generacion, o sea ~4 meses las 500 que se pidieron.

        >>> round(genetic_noisy_seconds_per_gen(14) / 3600, 1)
        7.7
    """
    d = int(n_qubits) - GENETIC_NOISY_REF_QUBITS
    return GENETIC_NOISY_REF_SEC_PER_GEN * (GENETIC_NOISY_GROWTH_PER_QUBIT ** d)


def genetic_noisy_time_ceiling(generations: int,
                               budget_hours: float = DEFAULT_NOISY_TASK_HOURS
                               ) -> int:
    """Mayor ancho cuyo QGA con ruido cabe en `budget_hours` de reloj.

    Args:
        generations: generaciones que se van a correr.
        budget_hours: presupuesto de reloj por tarea, en horas.

    Returns:
        Techo de qubits; al menos 1, para que un presupuesto absurdo no
        produzca un plan vacio sin explicacion.
    Examples:
        Con 120 generaciones y dos dias de presupuesto caben 12 qubits; con
        las 500 de la campana vieja, solo 11:

        >>> genetic_noisy_time_ceiling(120, 48)
        12
        >>> genetic_noisy_time_ceiling(500, 48)
        11

        No es una constante: mas presupuesto concede mas anchura.

        >>> genetic_noisy_time_ceiling(120, 480)
        13
    """
    budget_s = max(float(budget_hours), 0.0) * 3600.0
    g = max(int(generations), 1)
    best = 1
    for n in range(1, 31):
        if genetic_noisy_seconds_per_gen(n) * g <= budget_s:
            best = n
        else:
            break
    return best


def noisy_qubit_ceiling(requested: Optional[int] = None,
                        mem_mb: Optional[float] = None,
                        quantum_training: bool = True,
                        generations: Optional[int] = None,
                        budget_hours: float = DEFAULT_NOISY_TASK_HOURS) -> int:
    """Techo de qubits admisible para una tarea con ruido.

    [REV] Antes devolvia una constante dura, argumentando que el limitante era
    el tiempo. Era una mala generalizacion, calibrada en una maquina pequena:
    en un nodo con RAM de verdad y sin presion de tiempo el limitante vuelve a
    ser la memoria, y esa SI se relaja al tener mas. Ahora el techo se deriva
    de `mem_mb` igual que los otros dos modelos del runner, y la constante
    queda como respaldo para cuando no hay cifra de RAM.

    El coste con ruido no es un unico numero: depende de cuantas matrices de
    densidad viven a la vez. Con entrenamiento cuantico el lote de
    parameter-shift multiplica por `2 * n_phi` y es el termino que manda.

    Args:
        requested: tope pedido por el usuario, o None. Actua como cota
            superior; nunca concede mas de lo que cabe en memoria.
        mem_mb: memoria disponible por tarea. None usa `DEFAULT_NOISY_QUBITS`.
        quantum_training: reservar sitio para el lote de parameter-shift.
            False es la ruta del genetico, donde ademas se aplica el techo por
            tiempo de [B-TIME].
        generations: generaciones previstas. Solo se usa con
            `quantum_training=False`; sin ella no se aplica el techo temporal.
        budget_hours: presupuesto de reloj por tarea para ese techo.

    Returns:
        Techo efectivo de qubits.
    """
    if mem_mb is None:
        ceiling = DEFAULT_NOISY_QUBITS
    else:
        ceiling = noisy_qubits_fitting_in(mem_mb, quantum_training)
    # [B-TIME] El QGA con ruido no lo frena la memoria sino el reloj: a 14
    # qubits caben 4.3 GB de rho en cualquier nodo, pero son ~7.7 h por
    # generacion. Sin este techo el plan acepta celdas de meses.
    if not quantum_training and generations is not None:
        ceiling = min(ceiling,
                      genetic_noisy_time_ceiling(generations, budget_hours))
    if requested is not None:
        ceiling = min(int(requested), ceiling)
    return ceiling


def add_noise_cli(parser) -> None:
    """Anade el flag `--noise` (y sus parametros) a un ArgumentParser.

    Se centraliza aqui para que los tres ejecutables y el runner HPC ofrezcan
    exactamente la misma interfaz y los mismos valores por defecto.

    Args:
        parser: `argparse.ArgumentParser` al que anadir el grupo.
    """
    g = parser.add_argument_group('eje de ruido NISQ')
    g.add_argument('--noise', type=str, default='none',
                   help="nivel de ruido: none | readout | full | "
                        "<backend> (p. ej. FakeBrisbane). Por defecto 'none', "
                        "que reproduce bit a bit los resultados ideales.")
    g.add_argument('--noise-readout-p', type=float, default=DEFAULT_READOUT_P,
                   help=f"probabilidad de volteo de lectura de los peldanos "
                        f"sinteticos (por defecto {DEFAULT_READOUT_P})")
    g.add_argument('--noise-gate-p1', type=float, default=DEFAULT_GATE_P1,
                   help=f"despolarizacion de 1 qubit en 'full' "
                        f"(por defecto {DEFAULT_GATE_P1})")
    g.add_argument('--noise-gate-p2', type=float, default=DEFAULT_GATE_P2,
                   help=f"despolarizacion de 2 qubits en 'full' "
                        f"(por defecto {DEFAULT_GATE_P2})")


def spec_from_args(args) -> NoiseSpec:
    """Construye el `NoiseSpec` a partir de los args de `add_noise_cli`.

    Args:
        args: namespace de argparse.

    Returns:
        El `NoiseSpec` resuelto.
    """
    return NoiseSpec.from_level(
        getattr(args, 'noise', 'none'),
        readout_p=getattr(args, 'noise_readout_p', DEFAULT_READOUT_P),
        gate_p1=getattr(args, 'noise_gate_p1', DEFAULT_GATE_P1),
        gate_p2=getattr(args, 'noise_gate_p2', DEFAULT_GATE_P2))
