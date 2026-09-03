# Fase 4 — Primera corrida del eje de ruido: resultados

Corrida de calibración del instrumento, no de producción. ΛCDM, CC+BAO,
`nqpp=3` (6 qubits), 600 pasos, 4 cadenas, 25 iteraciones QVMC, `--seed 42`,
25 generaciones × 60 individuos para el genético, `n_bits=3`. Tamaños
pequeños a propósito: el objetivo es ver si el eje **mide algo** y en qué
dirección, no producir números citables.

Cinco columnas, no cuatro. La quinta (`none-counts`) es el control, y sin
ella tres de las cinco conclusiones de abajo habrían salido al revés.

---

## 1. El control era imprescindible

`--noise none` usa por defecto la ruta de amplitudes; cualquier peldaño con
ruido usa la ruta por conteos. Comparar la primera columna contra las
ruidosas mezcla **el ruido con el cambio de operador de lectura**. La columna
`none-counts` corre el peldaño ideal por la ruta de conteos y separa ambos.

QMCMC 50%:

| | none (amplitud) | none-counts | readout | full | FakeBrisbane |
|---|---|---|---|---|---|
| σ(Ωm) | 0.0208 | **0.0176** | **0.0176** | 0.0172 | 0.0175 |
| aceptación | 0.4756 | **0.5092** | **0.5092** | 0.5144 | 0.5078 |
| ESS | 101.3 | **140.8** | **140.8** | 151.7 | 142.7 |

Leyendo solo `none` → `readout` se concluiría que **el ruido de lectura
estrecha el posterior y mejora el mezclado** — σ baja 15%, ESS sube 39%. Es
falso. `none-counts` y `readout` son **idénticas hasta el último dígito
impreso**: el canal de lectura no le hace absolutamente nada a la propuesta.
Todo el salto es el cambio de ruta.

Es la invariancia demostrada en el módulo: la lectura uniforme reescala
`⟨Z_q⟩` por `(1−2p)` y la calibración a std unitaria lo divide. Aquí se ve
en una corrida completa, no solo en el test unitario.

**Conclusión operativa:** cualquier tabla del eje de ruido que compare contra
`none` sin el control está midiendo dos cosas a la vez.

---

## 2. La identidad FAITHFUL se rompe exactamente donde debe

QMCMC 50% (propuesta cuántica) vs 100% (propuesta + aceptación cuánticas):

| | none-counts | readout | full | FakeBrisbane |
|---|---|---|---|---|
| Ωm 50% | 0.2629 | 0.2629 | 0.2640 | 0.2637 |
| Ωm 100% | **0.2629** | 0.2648 | 0.2647 | 0.2640 |
| ESS 50% | 140.8 | 140.8 | 151.7 | 142.7 |
| ESS 100% | **140.8** | **117.2** | 118.2 | 115.9 |

En el límite ideal las dos filas son **idénticas**: la aceptación cuántica
reproduce Metropolis exactamente, que es la afirmación de fidelidad del
proyecto. Con ruido dejan de serlo.

La señal más limpia es el **ESS: 140.8 → 117.2 en cuanto se enciende la
lectura**, un 17% de pérdida de mezclado que la fila del 50% no sufre. Es
decir: el componente FAITHFUL es el que paga el ruido, y lo paga en
eficiencia de muestreo antes que en el valor central.

Esto convierte "identidad a 1.1e-16" en una **curva de degradación**, que es
lo que buscabas. El número ideal sigue ahí como punto de partida; ahora tiene
pendiente.

Mecanismo, ya medido en el módulo: la lectura pone un suelo de ~p sobre la
aceptación, así que los movimientos que deberían rechazarse casi siempre se
aceptan de más (A = 0.0025 → 0.0323 con p = 0.03, 13×). La cadena acepta
basura y mezcla peor.

---

## 3. La ventaja del entrenamiento cuántico se erosiona de forma monótona

El salto 33% → 67% del QVMC es la celda ALGORITHMIC: entrenamiento por
parameter-shift exacto en vez de COBYLA. Su ganancia en KL:

| peldaño | KL 33% | KL 67% | **ganancia** |
|---|---|---|---|
| none | 12.0885 | 10.5892 | **1.499** |
| readout | 12.5115 | 11.1595 | **1.352** |
| full | 12.7569 | 11.5727 | **1.184** |
| FakeBrisbane | 12.7339 | 11.6417 | **1.092** |

**−27% de la ventaja cuántica al llegar al ruido de un dispositivo real**, y
la erosión es monótona en los cuatro peldaños. Este es el resultado central
de la corrida: no es que el QVMC deje de funcionar, es que *aquello que lo
hacía mejor que su baseline* se encoge a un ritmo medible.

---

## 4. Tres invariancias que confirman que el eje está bien cableado

**El MCMC clásico es idéntico en las cinco columnas** (Ωm 0.2569, σ 0.0160,
aceptación 0.5269, ESS 101.6). Es NumPy puro y nunca toca un circuito, así
que el eje no puede moverlo. Que no se mueva ni en el último dígito es la
mejor evidencia de que el ruido está entrando solo por donde debe.

**QVMC 67% y 100% coinciden en todas las columnas.** Ya estaba anticipado:
`quantum_amplitude_normalization` ejecuta su circuito pero descarta el
resultado — devuelve la suma exacta. Ese componente es **inmune al eje por
construcción**, y cualquier degradación entre 67% y 100% vendría de otro
sitio.

**QGA 0% es bit-idéntico al CGA en los cuatro peldaños.** Con todos los
operadores apagados no corre ningún circuito, así que el ruido no tiene por
dónde entrar.

---

## 5. El VI clásico NO es invariante, y eso hay que declararlo

| | none | readout | full | FakeBrisbane |
|---|---|---|---|---|
| KL del VI clásico | 12.0885 | 12.5115 | 12.7569 | 12.7339 |

El MCMC clásico no se mueve; el VI clásico sí. No es un bug: en este
framework el VI clásico es `QVMCModular` con todos los componentes apagados,
y **sigue representando Q con el circuito del ansatz** — solo optimiza y
muestrea de forma clásica. El circuito es el sustrato, no un componente
conmutable, así que bajo ruido el sustrato es ruidoso para toda la escalera
del QVMC.

Es defendible físicamente (en un dispositivo ruidoso, un estado variacional
optimizado clásicamente también es ruidoso), pero tiene una consecuencia que
no se puede pasar por alto: **el "0%" de la escalera QVMC no es una
referencia fija bajo ruido**, mientras que el "0%" de la escalera QMCMC sí lo
es. Las dos escaleras no son simétricas en este eje, y una tabla que las
ponga lado a lado invita a leerlas como si lo fueran.

Decisión pendiente tuya: dejarlo así y documentarlo, o añadir una opción que
fuerce el baseline del QVMC a correr siempre sin ruido.

---

## 6. El ESS miente bajo ruido

| QVMC | none | readout | full | FakeBrisbane |
|---|---|---|---|---|
| ESS 67% | 91.2 | 106.9 | 116.0 | **125.2** |
| KL 67% | 10.5892 | 11.1595 | 11.5727 | **11.6417** |

El ESS **sube** un 37% mientras el ajuste **empeora**. El ruido aplana la
distribución, y una distribución más plana da muestras menos correlacionadas.
Cualquier lectura que use ESS como métrica de calidad concluiría que el ruido
ayuda.

σ(H0) crece de forma monótona en la misma escalera (2.876 → 2.907 → 2.949 →
2.976), así que el ensanchamiento real está ahí: es el ESS el que no sirve
como métrica de calidad en este eje. **Reportar ESS junto a KL o σ, nunca
solo.**

---

## 7. El QGA sale casi invariante — con una advertencia grande

| Ωm | none | readout | full | FakeBrisbane |
|---|---|---|---|---|
| CGA / QGA 0% | 0.2575 | 0.2575 | 0.2575 | 0.2575 |
| QGA 33% | 0.2575 | 0.2558 | 0.2558 | 0.2575 |
| QGA 67% | 0.2800 | 0.2800 | 0.2800 | 0.2800 |
| QGA 100% | 0.2800 | 0.2800 | 0.2800 | **0.2400** |

Tu predicción se sostiene: el QGA es con diferencia el más robusto. Pero
atribuirlo solo al algoritmo sería precipitado.

Con `n_bits=3` la rejilla tiene **8 niveles por eje**. Los valores de arriba
son puntos de rejilla, no números continuos: para que el ruido mueva el
resultado tiene que voltear bits suficientes para saltar a *otra celda* Y
sobrevivir a selección y elitismo, que preservan al mejor individuo intacto.
La discretización gruesa está cuantizando el ruido hasta hacerlo desaparecer.

Dicho de otro modo: parte de la robustez observada es del **operador** (mide,
y un bit volteado se parece a mutación extra) y parte es de la **rejilla**.
Separarlas exige repetir esto con `n_bits` 5–6, donde una celda es mucho más
estrecha. Hasta entonces, "el QGA es el más robusto" está apoyado pero no
aislado.

El χ² no discrimina nada aquí (27.4691 en las 20 celdas) porque se reporta en
el MAP refinado con Nelder-Mead, que borra las diferencias del GA. Para este
eje hay que mirar Ωm/H0 crudos, no el χ² refinado.

---

## 8. Qué NO se puede concluir todavía

- Tamaños pequeños (600 pasos, 25 iteraciones). Las diferencias de tercer
  decimal en Ωm están dentro del ruido de Monte Carlo de una sola semilla.
  **Nada de aquí es citable sin repetir con varias semillas.**
- Un solo modelo (ΛCDM, d=2) y una sola resolución (nqpp=3).
- El eje mide ruido **sin supresión de error** — DD y twirling quedan
  inertes en simulación. Es una cota inferior de lo alcanzable en hardware.
- La curva de degradación tiene cuatro puntos, y tres de ellos son modelos
  sintéticos. Para una curva de verdad hace falta barrer `p` de forma
  continua, no cuatro peldaños con nombre.

## 9. Lo siguiente que valdría la pena

1. Barrer `--noise-readout-p` de forma continua sobre el QMCMC 100% y ajustar
   la pendiente de la pérdida de ESS. Es barato (2–4 qubits) y da la curva
   real en vez de cuatro puntos.
2. Repetir el genético con `n_bits` 5–6 para separar robustez del operador de
   robustez de la rejilla.
3. Varias semillas antes de citar cualquier número.
