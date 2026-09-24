#!/usr/bin/env python3
"""inventario.py — el mapa del repositorio, generado del codigo.

Existe para que la guia no se desactualice sola: los tamanos, las clases y
las funcions publicas de cada modulo se leen del arbol de sintaxis, no se
escriben a mano.

    python docs/guia_codigo/inventario.py            # tabla legible
    python docs/guia_codigo/inventario.py --latex    # filas para el .tex
"""
from __future__ import annotations

import argparse
import ast
import os
import sys

RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

MOTOR = ['cosmo_core.py', 'cosmo_noise.py', 'cosmo_modular_quantum.py',
         'cosmo_genetic_optimizers.py', 'cosmo_hpc_runner.py',
         'cosmo_profiling.py', 'qpu_cosmo_samplers.py', 'data_manifest.py']
ANALISIS = ['triage_campana.py', 'comparar_semillas.py', 'graficas_ruido.py',
            'comparar_algoritmos.py']


def mide(nombre):
    ruta = os.path.join(RAIZ, nombre)
    if not os.path.exists(ruta):
        return None
    src = open(ruta, encoding='utf-8').read()
    arbol = ast.parse(src)
    return {
        'nombre': nombre,
        'kb': len(src.encode('utf-8')) // 1024,
        'lineas': len(src.splitlines()),
        'clases': [n.name for n in arbol.body if isinstance(n, ast.ClassDef)],
        'funcs': [n.name for n in arbol.body
                  if isinstance(n, ast.FunctionDef)
                  and not n.name.startswith('_')],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--latex', action='store_true')
    a = p.parse_args(argv)

    for titulo, lista in (('MOTOR', MOTOR), ('ANALISIS', ANALISIS)):
        if not a.latex:
            print(f"\n=== {titulo} ===")
            print(f"  {'modulo':32s} {'KB':>4s} {'lineas':>7s} "
                  f"{'clases':>7s} {'funcs':>6s}")
        for n in lista:
            m = mide(n)
            if not m:
                continue
            if a.latex:
                print(f"\\texttt{{{m['nombre'].replace('_', chr(92)+'_')}}} & "
                      f"{m['lineas']} & {len(m['clases'])} & "
                      f"{len(m['funcs'])} \\\\")
            else:
                print(f"  {m['nombre']:32s} {m['kb']:4d} {m['lineas']:7d} "
                      f"{len(m['clases']):7d} {len(m['funcs']):6d}")
                if m['clases']:
                    print(f"      clases: {', '.join(m['clases'])}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
