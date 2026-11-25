"""
Experimento completo del Desafío 4: Robotito vs A* vs Q-Learning en los mismos laberintos.

    python experimento.py [--semillas 30] [--semillas-barrido 10] [--procesos 2]

Escribe resultados/resultados.csv (una fila por corrida). Reproducible: todo depende de semillas.
"""
import argparse
import csv
import glob
import os
import statistics as st
import sys
import time
from multiprocessing import Pool

import alcanzable
import astar_agente as A
import qlearning as Q
import robotito as R

SALIDA = "resultados/resultados.csv"
COLUMNAS = ["agente", "escenario", "densidad_obstaculos", "semilla", "llego", "pasos", "choques",
            "nodos_explorados", "episodios_entrenamiento", "tiempo_s",
            # variables y medidas adicionales
            "distancia", "grupo", "optimo_bfs", "razon_optimalidad", "episodios_convergencia",
            "episodios_convergencia_curva", "choques_entrenamiento", "intentos",
            "alcanzable_robotito"]
ALFA, GAMMA, E_INI, E_FIN = 0.5, 0.95, 1.0, 0.05
NIVELES = [0.05, 0.10, 0.20, 0.50, 1.00]       # fracción de los episodios por defecto


def media_movil(v, w=20):
    out, s = [], 0.0
    for i, x in enumerate(v):
        s += x
        if i >= w:
            s -= v[i - w]
        out.append(s / min(i + 1, w))
    return out


def conv_curva(pasos, optimo, w=20, factor=1.5):
    """Primer episodio desde el cual la media móvil de pasos nunca supera factor × óptimo."""
    mm = media_movil(pasos, w)
    malo = 0
    for i, x in enumerate(mm, start=1):
        if i >= w and x > factor * optimo:
            malo = i
    return malo + 1


def fila(**kw):
    return [kw.get(c, "") for c in COLUMNAS]


def correr_escenario(tarea):
    ruta, grupo, semillas, semillas_barrido = tarea
    escenario = os.path.splitext(os.path.basename(ruta))[0]
    ancho, alto, obst, inicio, meta = R.cargar_laberinto(ruta)
    celdas = ancho * alto
    dens = round(len(obst) / celdas, 4)
    dist = abs(inicio[0] - meta[0]) + abs(inicio[1] - meta[1])
    lab = Q.Laberinto(ruta)
    optimo = lab.bfs_longitud_optima()
    alc = int(alcanzable.alcanzable(ancho, alto, obst, inicio, meta))
    comunes = dict(escenario=escenario, densidad_obstaculos=dens, distancia=dist, grupo=grupo,
                   optimo_bfs=optimo, alcanzable_robotito=alc)
    filas = []

    # ---- A*: una corrida; el tiempo es la mediana de 10 repeticiones de la búsqueda
    tiempos = []
    for _ in range(10):
        t0 = time.perf_counter()
        r, expl = A.a_estrella(ancho, alto, obst, inicio, meta)
        tiempos.append(time.perf_counter() - t0)
    filas.append(fila(agente="A*", llego=int(r is not None), pasos=len(r) - 1,
                      choques=0, nodos_explorados=len(expl), tiempo_s=round(st.median(tiempos), 6),
                      razon_optimalidad=round((len(r) - 1) / optimo, 4), intentos=len(r) - 1,
                      **comunes))

    # ---- Robotito
    max_pasos = 4 * celdas
    for s in range(semillas):
        x = R.correr(ancho, alto, obst, inicio, meta, s, max_pasos)
        filas.append(fila(agente="Robotito", semilla=s, llego=int(x["llego"]),
                          pasos=x["movimientos"], choques=x["choques"],
                          nodos_explorados=len(set(x["ruta"])), tiempo_s=round(x["tiempo"], 6),
                          razon_optimalidad=round(x["movimientos"] / optimo, 4) if x["llego"] else "",
                          intentos=x["intentos"], **comunes))

    # ---- Q-Learning
    eps_defecto = max(1000, 8 * celdas)
    if grupo == "generado":
        niveles = [(round(eps_defecto * f), semillas if f == 1.0 else semillas_barrido) for f in NIVELES]
    else:
        niveles = [(eps_defecto, semillas)]
    for eps, n_sem in niveles:
        dec = (E_FIN / E_INI) ** (1.0 / (0.7 * eps))
        for s in range(n_sem):
            estado = {"malo": 0}

            def cb(ep, Qt):
                if ep % 5:
                    return
                rt = Q.ruta_greedy(lab, Qt, max_pasos)
                if not (rt[3] and len(rt[0]) - 1 == optimo):
                    estado["malo"] = ep

            t0 = time.perf_counter()
            Qt, hist, vis = Q.entrenar(lab, eps, ALFA, GAMMA, E_INI, E_FIN, dec, max_pasos, s,
                                       al_terminar_episodio=cb)
            t_entr = time.perf_counter() - t0
            t1 = time.perf_counter()
            rt, _, _, llego, motivo = Q.ruta_greedy(lab, Qt, max_pasos)
            t_rec = time.perf_counter() - t1
            pasos = len(rt) - 1
            optima = bool(llego and pasos == optimo)
            conv = min(eps, estado["malo"] + 5) if optima else ""
            conv_c = conv_curva([h["pasos"] for h in hist], optimo) if optima else ""
            filas.append(fila(agente="Q-Learning", semilla=s, llego=int(llego),
                              pasos=pasos if llego else "", choques=0 if llego else 1,
                              nodos_explorados=len(vis), episodios_entrenamiento=eps,
                              tiempo_s=round(t_entr + t_rec, 6),
                              razon_optimalidad=round(pasos / optimo, 4) if llego else "",
                              episodios_convergencia=conv, episodios_convergencia_curva=conv_c,
                              choques_entrenamiento=sum(h["choques"] for h in hist),
                              intentos=sum(h["pasos"] for h in hist), **comunes))
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, default=30)
    ap.add_argument("--semillas-barrido", type=int, default=10)
    ap.add_argument("--procesos", type=int, default=2)
    ap.add_argument("--solo", default=None, help="subcadena para filtrar laberintos (pruebas)")
    ap.add_argument("--salida", default=SALIDA)
    args = ap.parse_args()

    tareas = [(f"laberintos/{n}.json", "asignado", args.semillas, args.semillas_barrido)
              for n in ("facil", "medio", "dificil")]
    tareas += [(f, "generado", args.semillas, args.semillas_barrido)
               for f in sorted(glob.glob("laberintos/generados/*.json"))]
    if args.solo:
        tareas = [t for t in tareas if args.solo in t[0]]
    os.makedirs(os.path.dirname(args.salida), exist_ok=True)
    t0 = time.time()
    with open(args.salida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNAS)
        with Pool(args.procesos) as p:
            for i, filas in enumerate(p.imap_unordered(correr_escenario, tareas), 1):
                w.writerows(filas)
                f.flush()
                print(f"[{i}/{len(tareas)}] {filas[0][1]} listo ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
