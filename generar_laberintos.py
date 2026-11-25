"""Genera los laberintos del experimento (15×15) en formato JSON del TestBed.

9+3 configuraciones: densidad {0, 10, 20, 30 %} × distancia start-goal {7, 14, 28}
(Manhattan). El goal está fijo en la esquina inferior derecha (14, 14), como en los tres
laberintos del TestBed, y el start se coloca al azar a esa distancia (28 = esquina (0, 0)).
5 laberintos por configuración (semillas 0-4).
Solo se aceptan laberintos con solución (BFS). Todo es reproducible.
"""
import json
import os
import random
from collections import deque

N = 15
DENSIDADES = [0.0, 0.10, 0.20, 0.30]
DISTANCIAS = [7, 14, 28]
SEMILLAS_LAB = range(5)
SALIDA = "laberintos/generados"


def hay_ruta(obst, inicio, meta, n=N):
    vistos = {inicio}
    cola = deque([inicio])
    while cola:
        x, y = cola.popleft()
        if (x, y) == meta:
            return True
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            v = (x + dx, y + dy)
            if 0 <= v[0] < n and 0 <= v[1] < n and v not in obst and v not in vistos:
                vistos.add(v)
                cola.append(v)
    return False


def nombre(dens, dist, s):
    return f"lab_{N}x{N}_d{int(round(dens * 100)):02d}_x{dist:02d}_m{s}"


def generar(dens, dist, s):
    rng = random.Random(f"{dens}-{dist}-{s}")
    meta = (N - 1, N - 1)
    candidatos = sorted((N - 1 - x, N - 1 - (dist - x)) for x in range(N) if 0 <= dist - x < N)
    inicio = rng.choice(candidatos)
    libres = [(x, y) for x in range(N) for y in range(N) if (x, y) not in (inicio, meta)]
    k = round(dens * N * N)
    for _ in range(10000):
        obst = set(rng.sample(libres, k))
        if hay_ruta(obst, inicio, meta):
            return inicio, meta, obst
    raise RuntimeError("no se encontró un laberinto con solución")


def main():
    os.makedirs(SALIDA, exist_ok=True)
    for dens in DENSIDADES:
        for dist in DISTANCIAS:
            for s in SEMILLAS_LAB:
                inicio, meta, obst = generar(dens, dist, s)
                datos = {
                    "gridSize": {"x": N, "y": N},
                    "obstaculos": [{"x": x, "y": y} for x, y in sorted(obst)],
                    "recolectables": [],
                    "start": {"x": inicio[0], "y": inicio[1]},
                    "goal": {"x": meta[0], "y": meta[1]},
                }
                with open(f"{SALIDA}/{nombre(dens, dist, s)}.json", "w", encoding="utf-8") as f:
                    json.dump(datos, f, indent=2)
    print("laberintos generados:", len(os.listdir(SALIDA)))


if __name__ == "__main__":
    main()
