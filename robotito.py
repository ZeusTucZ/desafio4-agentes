"""
Robotito reactivo (Desafío 1) adaptado al TestBed.

Regla (la del Desafío 1):
    - Si en el paso anterior NO chocó  -> intenta ir a la derecha.
    - Si en el paso anterior chocó     -> elige al azar arriba, abajo, izquierda o derecha.
Solo percibe "choque" / "sin choque". No conoce su posición, el mapa ni la meta.
La meta solo sirve para que el ambiente termine el episodio.

Uso:
    python robotito.py laberintos/medio.json --semilla 1
    python robotito.py laberintos/medio.json --semilla 1 --max-pasos 900 -o resultado.json

Coordenadas del TestBed: x = columna, y = fila, origen (0, 0) arriba a la izquierda,
y crece hacia abajo. Un movimiento por paso, 4 direcciones.
Códigos de salida: 0 llegó a la meta, 1 no llegó (no se escribe JSON), 2 error de entrada.
Solo usa la biblioteca estándar de Python.
"""
import argparse
import json
import random
import sys
import time

ACCIONES = {"arriba": (0, -1), "abajo": (0, 1), "izquierda": (-1, 0), "derecha": (1, 0)}
NOMBRES = list(ACCIONES)


class LaberintoInvalido(Exception):
    pass


def cargar_laberinto(ruta):
    try:
        with open(ruta, encoding="utf-8") as f:
            d = json.load(f)
        ancho, alto = int(d["gridSize"]["x"]), int(d["gridSize"]["y"])
        obst = {(int(o["x"]), int(o["y"])) for o in d.get("obstaculos", [])}
        inicio = (int(d["start"]["x"]), int(d["start"]["y"]))
        meta = (int(d["goal"]["x"]), int(d["goal"]["y"]))
    except FileNotFoundError:
        raise LaberintoInvalido(f"No se encontró '{ruta}'.")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise LaberintoInvalido(f"'{ruta}' no es un laberinto válido del TestBed: {e!r}")
    for nombre, c in (("start", inicio), ("goal", meta)):
        if not (0 <= c[0] < ancho and 0 <= c[1] < alto):
            raise LaberintoInvalido(f"{nombre} {c} está fuera del grid.")
        if c in obst:
            raise LaberintoInvalido(f"{nombre} {c} está sobre un obstáculo.")
    return ancho, alto, obst, inicio, meta


def correr(ancho, alto, obstaculos, inicio, meta, semilla, max_pasos):
    """Simula al robotito. max_pasos limita los INTENTOS (movimientos + choques).

    Devuelve dict con: llego, ruta (solo celdas a las que se movió, incluye start),
    movimientos, choques, intentos, tiempo."""
    rng = random.Random(semilla)
    pos = inicio
    ruta = [pos]
    choques = 0
    percepcion = "sin_choque"
    intentos = 0
    llego = pos == meta
    t0 = time.perf_counter()
    while not llego and intentos < max_pasos:
        accion = "derecha" if percepcion == "sin_choque" else rng.choice(NOMBRES)
        dx, dy = ACCIONES[accion]
        nueva = (pos[0] + dx, pos[1] + dy)
        intentos += 1
        if 0 <= nueva[0] < ancho and 0 <= nueva[1] < alto and nueva not in obstaculos:
            pos = nueva
            ruta.append(pos)
            percepcion = "sin_choque"
            llego = pos == meta
        else:
            choques += 1
            percepcion = "choque"
    return {"llego": llego, "ruta": ruta, "movimientos": len(ruta) - 1, "choques": choques,
            "intentos": intentos, "tiempo": time.perf_counter() - t0}


def _celdas(lista):
    return [{"x": x, "y": y} for x, y in lista]


def construir_resultado(r):
    return {
        "algoritmo": "aleatorio",
        "ruta": _celdas(r["ruta"]),
        "pasos": r["movimientos"],
        "nodosExplorados": _celdas(r["ruta"]),   # mismas celdas de la ruta
        "tiempoMeta": round(r["tiempo"], 6),
        "tiempoRecolectables": 0,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Robotito reactivo sobre un laberinto del TestBed.")
    ap.add_argument("laberinto")
    ap.add_argument("-o", "--salida", default="resultado.json")
    ap.add_argument("--semilla", type=int, default=1)
    ap.add_argument("--max-pasos", type=int, default=None,
                    help="límite de intentos (por defecto 4 × celdas del grid)")
    args = ap.parse_args(argv)
    try:
        ancho, alto, obst, inicio, meta = cargar_laberinto(args.laberinto)
    except LaberintoInvalido as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    max_pasos = args.max_pasos or 4 * ancho * alto
    r = correr(ancho, alto, obst, inicio, meta, args.semilla, max_pasos)
    if not r["llego"]:
        print(f"NO LLEGÓ a la meta {meta} en {max_pasos} intentos (semilla {args.semilla}): "
              f"{r['movimientos']} movimientos, {r['choques']} choques. No se generó JSON.",
              file=sys.stderr)
        return 1
    with open(args.salida, "w", encoding="utf-8") as f:
        json.dump(construir_resultado(r), f, indent=2, ensure_ascii=False)
    print(f"LLEGÓ a la meta: {r['movimientos']} pasos, {r['choques']} choques, "
          f"{len(set(r['ruta']))} celdas distintas, {r['tiempo']:.6f} s. "
          f"Guardado en '{args.salida}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
