"""
Desafío 2: agente basado en metas con A* (heurística Manhattan).

Uso:
    python astar_agente.py laberintos/laberinto_1.json
    python astar_agente.py laberintos/laberinto_1.json -o resultado.json

Lee el JSON del laberinto (formato TestBed), busca la ruta más corta de
`start` a `goal` y escribe un JSON de resultado listo para subir al TestBed.

Coordenadas: x = columna, y = fila, origen (0, 0) arriba a la izquierda,
y crece hacia abajo. Movimiento en 4 direcciones, costo 1 por paso.
Solo usa la biblioteca estándar de Python.
"""
import argparse
import heapq
import itertools
import json
import sys
import time

# Orden fijo de vecinos (afecta solo el desempate, no la longitud de la ruta):
# arriba, derecha, abajo, izquierda. Se expresan como (dx, dy).
MOVIMIENTOS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


class LaberintoInvalido(Exception):
    """El JSON del laberinto está mal formado o es inconsistente."""


# ---------------------------------------------------------------- entrada
def cargar_laberinto(ruta_archivo):
    """Lee y valida el JSON. Devuelve (ancho, alto, obstaculos, inicio, meta)."""
    try:
        with open(ruta_archivo, encoding="utf-8") as f:
            datos = json.load(f)
    except FileNotFoundError:
        raise LaberintoInvalido(f"No se encontró el archivo '{ruta_archivo}'.")
    except json.JSONDecodeError as e:
        raise LaberintoInvalido(f"'{ruta_archivo}' no es un JSON válido: {e}")

    try:
        ancho = int(datos["gridSize"]["x"])
        alto = int(datos["gridSize"]["y"])
        obstaculos = {(int(o["x"]), int(o["y"])) for o in datos.get("obstaculos", [])}
        inicio = (int(datos["start"]["x"]), int(datos["start"]["y"]))
        meta = (int(datos["goal"]["x"]), int(datos["goal"]["y"]))
    except (KeyError, TypeError, ValueError) as e:
        raise LaberintoInvalido(f"Falta o es inválido un campo del laberinto: {e!r}")
    # "recolectables" se ignora a propósito en este desafío.

    def dentro(c):
        return 0 <= c[0] < ancho and 0 <= c[1] < alto

    if ancho <= 0 or alto <= 0:
        raise LaberintoInvalido("gridSize debe ser positivo.")
    if not dentro(inicio):
        raise LaberintoInvalido(f"start {inicio} está fuera del grid.")
    if not dentro(meta):
        raise LaberintoInvalido(f"goal {meta} está fuera del grid.")
    if inicio in obstaculos:
        raise LaberintoInvalido(f"start {inicio} está sobre un obstáculo.")
    if meta in obstaculos:
        raise LaberintoInvalido(f"goal {meta} está sobre un obstáculo.")
    return ancho, alto, obstaculos, inicio, meta


# ---------------------------------------------------------------- A*
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_estrella(ancho, alto, obstaculos, inicio, meta):
    """
    A* con heurística Manhattan (admisible y consistente en una malla de
    4 direcciones con costo 1), así que la primera vez que se saca la meta
    de la lista abierta su ruta es óptima.

    Desempate en la lista abierta (de mayor a menor prioridad):
      1) menor f = g + h
      2) menor h (el más cercano a la meta: se profundiza hacia la meta)
      3) el que entró primero a la lista (contador creciente, FIFO)

    Devuelve (ruta, explorados). ruta es None si no existe camino.
    """
    contador = itertools.count()
    g = {inicio: 0}
    padre = {inicio: None}
    h0 = manhattan(inicio, meta)
    abierta = [(h0, h0, next(contador), inicio)]   # (f, h, orden, celda)
    cerrada = set()
    explorados = []

    while abierta:
        _, _, _, actual = heapq.heappop(abierta)
        if actual in cerrada:          # entrada vieja (ya hubo un mejor g)
            continue
        cerrada.add(actual)
        explorados.append(actual)      # "sacado de la lista abierta"

        if actual == meta:
            ruta = []
            while actual is not None:
                ruta.append(actual)
                actual = padre[actual]
            ruta.reverse()
            return ruta, explorados

        for dx, dy in MOVIMIENTOS:
            vecino = (actual[0] + dx, actual[1] + dy)
            if not (0 <= vecino[0] < ancho and 0 <= vecino[1] < alto):
                continue
            if vecino in obstaculos or vecino in cerrada:
                continue
            g_nuevo = g[actual] + 1
            if g_nuevo < g.get(vecino, float("inf")):
                g[vecino] = g_nuevo
                padre[vecino] = actual
                h = manhattan(vecino, meta)
                heapq.heappush(abierta, (g_nuevo + h, h, next(contador), vecino))

    return None, explorados


# ---------------------------------------------------------------- salida
def _lista(celdas):
    return [{"x": x, "y": y} for x, y in celdas]


def construir_resultado(ruta, explorados, tiempo):
    return {
        "algoritmo": "A*",
        "ruta": _lista(ruta),
        "pasos": len(ruta) - 1,
        "nodosExplorados": _lista(explorados),
        "tiempoMeta": round(tiempo, 6),
        "tiempoRecolectables": 0,
    }


def resolver(ruta_laberinto):
    """Carga, resuelve y mide el tiempo SOLO de la búsqueda."""
    ancho, alto, obstaculos, inicio, meta = cargar_laberinto(ruta_laberinto)
    t0 = time.perf_counter()
    ruta, explorados = a_estrella(ancho, alto, obstaculos, inicio, meta)
    tiempo = time.perf_counter() - t0
    if ruta is None:
        return None, inicio, meta, len(explorados)
    return construir_resultado(ruta, explorados, tiempo), inicio, meta, len(explorados)


def main(argv=None):
    ap = argparse.ArgumentParser(description="A* (Manhattan) sobre un laberinto del TestBed.")
    ap.add_argument("laberinto", help="JSON del laberinto descargado del TestBed")
    ap.add_argument("-o", "--salida", default="resultado.json",
                    help="archivo de resultado (por defecto: resultado.json)")
    args = ap.parse_args(argv)

    try:
        resultado, inicio, meta, n_exp = resolver(args.laberinto)
    except LaberintoInvalido as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if resultado is None:
        print(f"NO EXISTE RUTA de {inicio} a {meta}: la meta es inalcanzable "
              f"(A* agotó la lista abierta tras explorar {n_exp} celdas). "
              f"No se generó archivo de resultado.", file=sys.stderr)
        return 1

    with open(args.salida, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    print(f"Ruta encontrada: {resultado['pasos']} pasos, "
          f"{len(resultado['nodosExplorados'])} nodos explorados, "
          f"{resultado['tiempoMeta']} s. Guardado en '{args.salida}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
