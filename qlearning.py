"""
Q-Learning tabular para resolver los laberintos del TestBed.

Uso:
    # Entrenar (y generar todos los archivos de salida)
    python qlearning.py laberintos/facil.json
    python qlearning.py laberintos/dificil.json --episodios 5000 --semilla 7

    # Regenerar la ruta aprendida a partir de una tabla Q guardada, SIN reentrenar
    python qlearning.py laberintos/facil.json --cargar-tabla resultados/facil/tabla_q.json

Genera, en <salida>/<nombre_laberinto>/ (por defecto ./resultados/<nombre>/):
    resultado.json              -> JSON para subir al TestBed (mismo formato del Desafío 2)
    entrenamiento.csv           -> una fila por episodio
    tabla_q.json                -> tabla Q (valores completos) + metadatos; se puede recargar
    tabla_q.csv                 -> la misma tabla Q en formato legible
    resumen_entrenamiento.json  -> detalles extra (hiperparámetros, validación con BFS, etc.)

Solo usa la biblioteca estándar de Python.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import deque

# ---------------------------------------------------------------------------
# Definición del problema
# ---------------------------------------------------------------------------
# Acciones: (nombre, dx, dy). Coordenadas de pantalla: origen arriba a la izquierda,
# "arriba" disminuye y, "abajo" aumenta y.
ACCIONES = [
    ("arriba", 0, -1),
    ("abajo", 0, 1),
    ("izquierda", -1, 0),
    ("derecha", 1, 0),
]
NOMBRES_ACCIONES = [a[0] for a in ACCIONES]

# Recompensas
R_MOVIMIENTO = -1    # moverse a una celda libre (incentiva rutas cortas)
R_CHOQUE = -10       # intentar entrar a un obstáculo o salir del mapa (no se mueve)
R_META = 10          # llegar a la meta (termina el episodio)


class Laberinto:
    """Ambiente construido a partir del JSON del TestBed.

    Solo el ambiente conoce obstáculos y meta; el agente únicamente recibe
    (nuevo_estado, recompensa, terminado) al intentar una acción."""

    def __init__(self, ruta_json):
        try:
            with open(ruta_json, encoding="utf-8") as f:
                datos = json.load(f)
            self.ancho = int(datos["gridSize"]["x"])
            self.alto = int(datos["gridSize"]["y"])
            self.obstaculos = {(int(o["x"]), int(o["y"])) for o in datos.get("obstaculos", [])}
            self.start = (int(datos["start"]["x"]), int(datos["start"]["y"]))
            self.goal = (int(datos["goal"]["x"]), int(datos["goal"]["y"]))
        except FileNotFoundError:
            raise ValueError(f"No se encontró el archivo '{ruta_json}'.")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise ValueError(f"'{ruta_json}' no es un laberinto válido del TestBed: {e!r}")
        # "recolectables" se ignora a propósito en este desafío.
        if self.ancho <= 0 or self.alto <= 0:
            raise ValueError("gridSize debe ser positivo.")
        for nombre, c in (("start", self.start), ("goal", self.goal)):
            if not (0 <= c[0] < self.ancho and 0 <= c[1] < self.alto):
                raise ValueError(f"{nombre} {c} está fuera del grid.")
            if c in self.obstaculos:
                raise ValueError(f"{nombre} {c} está sobre un obstáculo.")

    def es_libre(self, x, y):
        return 0 <= x < self.ancho and 0 <= y < self.alto and (x, y) not in self.obstaculos

    def paso(self, estado, accion):
        """Aplica una acción. Devuelve (nuevo_estado, recompensa, terminado)."""
        _, dx, dy = ACCIONES[accion]
        nx, ny = estado[0] + dx, estado[1] + dy
        if not self.es_libre(nx, ny):
            return estado, R_CHOQUE, False
        nuevo = (nx, ny)
        if nuevo == self.goal:
            return nuevo, R_META, True
        return nuevo, R_MOVIMIENTO, False

    def bfs_longitud_optima(self):
        """Longitud (en pasos) del camino más corto start->goal, o None si no existe.
        Solo se usa para avisar si no hay ruta y para validar; el agente no la conoce."""
        dist = {self.start: 0}
        cola = deque([self.start])
        while cola:
            s = cola.popleft()
            if s == self.goal:
                return dist[s]
            for _, dx, dy in ACCIONES:
                n = (s[0] + dx, s[1] + dy)
                if self.es_libre(*n) and n not in dist:
                    dist[n] = dist[s] + 1
                    cola.append(n)
        return None


# ---------------------------------------------------------------------------
# Agente Q-Learning
# ---------------------------------------------------------------------------
def argmax_aleatorio(valores, rng):
    """Índice del máximo; desempata al azar (evita sesgo hacia la primera acción)."""
    m = max(valores)
    return rng.choice([i for i, v in enumerate(valores) if v == m])


def argmax_determinista(valores):
    """Índice del máximo; ante empate elige la primera acción (recorrido reproducible)."""
    return max(range(len(valores)), key=lambda i: valores[i])


def entrenar(lab, episodios, alfa, gamma, eps_inicial, eps_final, eps_decaimiento,
             max_pasos, semilla, al_terminar_episodio=None):
    """Devuelve (Q, historial, visitadas).

    visitadas: celdas distintas que el agente ocupó durante TODO el entrenamiento,
    en el orden de su primera visita.
    al_terminar_episodio: función opcional f(episodio, Q) que se llama al final de cada
    episodio (sirve para evaluar la política durante el entrenamiento; no afecta el
    aprendizaje ni la semilla)."""
    rng = random.Random(semilla)
    # Tabla Q: un valor por celda y por cada una de las 4 acciones, inicializada en 0.
    Q = {(x, y): [0.0] * len(ACCIONES) for y in range(lab.alto) for x in range(lab.ancho)}
    historial = []
    visitadas = [lab.start]
    vistas = {lab.start}
    epsilon = eps_inicial

    for ep in range(1, episodios + 1):
        s = lab.start
        recompensa_total = 0
        choques = 0
        llego = False
        pasos = 0

        while pasos < max_pasos:
            # Política ε-greedy
            if rng.random() < epsilon:
                a = rng.randrange(len(ACCIONES))
            else:
                a = argmax_aleatorio(Q[s], rng)

            s2, r, terminado = lab.paso(s, a)
            pasos += 1
            recompensa_total += r
            if r == R_CHOQUE:
                choques += 1
            if s2 not in vistas:
                vistas.add(s2)
                visitadas.append(s2)

            # Q(s,a) <- Q(s,a) + α [ r + γ·max_a' Q(s',a') − Q(s,a) ]
            # En un estado terminal no hay futuro: max_a' Q(s',a') = 0.
            objetivo = r if terminado else r + gamma * max(Q[s2])
            Q[s][a] += alfa * (objetivo - Q[s][a])

            s = s2
            if terminado:
                llego = True
                break

        historial.append({
            "episodio": ep,
            "pasos": pasos,
            "recompensa_total": recompensa_total,
            "choques": choques,
            "llego_meta": llego,
            "epsilon": round(epsilon, 6),
        })
        epsilon = max(eps_final, epsilon * eps_decaimiento)
        if al_terminar_episodio is not None:
            al_terminar_episodio(ep, Q)

    return Q, historial, visitadas


def ruta_greedy(lab, Q, max_pasos):
    """Recorre desde start eligiendo siempre argmax Q (sin explorar)."""
    s = lab.start
    ruta = [s]
    acciones = []
    recompensa = 0
    visitados = {s}
    llego = False
    motivo = "limite_pasos"

    for _ in range(max_pasos):
        a = argmax_determinista(Q[s])
        s2, r, terminado = lab.paso(s, a)
        acciones.append(NOMBRES_ACCIONES[a])
        recompensa += r
        if s2 == s:  # chocó: con política determinista se quedaría atascado para siempre
            motivo = "choque"
            break
        ruta.append(s2)
        s = s2
        if terminado:
            llego = True
            motivo = "meta"
            break
        if s in visitados:  # ciclo: la política greedy nunca llegará
            motivo = "ciclo"
            break
        visitados.add(s)

    return ruta, acciones, recompensa, llego, motivo


# ---------------------------------------------------------------------------
# Entrada / salida
# ---------------------------------------------------------------------------
def _celdas(lista):
    return [{"x": x, "y": y} for x, y in lista]


def guardar_tabla_q(carpeta, lab, ruta_json, params, Q, visitadas, tiempo_entrenamiento):
    """tabla_q.json con valores completos (sin redondear) y metadatos para recargarla."""
    tabla = [
        {"x": x, "y": y, **{n: v for n, v in zip(NOMBRES_ACCIONES, Q[(x, y)])}}
        for y in range(lab.alto) for x in range(lab.ancho)
    ]
    contenido = {
        "laberinto": os.path.basename(ruta_json),
        "gridSize": {"x": lab.ancho, "y": lab.alto},
        "acciones": NOMBRES_ACCIONES,
        "parametros": params,
        "tiempo_entrenamiento_s": tiempo_entrenamiento,
        "nodosExplorados": _celdas(visitadas),
        "q": tabla,
    }
    with open(os.path.join(carpeta, "tabla_q.json"), "w", encoding="utf-8") as f:
        json.dump(contenido, f, indent=1)


def cargar_tabla_q(ruta_tabla, lab):
    """Lee tabla_q.json. Devuelve (Q, meta) donde meta trae parámetros, tiempo y nodos."""
    try:
        with open(ruta_tabla, encoding="utf-8") as f:
            datos = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"No se encontró la tabla Q '{ruta_tabla}'.")
    except json.JSONDecodeError as e:
        raise ValueError(f"'{ruta_tabla}' no es un JSON válido: {e}")
    try:
        gs = datos["gridSize"]
        if (int(gs["x"]), int(gs["y"])) != (lab.ancho, lab.alto):
            raise ValueError(
                f"La tabla Q es de un grid {gs['x']}x{gs['y']} y el laberinto es "
                f"{lab.ancho}x{lab.alto}. ¿Cargaste la tabla de otro laberinto?")
        Q = {(x, y): [0.0] * len(ACCIONES) for y in range(lab.alto) for x in range(lab.ancho)}
        for fila in datos["q"]:
            Q[(int(fila["x"]), int(fila["y"]))] = [float(fila[n]) for n in NOMBRES_ACCIONES]
    except (KeyError, TypeError) as e:
        raise ValueError(f"'{ruta_tabla}' no tiene el formato esperado de tabla Q: {e!r}")
    return Q, datos


def escribir_entrenamiento_y_tablas(carpeta, lab, Q, historial):
    with open(os.path.join(carpeta, "entrenamiento.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(historial[0].keys()))
        w.writeheader()
        w.writerows(historial)

    with open(os.path.join(carpeta, "tabla_q.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "tipo"] + NOMBRES_ACCIONES + ["mejor_accion"])
        for y in range(lab.alto):
            for x in range(lab.ancho):
                if (x, y) in lab.obstaculos:
                    tipo = "obstaculo"
                elif (x, y) == lab.goal:
                    tipo = "meta"
                elif (x, y) == lab.start:
                    tipo = "inicio"
                else:
                    tipo = "libre"
                vals = Q[(x, y)]
                mejor = NOMBRES_ACCIONES[argmax_determinista(vals)] if tipo in ("libre", "inicio") else ""
                w.writerow([x, y, tipo] + [round(v, 6) for v in vals] + [mejor])


def dibujar(lab, ruta):
    """Mapa en texto: S inicio, G meta, # obstáculo, * ruta aprendida."""
    en_ruta = set(ruta)
    filas = []
    for y in range(lab.alto):
        fila = []
        for x in range(lab.ancho):
            c = (x, y)
            if c == lab.start:
                fila.append("S")
            elif c == lab.goal:
                fila.append("G")
            elif c in lab.obstaculos:
                fila.append("#")
            elif c in en_ruta:
                fila.append("*")
            else:
                fila.append(".")
        filas.append(" ".join(fila))
    return "\n".join(filas)


def main(argv=None):
    p = argparse.ArgumentParser(description="Q-Learning tabular sobre un laberinto del TestBed")
    p.add_argument("laberinto", help="ruta al JSON del laberinto")
    p.add_argument("--cargar-tabla", metavar="TABLA_Q.json", default=None,
                   help="no entrena: carga esta tabla Q y regenera la ruta aprendida")
    p.add_argument("--episodios", type=int, default=None,
                   help="número de episodios (por defecto max(1000, 8 × celdas))")
    p.add_argument("--alfa", type=float, default=0.5, help="tasa de aprendizaje α")
    p.add_argument("--gamma", type=float, default=0.95, help="factor de descuento γ")
    p.add_argument("--eps-inicial", type=float, default=1.0)
    p.add_argument("--eps-final", type=float, default=0.05)
    p.add_argument("--max-pasos", type=int, default=None,
                   help="límite de pasos por episodio (por defecto 4 × celdas)")
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--salida", default="resultados", help="carpeta base de salida")
    args = p.parse_args(argv)

    try:
        lab = Laberinto(args.laberinto)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    optimo = lab.bfs_longitud_optima()
    if optimo is None:
        print(f"NO EXISTE RUTA de {lab.start} a {lab.goal}: la meta es inalcanzable. "
              f"No se entrena ni se genera archivo de resultado.", file=sys.stderr)
        return 1

    nombre = os.path.splitext(os.path.basename(args.laberinto))[0]
    carpeta = os.path.join(args.salida, nombre)
    os.makedirs(carpeta, exist_ok=True)
    n_celdas = lab.ancho * lab.alto

    if args.cargar_tabla:
        # ---- Modo recarga: sin entrenamiento ----
        try:
            Q, meta = cargar_tabla_q(args.cargar_tabla, lab)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        params = meta.get("parametros", {})
        max_pasos = params.get("max_pasos_episodio", 4 * n_celdas)
        tiempo_entrenamiento = float(meta.get("tiempo_entrenamiento_s", 0.0))
        visitadas = [(int(c["x"]), int(c["y"])) for c in meta.get("nodosExplorados", [])]
        historial = None
        print(f"Tabla Q cargada de '{args.cargar_tabla}' (sin reentrenar).")
    else:
        # ---- Modo entrenamiento ----
        episodios = args.episodios or max(1000, 8 * n_celdas)
        max_pasos = args.max_pasos or 4 * n_celdas
        # ε decae exponencialmente hasta eps_final al ~70 % de los episodios.
        eps_decaimiento = (args.eps_final / args.eps_inicial) ** (1.0 / (0.7 * episodios))
        params = {
            "episodios": episodios,
            "alfa": args.alfa,
            "gamma": args.gamma,
            "epsilon_inicial": args.eps_inicial,
            "epsilon_final": args.eps_final,
            "epsilon_decaimiento": round(eps_decaimiento, 8),
            "max_pasos_episodio": max_pasos,
            "semilla": args.semilla,
        }
        t0 = time.perf_counter()
        Q, historial, visitadas = entrenar(lab, episodios, args.alfa, args.gamma,
                                           args.eps_inicial, args.eps_final,
                                           eps_decaimiento, max_pasos, args.semilla)
        tiempo_entrenamiento = time.perf_counter() - t0

    # ---- Recorrido final: argmax sin explorar ----
    t0 = time.perf_counter()
    ruta, acciones, recompensa, llego, motivo = ruta_greedy(lab, Q, max_pasos)
    tiempo_recorrido = time.perf_counter() - t0
    tiempo_meta = tiempo_entrenamiento + tiempo_recorrido

    # ---- Archivos de entrenamiento (solo si se entrenó) ----
    if historial is not None:
        escribir_entrenamiento_y_tablas(carpeta, lab, Q, historial)
        guardar_tabla_q(carpeta, lab, args.laberinto, params, Q, visitadas, tiempo_entrenamiento)

    pasos = len(ruta) - 1
    resumen = {
        "laberinto": os.path.basename(args.laberinto),
        "gridSize": {"x": lab.ancho, "y": lab.alto},
        "llego_meta": llego,
        "motivo_fin": motivo,
        "pasos": pasos,
        "longitud_optima_bfs": optimo,
        "es_ruta_optima": llego and pasos == optimo,
        "recompensa_total_ruta": recompensa,
        "acciones": acciones,
        "celdas_visitadas_entrenamiento": len(visitadas),
        "episodios_que_llegaron": (sum(h["llego_meta"] for h in historial)
                                   if historial is not None else None),
        "tiempo_entrenamiento_s": round(tiempo_entrenamiento, 6),
        "tiempo_recorrido_s": round(tiempo_recorrido, 6),
        "parametros": params,
        "recompensas": {"movimiento": R_MOVIMIENTO, "choque": R_CHOQUE, "meta": R_META},
    }
    with open(os.path.join(carpeta, "resumen_entrenamiento.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    ruta_resultado = os.path.join(carpeta, "resultado.json")
    if not llego:
        # El TestBed rechazaría este archivo: no se genera y se borra uno viejo.
        if os.path.exists(ruta_resultado):
            os.remove(ruta_resultado)
        print(f"ERROR: la ruta aprendida NO llega a la meta (motivo: {motivo}). "
              f"El agente no aprendió lo suficiente: aumenta --episodios o ajusta "
              f"--alfa/--gamma/--eps-final y reporta qué cambiaste. "
              f"No se generó resultado.json.", file=sys.stderr)
        return 1

    # JSON de resultado: exactamente el formato del Desafío 2.
    resultado = {
        "algoritmo": "Q-Learning",
        "ruta": _celdas(ruta),
        "pasos": pasos,
        "nodosExplorados": _celdas(visitadas),
        "tiempoMeta": round(tiempo_meta, 6),
        "tiempoRecolectables": 0,
    }
    with open(ruta_resultado, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    print(f"Laberinto: {nombre} ({lab.ancho}x{lab.alto}, {len(lab.obstaculos)} obstáculos)")
    if historial is not None:
        exitos = resumen["episodios_que_llegaron"]
        print(f"Episodios: {params['episodios']}  |  llegaron a la meta: {exitos} "
              f"({100 * exitos / params['episodios']:.1f} %)  |  entrenamiento: {tiempo_entrenamiento:.3f} s")
    print(f"Ruta aprendida: llega a la meta en {pasos} pasos  |  óptimo BFS: {optimo}"
          f"  |  celdas visitadas en el entrenamiento: {len(visitadas)}")
    print(dibujar(lab, ruta))
    print(f"Archivos escritos en: {carpeta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
