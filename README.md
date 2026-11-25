# Desafío 4: Robotito vs A* vs Q-Learning

Código del experimento. Los tres agentes leen el mismo JSON de laberinto del TestBed
(coordenadas desde 0, origen arriba a la izquierda, 4 direcciones, un movimiento por paso)
y escriben un JSON de resultado con los 6 campos del TestBed.

## Correr un agente (Python 3.10+, sin instalar nada)

    python robotito.py   laberintos\medio.json --semilla 1 -o resultado_robotito.json
    python astar_agente.py laberintos\medio.json -o resultado_astar.json
    python qlearning.py  laberintos\medio.json            # escribe resultados\medio\resultado.json

`robotito.py` devuelve código 1 y NO escribe JSON si no llegó a la meta (el TestBed solo acepta corridas
que llegaron). Límite de intentos por defecto: 4 × celdas del grid.
En los 3 laberintos asignados el robotito nunca puede llegar (ver `alcanzable.py`), por eso
`testbed/` incluye un laberinto de 15×15 donde sí llega.

## Reproducir el experimento

    python generar_laberintos.py          # 60 laberintos en laberintos\generados\
    python experimento.py                 # resultados\resultados.csv (11,043 corridas, ~3 min con 2 procesos)
    pip install -r requirements.txt
    python analisis.py                    # tablas en resultados\ y gráficas en figuras\

Todo depende de semillas, así que el CSV se reproduce exactamente (salvo los tiempos).
