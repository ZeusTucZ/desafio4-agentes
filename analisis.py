"""Tablas y gráficas del reporte, calculadas SOLO desde resultados/resultados.csv."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import astar_agente as A
import qlearning as Q
import robotito as R

SUP, T1, T2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"A*": "#2a78d6", "Q-Learning": "#eb6834", "Robotito": "#1baf7a"}
MARK = {"A*": "s", "Q-Learning": "o", "Robotito": "^"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": T2,
                     "xtick.color": T2, "ytick.color": T2, "text.color": T1, "axes.facecolor": SUP,
                     "figure.facecolor": SUP})

d = pd.read_csv("resultados/resultados.csv")
d["llego"] = d["llego"].astype(int)
GEN = d[d.grupo == "generado"]
# Comparación principal: A*, Robotito y Q-Learning con el 100 % de los episodios (1800)
P = GEN[(GEN.agente != "Q-Learning") | (GEN.episodios_entrenamiento == 1800)].copy()
P["dens_pct"] = (P.densidad_obstaculos * 100).round().astype(int)
AGENTES = ["Robotito", "A*", "Q-Learning"]


def ms(s, dec=1):
    s = s.dropna()
    if len(s) == 0:
        return "—"
    sd = s.std(ddof=1) if len(s) > 1 else 0.0
    return f"{s.mean():.{dec}f} ± {sd:.{dec}f}"


def tabla(df, por):
    filas = []
    for (ag, v), x in df.groupby(["agente", por]):
        ok = x[x.llego == 1]
        filas.append({"agente": ag, por: v, "corridas": len(x),
                      "pct_llego": round(100 * x.llego.mean(), 1),
                      "pasos": ms(ok.pasos), "choques": ms(x.choques if ag == "Robotito" else ok.choques),
                      "nodos": ms(x.nodos_explorados), "tiempo_ms": ms(x.tiempo_s * 1000, 3),
                      "razon_opt": ms(ok.razon_optimalidad, 2)})
    t = pd.DataFrame(filas)
    t["o"] = t.agente.map({a: i for i, a in enumerate(AGENTES)})
    return t.sort_values([por, "o"]).drop(columns="o")


t_dens, t_dist = tabla(P, "dens_pct"), tabla(P, "distancia")
t_asig = tabla(d[d.grupo == "asignado"], "escenario")
t_dens.to_csv("resultados/tabla_por_densidad.csv", index=False)
t_dist.to_csv("resultados/tabla_por_distancia.csv", index=False)
t_asig.to_csv("resultados/tabla_asignados.csv", index=False)

# Q-Learning: barrido de episodios y convergencia
QL = GEN[GEN.agente == "Q-Learning"].copy()
QL["opt"] = (QL.razon_optimalidad == 1).astype(int)
QL["dens_pct"] = (QL.densidad_obstaculos * 100).round().astype(int)
barrido = QL.groupby("episodios_entrenamiento").agg(
    pct_llego=("llego", lambda s: round(100 * s.mean(), 1)),
    pct_optima=("opt", lambda s: round(100 * s.mean(), 1)),
    conv_greedy=("episodios_convergencia", lambda s: ms(s, 0)),
    tiempo_s=("tiempo_s", lambda s: ms(s, 3))).reset_index()
barrido.to_csv("resultados/tabla_barrido_episodios.csv", index=False)
Q18 = QL[QL.episodios_entrenamiento == 1800]
conv_dens = Q18.groupby("dens_pct").agg(greedy=("episodios_convergencia", lambda s: ms(s, 0)),
                                        curva=("episodios_convergencia_curva", lambda s: ms(s, 0)),
                                        choques_ent=("choques_entrenamiento", lambda s: ms(s, 0))).reset_index()
conv_dist = Q18.groupby("distancia").agg(greedy=("episodios_convergencia", lambda s: ms(s, 0)),
                                         curva=("episodios_convergencia_curva", lambda s: ms(s, 0)),
                                         choques_ent=("choques_entrenamiento", lambda s: ms(s, 0))).reset_index()
conv_dens.to_csv("resultados/tabla_convergencia_densidad.csv", index=False)
conv_dist.to_csv("resultados/tabla_convergencia_distancia.csv", index=False)

# Robotito: alcanzabilidad estructural
RB = P[P.agente == "Robotito"]
alc = RB.groupby("dens_pct").apply(lambda x: pd.Series({
    "pct_laberintos_alcanzables": round(100 * x.groupby("escenario").alcanzable_robotito.first().mean(), 1),
    "pct_llego_si_alcanzable": round(100 * x[x.alcanzable_robotito == 1].llego.mean(), 1),
    "pct_llego_total": round(100 * x.llego.mean(), 1)}), include_groups=False).reset_index()
alc.to_csv("resultados/tabla_robotito_alcanzabilidad.csv", index=False)
alc_dist = RB.groupby("distancia").apply(lambda x: pd.Series({
    "pct_laberintos_alcanzables": round(100 * x.groupby("escenario").alcanzable_robotito.first().mean(), 1),
    "pct_llego_si_alcanzable": round(100 * x[x.alcanzable_robotito == 1].llego.mean(), 1),
    "pct_llego_total": round(100 * x.llego.mean(), 1)}), include_groups=False).reset_index()
alc_dist.to_csv("resultados/tabla_robotito_alcanzabilidad_distancia.csv", index=False)


def estilo(ax, xlabel, ylabel, titulo):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(titulo, loc="left", fontsize=11, color=T1)


def lineas(ax, df, por, metrica, solo_llego=False, sd=False, log=False):
    xs = sorted(df[por].unique())
    for ag in AGENTES:
        y, e = [], []
        for x in xs:
            s = df[(df.agente == ag) & (df[por] == x)]
            if solo_llego:
                s = s[s.llego == 1]
            v = s[metrica].dropna()
            y.append(v.mean() if len(v) else np.nan)
            e.append(v.std(ddof=1) if len(v) > 1 else 0)
        y, e = np.array(y), np.array(e)
        ax.plot(xs, y, color=COL[ag], marker=MARK[ag], linewidth=2, markersize=7 if ag != "Q-Learning" else 9,
                markerfacecolor=COL[ag] if ag != "Q-Learning" else SUP, markeredgewidth=2,
                markeredgecolor=COL[ag], label=ag, zorder=3)
        if sd:
            lo = np.maximum(y - e, y * 0.15) if log else y - e
            ax.fill_between(xs, lo, y + e, color=COL[ag], alpha=0.12, linewidth=0)
    ax.set_xticks(xs)
    if log:
        ax.set_yscale("log")


def leyenda(fig, ax):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=3, frameon=False, bbox_to_anchor=(0.99, 1.0))


# --- Fig 1: pasos de la ruta final (solo corridas que llegaron)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
lineas(axs[0], P, "dens_pct", "pasos", True, True, True)
estilo(axs[0], "Densidad de obstáculos (%)", "Pasos de la ruta (escala log)", "Pasos vs densidad")
lineas(axs[1], P, "distancia", "pasos", True, True, True)
estilo(axs[1], "Distancia start-goal (Manhattan)", "Pasos de la ruta (escala log)", "Pasos vs distancia")
leyenda(fig, axs[0])
fig.text(0.01, 0.01, "Media ± 1 desviación estándar, solo corridas que llegaron. A* y Q-Learning coinciden (ruta óptima).",
         fontsize=8.5, color=T2)
fig.tight_layout(rect=(0, 0.04, 1, 0.93))
fig.savefig("figuras/fig1_pasos.png", dpi=150)

# --- Fig 2: % de corridas que llegaron
P2 = P.copy()
P2["pct"] = P2.llego * 100
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
lineas(axs[0], P2, "dens_pct", "pct")
estilo(axs[0], "Densidad de obstáculos (%)", "Corridas que llegaron a la meta (%)", "Éxito vs densidad")
lineas(axs[1], P2, "distancia", "pct")
estilo(axs[1], "Distancia start-goal (Manhattan)", "Corridas que llegaron a la meta (%)", "Éxito vs distancia")
for ax in axs:
    ax.set_ylim(-5, 105)
leyenda(fig, axs[0])
fig.text(0.01, 0.01, "A* y Q-Learning (1800 episodios) se superponen en 100 %.", fontsize=8.5, color=T2)
fig.tight_layout(rect=(0, 0.04, 1, 0.93))
fig.savefig("figuras/fig2_exito.png", dpi=150)

# --- Fig 3: costo de decidir (nodos) y tiempo
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
lineas(axs[0], P, "dens_pct", "nodos_explorados", sd=True)
estilo(axs[0], "Densidad de obstáculos (%)", "Nodos o celdas explorados", "Nodos explorados vs densidad")
lineas(axs[1], P, "distancia", "tiempo_s", log=True)
estilo(axs[1], "Distancia start-goal (Manhattan)", "Tiempo (s, escala log)", "Tiempo de cómputo vs distancia")
leyenda(fig, axs[0])
fig.text(0.01, 0.01, "Q-Learning: celdas distintas visitadas y tiempo de entrenamiento + recorrido. Robotito: celdas distintas de su ruta.",
         fontsize=8.5, color=T2)
fig.tight_layout(rect=(0, 0.04, 1, 0.93))
fig.savefig("figuras/fig3_costo.png", dpi=150)

# --- Fig 4: Q-Learning, episodios de entrenamiento
VIOL = {7: "#c4bdf0", 14: "#8d80e0", 28: "#6250d6"}
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
niveles = sorted(QL.episodios_entrenamiento.unique())
for dist in (7, 14, 28):
    y = [100 * QL[(QL.distancia == dist) & (QL.episodios_entrenamiento == e)].opt.mean() for e in niveles]
    axs[0].plot(range(len(niveles)), y, color=VIOL[dist], marker="o", linewidth=2, markersize=7)

axs[0].annotate("distancia 28", (0, 22.8), xytext=(10, -4), textcoords="offset points", color=T1, fontsize=9)
axs[0].annotate("distancias 7 y 14", (0, 98.5), xytext=(10, -16), textcoords="offset points", color=T1, fontsize=9)
axs[0].set_xticks(range(len(niveles)))
axs[0].set_xticklabels([f"{int(e)}" for e in niveles])
axs[0].set_ylim(-5, 105)
estilo(axs[0], "Episodios de entrenamiento (5, 10, 20, 50 y 100 % de 1800)", "Rutas óptimas (%)",
       "Q-Learning: calidad de la ruta vs episodios")
xs = [7, 14, 28]
mm = [Q18[Q18.distancia == x].episodios_convergencia.mean() for x in xs]
ss = [Q18[Q18.distancia == x].episodios_convergencia.std() for x in xs]
axs[1].errorbar(xs, mm, yerr=ss, color="#6250d6", marker="o", linewidth=2, markersize=7, capsize=4)
axs[1].set_xticks(xs)
estilo(axs[1], "Distancia start-goal (Manhattan)", "Episodios para converger", "Convergencia vs distancia (1800 episodios)")
fig.text(0.01, 0.01, "30 semillas por laberinto y nivel. Convergencia: primer episodio desde el cual la ruta greedy es óptima y se mantiene (media ± 1 DE).",
         fontsize=8.5, color=T2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig("figuras/fig4_qlearning.png", dpi=150)

# --- Fig 5: rutas de los 3 agentes en el mismo laberinto (visualización propia)
def dibujar(nombre_lab, semilla_rob):
    ruta_lab = f"laberintos/generados/{nombre_lab}.json"
    a = R.cargar_laberinto(ruta_lab)
    ancho, alto, obst, ini, meta = a
    lab = Q.Laberinto(ruta_lab)
    rt_a, expl = A.a_estrella(*a)
    eps = 1800
    dec = (0.05 / 1.0) ** (1.0 / (0.7 * eps))
    Qt, hist, vis = Q.entrenar(lab, eps, 0.5, 0.95, 1.0, 0.05, dec, 4 * ancho * alto, 0)
    rt_q = Q.ruta_greedy(lab, Qt, 4 * ancho * alto)[0]
    rb = R.correr(*a, semilla_rob, 4 * ancho * alto)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.4))
    for ax, (ag, ruta, extra) in zip(axs, [
            ("Robotito", rb["ruta"], f"{rb['movimientos']} pasos, {rb['choques']} choques"),
            ("A*", rt_a, f"{len(rt_a) - 1} pasos, {len(expl)} nodos"),
            ("Q-Learning", rt_q, f"{len(rt_q) - 1} pasos, {len(vis)} celdas visitadas")]):
        ax.set_xlim(-0.5, ancho - 0.5)
        ax.set_ylim(alto - 0.5, -0.5)
        ax.set_aspect("equal")
        for x in range(ancho):
            for y in range(alto):
                if (x, y) in obst:
                    ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="#52514e"))
        ax.set_xticks(range(ancho)); ax.set_yticks(range(alto))
        ax.tick_params(labelsize=6, length=0)
        ax.grid(True, color=GRID, linewidth=0.6)
        xs_, ys_ = zip(*ruta)
        ax.plot(xs_, ys_, color=COL[ag], linewidth=2, alpha=0.9, zorder=3)
        ax.scatter(*ini, s=90, color="#0b0b0b", marker="s", zorder=4)
        ax.scatter(*meta, s=130, color="#0b0b0b", marker="*", zorder=4)
        ax.set_title(f"{ag}: {extra}", fontsize=10, loc="left", color=T1)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.text(0.01, 0.01, f"{nombre_lab}. Cuadrado = start, estrella = goal, gris = obstáculos. "
             f"Visualización propia (matplotlib), no es captura del TestBed. Robotito: semilla {semilla_rob}.",
             fontsize=8.5, color=T2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig("figuras/fig5_rutas_ejemplo.png", dpi=150)


# laberinto de ejemplo: el primero del grupo d10_x14 donde el robotito sí llega con alguna semilla
cand = d[(d.grupo == "generado") & (d.agente == "Robotito") & (d.llego == 1) & (d.escenario.str.contains("d10_x14"))]
ej = cand.iloc[(cand.pasos - cand.pasos.median()).abs().argsort()].iloc[0]
dibujar(ej.escenario, int(ej.semilla))

json.dump({"ejemplo": ej.escenario, "semilla_robotito": int(ej.semilla)}, open("resultados/ejemplo_fig5.json", "w"))
for n, t in [("DENSIDAD", t_dens), ("DISTANCIA", t_dist), ("ASIGNADOS", t_asig)]:
    print("==", n); print(t.to_string(index=False))
print("== BARRIDO"); print(barrido.to_string(index=False))
print("== CONV DENS"); print(conv_dens.to_string(index=False)); print(conv_dist.to_string(index=False))
print("== ALC"); print(alc.to_string(index=False)); print(alc_dist.to_string(index=False))
