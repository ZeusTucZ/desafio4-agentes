"""Diagramas de agente de los tres agentes (matplotlib)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SUP, T1, T2 = "#fcfcfb", "#0b0b0b", "#52514e"
AZ, NA, AQ = "#2a78d6", "#eb6834", "#1baf7a"


def caja(ax, x, y, w, h, titulo, texto="", color="#e4e3df", borde=T2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec=borde, lw=1.4, ls=ls))
    ax.text(x + w / 2, y + h - 0.22, titulo, ha="center", va="top", fontsize=10, fontweight="bold", color=T1)
    if texto:
        ax.text(x + w / 2, y + h - 0.62, texto, ha="center", va="top", fontsize=8.5, color=T1, linespacing=1.3)


def flecha(ax, a, b, texto="", off=(0, 0.12)):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=T1, lw=1.5))
    if texto:
        ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], texto, ha="center", fontsize=8.5, color=T2)


def retorno(ax, x0, y0, ybajo, x1, y1, texto):
    ax.plot([x0, x0, x1], [y0, ybajo, ybajo], color=T1, lw=1.5)
    ax.annotate("", xy=(x1, y1), xytext=(x1, ybajo), arrowprops=dict(arrowstyle="-|>", color=T1, lw=1.5))
    ax.text((x0 + x1) / 2, ybajo + 0.08, texto, ha="center", fontsize=8.5, color=T2)


def base(titulo, ancho=11, alto=5.2):
    fig, ax = plt.subplots(figsize=(ancho, alto))
    ax.set_xlim(0, ancho); ax.set_ylim(0, alto); ax.axis("off")
    fig.patch.set_facecolor(SUP)
    ax.set_title(titulo, loc="left", fontsize=11.5, color=T1)
    return fig, ax


def ambiente(ax, texto):
    caja(ax, 0.2, 0.6, 2.3, 3.9, "Ambiente", texto, color="#f1efe9")


def reactivo():
    fig, ax = base("Agente reactivo simple (robotito)", 11, 4.6)
    ambiente(ax, "Malla del TestBed\nObstáculos\nstart y goal\n(el agente no los\nconoce)")
    ax.add_patch(FancyBboxPatch((3.1, 0.6), 7.6, 3.9, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc="none", ec=AQ, lw=2, ls="--"))
    ax.text(3.3, 4.3, "Agente (sin memoria, sin modelo, sin meta)", fontsize=9.5, color=T1, fontweight="bold")
    caja(ax, 3.4, 1.6, 2.3, 1.7, "Sensor", "¿chocó?\nchoque / sin choque", color="#d9f0e7")
    caja(ax, 6.0, 1.6, 2.4, 1.7, "Reglas condición-acción",
         "sin choque → derecha\nchoque → dirección\nal azar (4 opciones)", color="#d9f0e7")
    caja(ax, 8.7, 1.6, 1.8, 1.7, "Actuadores", "arriba, abajo,\nizquierda, derecha", color="#d9f0e7")
    flecha(ax, (2.5, 2.45), (3.4, 2.45), "percepción")
    flecha(ax, (5.7, 2.45), (6.0, 2.45))
    flecha(ax, (8.4, 2.45), (8.7, 2.45))
    retorno(ax, 9.6, 1.6, 0.25, 1.35, 0.6, "acción (un movimiento por paso)")
    fig.tight_layout(); fig.savefig("figuras/diagrama_reactivo.png", dpi=150)


def astar():
    fig, ax = base("Agente basado en metas (A*)", 11.5, 4.9)
    ambiente(ax, "Malla del TestBed\nObstáculos\nstart y goal\n(JSON completo)")
    ax.add_patch(FancyBboxPatch((3.0, 0.5), 8.3, 4.2, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc="none", ec=AZ, lw=2, ls="--"))
    ax.text(3.2, 4.5, "Agente basado en metas", fontsize=9.5, color=T1, fontweight="bold")
    caja(ax, 3.2, 2.9, 2.3, 1.4, "Sensores", "lee el JSON:\ngrid, obstáculos,\nstart, goal", color="#dbe8f8")
    caja(ax, 5.8, 2.9, 2.5, 1.4, "Estado del mundo", "grafo de celdas libres\n(vecinos a 4 dir.)", color="#dbe8f8")
    caja(ax, 8.6, 2.9, 2.5, 1.4, "Metas", "llegar a goal\ncon costo mínimo", color="#dbe8f8")
    caja(ax, 5.8, 0.85, 2.5, 1.5, "Decisión", "A*: f = g + h\nh = Manhattan\nempate: f → h → FIFO", color="#dbe8f8")
    caja(ax, 8.6, 0.85, 2.5, 1.5, "Actuadores", "ejecuta la ruta\ncelda por celda", color="#dbe8f8")
    flecha(ax, (2.5, 3.6), (3.2, 3.6))
    flecha(ax, (5.5, 3.6), (5.8, 3.6))
    flecha(ax, (8.3, 3.6), (8.6, 3.6))
    flecha(ax, (7.05, 2.9), (7.05, 2.35))
    flecha(ax, (9.85, 2.9), (9.85, 2.35), "")
    flecha(ax, (8.3, 1.6), (8.6, 1.6))
    retorno(ax, 9.85, 0.85, 0.25, 1.35, 0.6, "acciones (un movimiento por paso)")
    fig.tight_layout(); fig.savefig("figuras/diagrama_astar.png", dpi=150)


def qlearning():
    fig, ax = base("Agente que aprende (Q-Learning)", 12, 5.4)
    ambiente(ax, "Malla N×N\nObstáculos\nMeta\nRecompensas\n(−1, −10, +10)")
    ax.add_patch(FancyBboxPatch((3.0, 0.45), 8.8, 4.55, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc="none", ec=NA, lw=2, ls="--"))
    ax.text(3.2, 4.8, "Agente que aprende", fontsize=9.5, color=T1, fontweight="bold")
    caja(ax, 3.2, 3.15, 2.8, 1.35, "Crítico", "error TD:\nr + γ·max Q(s',·) − Q(s,a)", color="#fbe3d9")
    caja(ax, 8.6, 3.15, 2.9, 1.35, "Estándar de desempeño", "r = −1, −10, +10", color="#fbe3d9")
    caja(ax, 3.2, 1.95, 2.8, 1.0, "Elemento de aprendizaje", "actualiza la tabla Q", color="#fbe3d9")
    caja(ax, 8.6, 1.95, 2.9, 1.0, "Generador de problemas", "explora con ε-greedy", color="#fbe3d9")
    caja(ax, 3.2, 0.7, 8.3, 1.0, "Elemento de desempeño", "elige la acción con la tabla Q (argmax)", color="#fbe3d9")
    flecha(ax, (2.5, 4.0), (3.2, 3.95), "s', r")
    flecha(ax, (8.6, 3.8), (6.0, 3.8))
    flecha(ax, (4.6, 3.15), (4.6, 2.95))
    flecha(ax, (4.6, 1.95), (4.6, 1.7))
    flecha(ax, (8.6, 2.45), (6.0, 2.45), "")
    flecha(ax, (10.0, 1.95), (10.0, 1.7))
    retorno(ax, 11.0, 0.7, 0.2, 1.35, 0.6, "acción a")
    fig.tight_layout(); fig.savefig("figuras/diagrama_qlearning.png", dpi=150)


if __name__ == "__main__":
    reactivo(); astar(); qlearning()
