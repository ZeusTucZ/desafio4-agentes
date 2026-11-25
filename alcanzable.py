"""¿Puede el robotito llegar a la meta con probabilidad > 0? BFS sobre (celda, percepción)."""
from collections import deque
import robotito as R

def alcanzable(ancho, alto, obst, inicio, meta):
    ini = (inicio, "sin_choque")
    vistos = {ini}
    cola = deque([ini])
    while cola:
        pos, per = cola.popleft()
        if pos == meta:
            return True
        accs = ["derecha"] if per == "sin_choque" else R.NOMBRES
        for a in accs:
            dx, dy = R.ACCIONES[a]
            n = (pos[0] + dx, pos[1] + dy)
            if 0 <= n[0] < ancho and 0 <= n[1] < alto and n not in obst:
                s = (n, "sin_choque")
            else:
                s = (pos, "choque")
            if s not in vistos:
                vistos.add(s)
                cola.append(s)
    return False

if __name__ == "__main__":
    for n in ["facil", "medio", "dificil"]:
        a = R.cargar_laberinto(f"laberintos/{n}.json")
        print(n, alcanzable(*a))
