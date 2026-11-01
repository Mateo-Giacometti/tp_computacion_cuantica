"""Demo 1: un solo qubit que aprende a clasificar (data re-uploading).

Basado en Pérez-Salinas et al., "Data re-uploading for a universal quantum
classifier", Quantum 4, 226 (2020). https://arxiv.org/abs/1907.02085

Tarea: decidir si un punto (x1, x2) del cuadrado [-1, 1]^2 cae dentro de un
círculo. Un clasificador lineal no puede hacerlo; un qubit con los datos
"re-subidos" varias veces sí.

Genera:
  salidas/reuploading_entrenamiento.gif  frontera de decisión época a época
  salidas/reuploading_final.png          resultado final
  salidas/reuploading_circuito.png       el circuito dibujado
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np
from PIL import Image

SALIDAS = Path(__file__).parent / "salidas"
SALIDAS.mkdir(exist_ok=True)

CAPAS = 4          # cuántas veces se vuelven a cargar los datos
EPOCAS = 40
RADIO = np.sqrt(2 / np.pi)  # círculo que ocupa la mitad del cuadrado

np.random.seed(0)


def generar_datos(n):
    x = np.random.uniform(-1, 1, size=(n, 2), requires_grad=False)
    y = np.where(np.sum(x**2, axis=1) < RADIO**2, 1.0, -1.0)
    return x, np.array(y, requires_grad=False)


dev = qml.device("default.qubit", wires=1)


@qml.qnode(dev)
def circuito(params, x1, x2):
    # Cada capa: rotación general en la esfera de Bloch, Rot(a, b, c) = R_Z(c) R_Y(b) R_Z(a),
    # cuyos ángulos dependen del dato (w * x + b).
    # x1 y x2 pueden ser vectores: PennyLane evalúa todos los puntos a la vez.
    for capa in range(CAPAS):
        w, b = params[capa, 0], params[capa, 1]
        qml.Rot(w[0] * x1 + b[0], w[1] * x2 + b[1], b[2], wires=0)
    # Valor medio del observable σz: autovalor +1 = adentro, -1 = afuera
    return qml.expval(qml.PauliZ(0))


def costo(params, x, y):
    pred = circuito(params, x[:, 0], x[:, 1])
    return np.mean((pred - y) ** 2)


def precision(params, x, y):
    pred = np.sign(circuito(params, x[:, 0], x[:, 1]))
    return np.mean(pred == y)


def dibujar_frontera(params, x, y, titulo, archivo):
    malla = np.linspace(-1, 1, 60)
    gx, gy = np.meshgrid(malla, malla)
    z = circuito(params, gx.ravel(), gy.ravel()).reshape(gx.shape)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contourf(gx, gy, z, levels=20, cmap="RdBu", vmin=-1, vmax=1, alpha=0.8)
    ax.contour(gx, gy, z, levels=[0], colors="k", linewidths=2)
    ax.add_patch(plt.Circle((0, 0), RADIO, fill=False, ls="--", color="gray"))
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap="RdBu", edgecolors="k", s=18)
    ax.set_title(titulo)
    ax.set_aspect("equal")
    fig.savefig(archivo, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    x_train, y_train = generar_datos(200)
    x_test, y_test = generar_datos(1000)

    params = np.random.uniform(-np.pi, np.pi, size=(CAPAS, 2, 3), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.1)

    fig, _ = qml.draw_mpl(circuito)(params, 0.5, -0.3)
    fig.savefig(SALIDAS / "reuploading_circuito.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    cuadros = []
    for epoca in range(EPOCAS + 1):
        if epoca % 2 == 0:
            archivo = SALIDAS / f"_cuadro_{epoca:03d}.png"
            acc = precision(params, x_test, y_test)
            dibujar_frontera(params, x_train, y_train,
                             f"Época {epoca} - precisión test {acc:.0%}", archivo)
            cuadros.append(archivo)
        if epoca < EPOCAS:
            params = opt.step(lambda p: costo(p, x_train, y_train), params)

    print(f"Costo final: {costo(params, x_train, y_train):.3f}")
    print(f"Precisión en test: {precision(params, x_test, y_test):.1%}")

    imagenes = [Image.open(c) for c in cuadros]
    imagenes[0].save(SALIDAS / "reuploading_entrenamiento.gif", save_all=True,
                     append_images=imagenes[1:] + [imagenes[-1]] * 5, duration=250, loop=0)
    imagenes[-1].save(SALIDAS / "reuploading_final.png")
    for c in cuadros:
        c.unlink()


if __name__ == "__main__":
    main()
