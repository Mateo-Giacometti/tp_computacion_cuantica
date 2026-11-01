"""Demo 3: SVM con kernel cuántico (QSVC) vs. SVM clásica.

Basado en Havlíček et al., "Supervised learning with quantum-enhanced feature
spaces", Nature 567, 209 (2019). https://arxiv.org/abs/1804.11326

Cada punto x se codifica en un estado |φ(x)⟩ con una codificación tipo IQP
(rotaciones Z más interacciones ZZ, como el "ZZ feature map" de IBM). El
kernel es k(x, x') = |⟨φ(x)|φ(x')⟩|². En hardware real se estima midiendo la
probabilidad de volver a |00⟩ tras U(x) y U†(x'); acá, como simulamos, lo
calculamos directamente con los vectores de estado (da el mismo número).

Genera: salidas/kernel_cuantico.png  (matriz de kernel + fronteras de decisión)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

SALIDAS = Path(__file__).parent / "salidas"
SALIDAS.mkdir(exist_ok=True)

N_QUBITS = 2
REPETICIONES = 2

dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev)
def estado(x):
    qml.IQPEmbedding(x, wires=range(N_QUBITS), n_repeats=REPETICIONES)
    return qml.state()


def estados(X):
    return np.array([estado(x) for x in X])


def kernel_cuantico(A, B):
    # k(a, b) = |<φ(a)|φ(b)>|^2 para todos los pares a la vez
    return np.abs(estados(A).conj() @ estados(B).T) ** 2


def main():
    X, y = make_moons(n_samples=160, noise=0.15, random_state=0)
    X = MinMaxScaler((0, np.pi)).fit_transform(X)  # los datos se usan como ángulos
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    orden = np.argsort(y_tr)
    X_tr, y_tr = X_tr[orden], y_tr[orden]  # ordenar por clase para ver bloques en el kernel

    K_tr = kernel_cuantico(X_tr, X_tr)
    qsvc = SVC(kernel="precomputed").fit(K_tr, y_tr)
    acc_q = qsvc.score(kernel_cuantico(X_te, X_tr), y_te)

    svc = SVC(kernel="rbf").fit(X_tr, y_tr)
    acc_c = svc.score(X_te, y_te)
    print(f"Precisión QSVC (kernel cuántico): {acc_q:.1%}")
    print(f"Precisión SVC (kernel RBF clásico): {acc_c:.1%}")

    malla = np.linspace(0, np.pi, 50)
    gx, gy = np.meshgrid(malla, malla)
    G = np.c_[gx.ravel(), gy.ravel()]
    z_q = qsvc.decision_function(kernel_cuantico(G, X_tr)).reshape(gx.shape)
    z_c = svc.decision_function(G).reshape(gx.shape)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    im = axes[0].imshow(K_tr, cmap="viridis")
    axes[0].set_title("Matriz de kernel cuántico\n(ordenada por clase)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    for ax, z, titulo in [(axes[1], z_q, f"QSVC cuántico - test {acc_q:.0%}"),
                          (axes[2], z_c, f"SVC clásico (RBF) - test {acc_c:.0%}")]:
        ax.contourf(gx, gy, z, levels=20, cmap="RdBu", alpha=0.8)
        ax.contour(gx, gy, z, levels=[0], colors="k", linewidths=2)
        ax.scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, cmap="RdBu", edgecolors="k", s=20)
        ax.scatter(X_te[:, 0], X_te[:, 1], c=y_te, cmap="RdBu", edgecolors="k", marker="^", s=30)
        ax.set_title(titulo)
    fig.savefig(SALIDAS / "kernel_cuantico.png", dpi=130, bbox_inches="tight")
    print(f"Gráfico guardado en {SALIDAS / 'kernel_cuantico.png'}")


if __name__ == "__main__":
    main()
