"""Demo 2: mesetas estériles (barren plateaus) medidas por nosotros.

Reproduce en pequeño el resultado de McClean et al., "Barren plateaus in
quantum neural network training landscapes", Nat. Commun. 9, 4812 (2018).
https://arxiv.org/abs/1803.11173

Para cada cantidad de qubits n se generan muchos circuitos con ángulos al azar
y se calcula la derivada del costo respecto de un parámetro con la regla de
desplazamiento de parámetros. Si la varianza de esa derivada cae
exponencialmente con n, el paisaje es una "meseta": casi plano en todos lados.

Se comparan dos casos:
  - circuito profundo + costo global (σz en todos los qubits): meseta estéril
  - circuito corto + costo local (σz en un qubit): la varianza no se apaga

Genera: salidas/mesetas_esteriles.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml

SALIDAS = Path(__file__).parent / "salidas"
SALIDAS.mkdir(exist_ok=True)

QUBITS = range(2, 11, 2)
MUESTRAS = 150
rng = np.random.default_rng(1)


def crear_circuito(n, capas, local):
    dev = qml.device("default.qubit", wires=n)
    observable = qml.PauliZ(0) if local else qml.prod(*[qml.PauliZ(i) for i in range(n)])

    @qml.qnode(dev)
    def circuito(angulos, ejes):
        for i in range(n):
            qml.RY(np.pi / 4, wires=i)
        for c in range(capas):
            for i in range(n):
                # Rotación en un eje elegido al azar (X, Y o Z), como en McClean et al.
                [qml.RX, qml.RY, qml.RZ][ejes[c, i]](angulos[c, i], wires=i)
            for i in range(n - 1):
                # CZ = compuerta de fase controlada CPH(π): entrelaza qubits vecinos
                qml.CZ(wires=[i, i + 1])
        return qml.expval(observable)

    return circuito


def varianza_gradiente(n, capas, local):
    circuito = crear_circuito(n, capas, local)
    derivadas = []
    for _ in range(MUESTRAS):
        angulos = rng.uniform(0, 2 * np.pi, size=(capas, n))
        ejes = rng.integers(0, 3, size=(capas, n))
        # Parameter-shift sobre el primer ángulo: exacto con dos evaluaciones.
        mas, menos = angulos.copy(), angulos.copy()
        mas[0, 0] += np.pi / 2
        menos[0, 0] -= np.pi / 2
        derivadas.append(0.5 * (circuito(mas, ejes) - circuito(menos, ejes)))
    return np.var(derivadas)


def main():
    casos = {
        "Profundo (40 capas), costo global": dict(capas=40, local=False),
        "Corto (2 capas), costo local": dict(capas=2, local=True),
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    for nombre, cfg in casos.items():
        varianzas = []
        for n in QUBITS:
            v = varianza_gradiente(n, **cfg)
            varianzas.append(v)
            print(f"{nombre:38s} n={n:2d}  Var[dC/dθ] = {v:.2e}")
        ax.semilogy(list(QUBITS), varianzas, "o-", label=nombre)

    ax.set_xlabel("Número de qubits")
    ax.set_ylabel("Varianza del gradiente (escala log)")
    ax.set_title("Mesetas estériles: el gradiente se apaga con más qubits")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(SALIDAS / "mesetas_esteriles.png", dpi=150, bbox_inches="tight")
    print(f"Gráfico guardado en {SALIDAS / 'mesetas_esteriles.png'}")


if __name__ == "__main__":
    main()
