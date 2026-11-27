"""Visualization utilities for the QNN notebook (matching QSVC visual style)."""
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pennylane as qml

CLASS_RED = "#e63946"
CLASS_BLUE = "#1d84e8"
CURVE_BLUE = "#1f5a99"
CURVE_RED = "#c0392b"
CURVE_GRAY = "#5c677d"


def init_environment(figures_dir):
    """Configura el estilo de matplotlib y asegura la carpeta de salida."""
    warnings.filterwarnings("ignore")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.grid": False,
        "font.size": 11,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
    })
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _class_colors(y):
    """Asigna colores de alto contraste a etiquetas en {-1, +1}."""
    return [CLASS_BLUE if val > 0 else CLASS_RED for val in np.asarray(y)]


def _class_legend():
    """Leyenda unificada para clases y splits train/test."""
    return [
        Line2D([0], [0], marker="o", color="w", label="Clase -1 (rojo)",
               markerfacecolor=CLASS_RED, markeredgecolor="white", markeredgewidth=1.0, markersize=8.5),
        Line2D([0], [0], marker="o", color="w", label="Clase +1 (azul)",
               markerfacecolor=CLASS_BLUE, markeredgecolor="white", markeredgewidth=1.0, markersize=8.5),
        Line2D([0], [0], marker="o", color="w", label="Train",
               markerfacecolor="#999999", markeredgecolor="white", markeredgewidth=0.9, markersize=7.5),
        Line2D([0], [0], marker="^", color="w", label="Test",
               markerfacecolor="#999999", markeredgecolor="#333333", markersize=8.0),
    ]


def plot_qnn_circuit(qnode, x_sample, weights, output_path):
    """Dibuja el circuito de la QNN (codificación + ansatz + medición)."""
    fig, ax = qml.draw_mpl(qnode, style="pennylane")(x_sample, weights)
    ax.set_title("Circuito QNN: codificación RY(x), ansatz (RY, RZ, CNOT) y medición ⟨Z⟩", pad=12)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_parameter_shift(thetas, f_vals, theta0, f0, f_plus, f_minus, grad_val, output_path):
    """Muestra f(θ) como sinusoide y la estimación exacta por parameter shift."""
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    ax.plot(thetas, f_vals, color=CURVE_BLUE, lw=2.3, label=r"$f(\theta) = \langle Z \rangle$")

    # Recta tangente en theta0
    dt = np.linspace(-0.75, 0.75, 40)
    ax.plot(theta0 + dt, f0 + grad_val * dt, color=CURVE_RED, lw=2.0, ls="--",
            label=rf"Tangente exacta ($\partial f/\partial \theta = {grad_val:+.3f}$)")

    # Puntos desplazados en +/- pi/2
    ax.scatter([theta0], [f0], color="#111111", s=65, zorder=5)
    ax.scatter([theta0 - np.pi / 2, theta0 + np.pi / 2], [f_minus, f_plus],
               color=CURVE_RED, s=70, zorder=5, label=r"Evaluaciones en $\theta_0 \pm \pi/2$")
    ymin_ref = min(f_vals) - 0.12
    ax.vlines([theta0 - np.pi / 2, theta0 + np.pi / 2], ymin=ymin_ref,
              ymax=[f_minus, f_plus], colors=CURVE_RED, linestyles=":", lw=1.3, alpha=0.8)

    ax.annotate(r"$\theta_0 - \frac{\pi}{2}$", xy=(theta0 - np.pi / 2, f_minus),
                xytext=(theta0 - np.pi / 2, f_minus - 0.22), ha="center", fontsize=10,
                color=CURVE_RED, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.annotate(r"$\theta_0 + \frac{\pi}{2}$", xy=(theta0 + np.pi / 2, f_plus),
                xytext=(theta0 + np.pi / 2, f_plus + 0.15), ha="center", fontsize=10,
                color=CURVE_RED, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.annotate(r"$\theta_0$", xy=(theta0, f0), xytext=(theta0 + 0.22, f0 - 0.14),
                ha="left", fontsize=10.5, fontweight="bold")

    ax.set_xlabel(r"Ángulo entrenable $\theta$ (rad)")
    ax.set_ylabel(r"Salida cuántica $f(\theta) = \langle Z \rangle$")
    ax.set_title(r"Regla de parameter shift: $\partial f/\partial \theta = [f(\theta+\pi/2) - f(\theta-\pi/2)]/2$")
    ax.set_xlim(thetas[0], thetas[-1])
    ax.set_ylim(-1.12, 1.12)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=9.5, frameon=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_and_boundaries(
    losses_ent, losses_no_ent, X_train, X_test, y_train, y_test,
    predict_fns, titles, output_path
):
    """Grafica la evolución de la pérdida y las fronteras de decisión en XOR 2D."""
    fig, axes = plt.subplots(1, 4, figsize=(17.2, 4.3))

    # Panel 1: curvas de pérdida durante el entrenamiento
    ax0 = axes[0]
    epochs = np.arange(len(losses_ent))
    ax0.plot(epochs, losses_ent, color=CURVE_BLUE, lw=2.3, label="QNN con CNOT")
    ax0.plot(epochs, losses_no_ent, color=CURVE_RED, lw=2.0, ls="--", label="QNN sin CNOT")
    ax0.set_xlabel("Época (Adam)")
    ax0.set_ylabel("Pérdida MSE")
    ax0.set_title("1. Evolución del entrenamiento")
    ax0.set_ylim(0, max(max(losses_ent), max(losses_no_ent)) * 1.1)
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="upper right", fontsize=9.5, frameon=True)

    # Paneles 2-4: fronteras de decisión en [-pi/2, pi/2]^2
    axis = np.linspace(-np.pi / 2, np.pi / 2, 45)
    gx, gy = np.meshgrid(axis, axis)
    grid = np.column_stack([gx.ravel(), gy.ravel()])

    for ax, pred_fn, title in zip(axes[1:], predict_fns, titles):
        zz = np.asarray(pred_fn(grid)).reshape(gx.shape)
        ax.contourf(gx, gy, zz, levels=np.linspace(-1.0, 1.0, 21), cmap="RdBu", alpha=0.52, extend="both")
        ax.contour(gx, gy, zz, levels=[0.0], colors="#111111", linewidths=2.0)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=_class_colors(y_train),
                   edgecolors="white", linewidths=0.7, s=34, zorder=3)
        ax.scatter(X_test[:, 0], X_test[:, 1], c=_class_colors(y_test),
                   marker="^", edgecolors="#1a1a1a", linewidths=0.7, s=46, zorder=4)
        ax.set_xlim(-np.pi / 2, np.pi / 2)
        ax.set_ylim(-np.pi / 2, np.pi / 2)
        ax.set_xticks([-np.pi / 2, 0, np.pi / 2])
        ax.set_xticklabels([r"$-\pi/2$", "0", r"$+\pi/2$"])
        ax.set_yticks([-np.pi / 2, 0, np.pi / 2])
        ax.set_yticklabels([r"$-\pi/2$", "0", r"$+\pi/2$"])
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_title(title)

    fig.legend(handles=_class_legend(), loc="lower center", ncol=4,
               bbox_to_anchor=(0.62, -0.06), frameon=True, fontsize=9.5)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_barren_plateaus(qubit_counts, emp_vars, exact_vars, shot_noise_var, shots, output_path):
    """Grafica la caída exponencial de Var[∂f/∂θ₁] frente al ruido de shots."""
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.semilogy(qubit_counts, emp_vars, "o-", color=CURVE_BLUE, lw=2.3, ms=7.5,
                label=r"Varianza empírica $\operatorname{Var}[\partial f/\partial \theta_1]$ (circuito de paridad)")
    ax.semilogy(qubit_counts, exact_vars, "s--", color=CURVE_GRAY, lw=1.6, ms=5.5,
                label=r"Predicción analítica $4^{-n}$")
    ax.axhline(shot_noise_var, color=CURVE_RED, ls=":", lw=2.1,
               label=rf"Piso de ruido estadístico ($S = {shots}$ shots, $\approx 1/(2S)$)")

    ax.fill_between(qubit_counts, 1e-8, shot_noise_var, color=CURVE_RED, alpha=0.07)
    ax.text(qubit_counts[-1] - 0.15, shot_noise_var * 0.42,
            "Zona dominada por ruido de shots\n(el gradiente queda oculto)",
            ha="right", va="top", fontsize=9.5, color=CURVE_RED)

    ax.set_xticks(qubit_counts)
    ax.set_xlabel("Número de qubits $n$")
    ax.set_ylabel(r"Varianza del gradiente $\operatorname{Var}[\partial f/\partial \theta_1]$ (escala log)")
    ax.set_title("Barren plateaus: el gradiente decae exponencialmente con $n$")
    ax.set_ylim(min(min(emp_vars), min(exact_vars)) * 0.35, 0.3)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9.5, frameon=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
