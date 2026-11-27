"""Visualization utilities for the QCNN notebook (matching QSVC/QNN visual style)."""
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
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


def plot_qcnn_circuit(qnode, x_sample, conv_p, pool_p, output_path):
    """Dibuja el circuito completo 8 -> 4 -> 2 -> 1 de la QCNN."""
    fig, ax = qml.draw_mpl(qnode, style="pennylane", fontsize="small")(x_sample, conv_p, pool_p)
    fig.set_size_inches(16.0, 4.8)
    ax.set_title("Arquitectura QCNN (8 → 4 → 2 → 1 qubits): convolución compartida, pooling y medición ⟨Z₇⟩", pad=12)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.show()


def plot_ising_phase_transition(
    losses, hs_train, y_train, hs_grid, preds_grid, crossing_h, output_path
):
    """Grafica el entrenamiento y la detección de transición de fase en el TFIM."""
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.5), gridspec_kw={"width_ratios": [1.0, 1.55]})

    # Panel 1: curva de pérdida
    ax0 = axes[0]
    ax0.plot(np.arange(len(losses)), losses, color=CURVE_BLUE, lw=2.3)
    ax0.set_xlabel("Época (Adam)")
    ax0.set_ylabel("Pérdida MSE")
    ax0.set_title("1. Entrenamiento con 12 estados cuánticos")
    ax0.set_ylim(0, max(losses) * 1.1)
    ax0.grid(True, alpha=0.25)

    # Panel 2: salida de la QCNN vs campo transversal h
    ax1 = axes[1]
    ax1.axvspan(hs_grid[0], 1.0, color=CLASS_BLUE, alpha=0.07, label="Fase ordenada teórica ($h < 1$)")
    ax1.axvspan(1.0, hs_grid[-1], color=CLASS_RED, alpha=0.07, label="Fase desordenada teórica ($h > 1$)")
    ax1.axvspan(0.70, 1.30, color="#f4a261", alpha=0.14, label="Zona crítica no vista en train")

    ax1.axhline(0.0, color="#444444", lw=1.1, ls="-")
    ax1.axvline(1.0, color="#111111", lw=1.6, ls="--", label="Transición teórica ($h_c = 1$)")
    ax1.plot(hs_grid, preds_grid, color=CURVE_BLUE, lw=2.6, label=r"Salida QCNN $\langle Z_7 \rangle(h)$")

    tr_colors = [CLASS_BLUE if y > 0 else CLASS_RED for y in y_train]
    ax1.scatter(hs_train, y_train, c=tr_colors, edgecolors="#111111", linewidths=0.9, s=55, zorder=5,
                label="Estados de train (etiquetas ±1)")

    if crossing_h is not None:
        ax1.scatter([crossing_h], [0.0], color="#e76f51", edgecolors="#111111", s=85, zorder=6)
        ax1.annotate(
            rf"Cruce por 0 en $h \approx {crossing_h:.2f}$",
            xy=(crossing_h, 0.0),
            xytext=(crossing_h + 0.14, 0.34),
            fontsize=10,
            fontweight="bold",
            color="#111111",
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.92),
        )

    ax1.set_xlim(hs_grid[0], hs_grid[-1])
    ax1.set_ylim(-1.15, 1.15)
    ax1.set_xlabel("Campo magnético transversal $h$")
    ax1.set_ylabel(r"Predicción de la QCNN $\langle Z_7 \rangle$")
    ax1.set_title("2. Reconocimiento de fases en la cadena de Ising (8 espines)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="lower left", fontsize=8.1, frameon=True, ncol=1, framealpha=0.92)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_digits_results(
    losses, img_test, y_test, preds_test, acc_qcnn, acc_logreg, output_path
):
    """Muestra ejemplos de dígitos 0 vs 1, pérdida y distribución de predicciones."""
    fig = plt.figure(figsize=(15.4, 4.7))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.24)
    left = outer[0].subgridspec(1, 2, width_ratios=[0.95, 1.15], wspace=0.36)
    right = outer[1].subgridspec(2, 4, hspace=0.42, wspace=0.18)

    # Panel 1: curva de pérdida
    ax_loss = fig.add_subplot(left[0, 0])
    ax_loss.plot(np.arange(len(losses)), losses, color=CURVE_BLUE, lw=2.3)
    ax_loss.set_xlabel("Época (Adam)")
    ax_loss.set_ylabel("Pérdida MSE")
    ax_loss.set_title("1. Entrenamiento (60 dígitos)")
    ax_loss.set_ylim(0, max(losses) * 1.1)
    ax_loss.grid(True, alpha=0.25)

    # Panel 2: distribución de salida <Z7> en test
    ax_hist = fig.add_subplot(left[0, 1])
    bins = np.linspace(-1.0, 1.0, 21)
    ax_hist.hist(preds_test[y_test > 0], bins=bins, color=CLASS_BLUE, alpha=0.68, edgecolor="white", label="Dígito 0 (y = +1)")
    ax_hist.hist(preds_test[y_test < 0], bins=bins, color=CLASS_RED, alpha=0.68, edgecolor="white", label="Dígito 1 (y = -1)")
    ax_hist.axvline(0.0, color="#111111", ls="--", lw=1.5, label="Umbral (0)")
    ax_hist.set_xlabel(r"Salida QCNN $\langle Z_7 \rangle$ en test")
    ax_hist.set_ylabel("Cantidad de dígitos")
    ax_hist.set_title(f"2. Test: QCNN {acc_qcnn:.1%} vs. LogReg {acc_logreg:.0%}")
    ax_hist.grid(True, alpha=0.22)
    ax_hist.legend(loc="upper left", fontsize=8.4, frameon=True)

    # Panel 3: 8 ejemplos de dígitos de test (4 ceros y 4 unos)
    idx_zeros = np.where(y_test > 0)[0][:4]
    idx_ones = np.where(y_test < 0)[0][:4]
    sample_indices = list(idx_zeros) + list(idx_ones)

    for pos, idx in enumerate(sample_indices):
        row, col = pos // 4, pos % 4
        ax = fig.add_subplot(right[row, col])
        ax.imshow(img_test[idx], cmap="gray_r", interpolation="nearest")
        true_digit = 0 if y_test[idx] > 0 else 1
        pred_digit = 0 if preds_test[idx] >= 0 else 1
        ok = true_digit == pred_digit
        title_color = "#1b7f3a" if ok else CURVE_RED
        ax.set_title(f"Real:{true_digit} | Pred:{pred_digit}\n⟨Z⟩={preds_test[idx]:+.2f}",
                     fontsize=8.6, color=title_color, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_barren_plateau_comparison(
    qcnn_ns, qcnn_vars, qnn_ns, qnn_vars, shot_noise_var, shots, output_path
):
    """Compara Var[∂f/∂θ] de la QCNN frente a una QNN genérica (Pesah et al., 2021)."""
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    ax.semilogy(qcnn_ns, qcnn_vars, "o-", color=CURVE_BLUE, lw=2.5, ms=8.0,
                label=r"QCNN ($n \to n/2 \to \dots \to 1$): decaimiento polinomial")
    ax.semilogy(qnn_ns, qnn_vars, "s--", color=CURVE_RED, lw=2.2, ms=7.0,
                label=r"QNN genérica de paridad: decaimiento exponencial ($\sim 4^{-n}$)")
    ax.axhline(shot_noise_var, color=CURVE_GRAY, ls=":", lw=2.0,
               label=rf"Piso de ruido estadístico ($S = {shots}$ shots, $\approx 1/(2S)$)")

    ax.fill_between(qnn_ns, 1e-8, shot_noise_var, color=CURVE_RED, alpha=0.06)
    ax.annotate(
        f"En n = 8 qubits:\nQCNN es ~{qcnn_vars[-1] / qnn_vars[qnn_ns.index(8)]:,.0f}× mayor".replace(",", "."),
        xy=(8, qcnn_vars[-1]),
        xytext=(5.8, 2.2e-3),
        fontsize=9.8,
        fontweight="bold",
        color=CURVE_BLUE,
        arrowprops=dict(arrowstyle="->", color=CURVE_BLUE, lw=1.4),
        bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=CURVE_BLUE, alpha=0.95),
    )

    ax.set_xticks(qnn_ns)
    ax.set_xlabel("Número de qubits $n$")
    ax.set_ylabel(r"Varianza del gradiente $\operatorname{Var}[\partial f/\partial \theta]$ (escala log)")
    ax.set_title("Ausencia de barren plateaus en QCNN frente a una QNN genérica")
    ax.set_ylim(min(qnn_vars) * 0.35, 0.5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower left", fontsize=9.2, frameon=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
