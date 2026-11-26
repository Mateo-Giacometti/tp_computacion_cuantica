"""Visualization and GIF export utilities for the QSVC notebook."""
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from PIL import Image
from sklearn.metrics.pairwise import rbf_kernel

CLASS_RED = "#e63946"
CLASS_BLUE = "#1d84e8"
SV_RING_COLOR = "#000000"
CURVE_BLUE = "#1f5a99"
CURVE_RED = "#c0392b"


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
    """Asigna colores de alto contraste a etiquetas binarias ({0, 1} o {-1, 1})."""
    return [CLASS_BLUE if val > 0 else CLASS_RED for val in np.asarray(y)]


def _class_legend(include_sv=False):
    """Construye la leyenda de clases, splits y support vectors."""
    handles = [
        Line2D([0], [0], marker="o", color="w", label="Clase -1 / 0 (rojo)",
               markerfacecolor=CLASS_RED, markeredgecolor="white", markeredgewidth=1.0, markersize=8.5),
        Line2D([0], [0], marker="o", color="w", label="Clase +1 / 1 (azul)",
               markerfacecolor=CLASS_BLUE, markeredgecolor="white", markeredgewidth=1.0, markersize=8.5),
        Line2D([0], [0], marker="o", color="w", label="Train",
               markerfacecolor="#999999", markeredgecolor="white", markeredgewidth=0.9, markersize=7.5),
        Line2D([0], [0], marker="^", color="w", label="Test",
               markerfacecolor="#999999", markeredgecolor="#333333", markersize=8.0),
    ]
    if include_sv:
        handles.append(
            Line2D([0], [0], marker="o", color="w", label="Support vectors (αᵢ > 0)",
                   markerfacecolor="none", markeredgecolor=SV_RING_COLOR, markeredgewidth=2.3, markersize=11)
        )
    return handles


def plot_1d_dataset(X_train, y_train):
    """Grafica el dataset 1D inicial para introducir el problema visualmente."""
    fig, ax = plt.subplots(figsize=(8.2, 2.7))
    ax.axhline(0, color="#666666", lw=1.5, zorder=1)
    ax.axvspan(-np.pi / 2, np.pi / 2, color=CLASS_BLUE, alpha=0.10, label="Región central (clase +1)")
    ax.axvspan(-np.pi, -np.pi / 2, color=CLASS_RED, alpha=0.09, label="Extremos (clase -1)")
    ax.axvspan(np.pi / 2, np.pi, color=CLASS_RED, alpha=0.09)
    ax.scatter(X_train[:, 0], np.zeros_like(X_train[:, 0]), c=_class_colors(y_train),
               edgecolors="#1a1a1a", linewidths=0.9, s=85, zorder=3)
    ax.axvline(-np.pi / 2, color="#444444", ls="--", lw=1.2)
    ax.axvline(np.pi / 2, color="#444444", ls="--", lw=1.2)
    ax.set_xlim(-np.pi - 0.25, np.pi + 0.25)
    ax.set_ylim(-0.45, 0.72)
    ax.set_yticks([])
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$+\pi/2$", r"$+\pi$"])
    ax.set_xlabel("Feature original $x$")
    ax.set_title("Dataset 1D no separable mediante un único corte lineal")
    ax.legend(loc="upper center", ncol=2, fontsize=9.5, frameon=True)
    plt.tight_layout()
    plt.show()


def plot_bloch_mapping(X_train, y_train, K_train, acc_linear, acc_quantum, output_path):
    """Grafica los datos 1D, su proyección en la esfera de Bloch y la matriz de kernel."""
    fig = plt.figure(figsize=(16, 5.0))
    train_colors = _class_colors(y_train)

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.axhline(0, color="#666666", lw=1.5, zorder=1)
    ax1.axvspan(-np.pi / 2, np.pi / 2, color=CLASS_BLUE, alpha=0.10, label="Clase +1 (|x| < π/2)")
    ax1.axvspan(-np.pi, -np.pi / 2, color=CLASS_RED, alpha=0.09, label="Clase -1 (|x| ≥ π/2)")
    ax1.axvspan(np.pi / 2, np.pi, color=CLASS_RED, alpha=0.09)
    ax1.scatter(X_train[:, 0], np.zeros_like(X_train[:, 0]), c=train_colors, edgecolors="#1a1a1a", s=75, zorder=3)
    ax1.axvline(-np.pi / 2, color="#444444", ls="--", lw=1.2)
    ax1.axvline(np.pi / 2, color="#444444", ls="--", lw=1.2)
    ax1.set_xlim(-np.pi - 0.25, np.pi + 0.25)
    ax1.set_ylim(-0.6, 0.85)
    ax1.set_yticks([])
    ax1.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax1.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$+\pi/2$", r"$+\pi$"])
    ax1.set_xlabel("Feature original $x$")
    ax1.set_title(f"1. Espacio original 1D\nSVM lineal en test ({acc_linear:.0%})")
    ax1.text(0, 0.48, "No separable con un único corte", ha="center",
             bbox=dict(boxstyle="round,pad=0.35", fc="#fff8dc", ec="#d4a017"))
    ax1.legend(loc="lower center", fontsize=9, frameon=True)

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 25)
    ax2.plot_wireframe(
        np.outer(np.cos(u), np.sin(v)),
        np.outer(np.sin(u), np.sin(v)),
        np.outer(np.ones_like(u), np.cos(v)),
        color="#cccccc", alpha=0.35, linewidth=0.6,
    )
    r_disc = np.linspace(0, 1.15, 10)
    t_disc = np.linspace(0, 2 * np.pi, 50)
    R_d, T_d = np.meshgrid(r_disc, t_disc)
    ax2.plot_surface(
        R_d * np.cos(T_d), R_d * np.sin(T_d), np.zeros_like(R_d),
        color="#2ecc71", alpha=0.22, edgecolor="none",
    )
    ax2.plot(np.cos(t_disc), np.sin(t_disc), np.zeros_like(t_disc), color="#1e8449", lw=1.8)

    bx = np.sin(X_train[:, 0])
    by = np.zeros_like(bx)
    bz = np.cos(X_train[:, 0])
    ax2.scatter(bx, by, bz, c=train_colors, edgecolors="#1a1a1a", s=72, depthshade=False, zorder=5)
    ax2.text(0, 0, 1.28, r"$|0\rangle$ (z = +1)", ha="center", color=CLASS_BLUE, fontweight="bold", fontsize=9.5)
    ax2.text(0, 0, -1.35, r"$|1\rangle$ (z = -1)", ha="center", color=CLASS_RED, fontweight="bold", fontsize=9.5)
    ax2.set_box_aspect([1, 1, 1])
    ax2.set_xlim(-1.2, 1.2); ax2.set_ylim(-1.2, 1.2); ax2.set_zlim(-1.2, 1.2)
    ax2.set_axis_off()
    ax2.view_init(elev=18, azim=65)
    ax2.set_title(f"2. Feature map $R_Y(x)|0\\rangle$ en esfera de Bloch\nPlano $z = 0$ separa en test ({acc_quantum:.0%})")

    ax3 = fig.add_subplot(1, 3, 3)
    im = ax3.imshow(K_train, cmap="viridis", vmin=0, vmax=1)
    split_idx = np.sum(y_train == -1) - 0.5
    ax3.axhline(split_idx, color="white", ls="--", lw=1.5)
    ax3.axvline(split_idx, color="white", ls="--", lw=1.5)
    ax3.set_title(r"3. Matriz de kernel $k(x,x') = \cos^2\!\left(\frac{x-x'}{2}\right)$" + "\n(ordenada por clase)")
    ax3.set_xlabel("Dato $j$")
    ax3.set_ylabel("Dato $i$")
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label="Similitud")

    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.show()


def plot_compute_uncompute_circuit(x1, x2, output_path):
    """Dibuja el circuito compute uncompute descompuesto para un ZZ Feature Map."""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def draw_circuit(a, b):
        for i in range(2):
            qml.Hadamard(wires=i)
            qml.RZ(a[i], wires=i)
        qml.CNOT(wires=[0, 1])
        qml.RZ(a[0] * a[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.Barrier(wires=[0, 1])
        qml.CNOT(wires=[0, 1])
        qml.RZ(-b[0] * b[1], wires=1)
        qml.CNOT(wires=[0, 1])
        for i in range(2):
            qml.RZ(-b[i], wires=i)
            qml.Hadamard(wires=i)
        return qml.probs(wires=[0, 1])

    fig, ax = qml.draw_mpl(draw_circuit, style="pennylane")(x1, x2)
    ax.set_title(
        r"Circuito compute uncompute $U^\dagger(x') U(x)$ con medición de $P(|00\rangle) = k(x, x')$",
        fontsize=12, fontweight="bold", pad=12,
    )
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.show()


def plot_moons_comparison(
    X_train, X_test, y_train, y_test, models, state_kernel_fn, angle_state_fn, iqp_state_fn, output_path
):
    """Grafica fronteras de decisión, support vectors y matrices de kernel de 4 modelos."""
    clf_lin, clf_angle, clf_iqp, clf_rbf = models["clfs"]
    K_angle_tr, K_iqp_tr = models["K_angle_train"], models["K_iqp_train"]
    accs = models["accs"]

    axis = np.linspace(-0.1, np.pi + 0.1, 55)
    gx, gy = np.meshgrid(axis, axis)
    grid = np.c_[gx.ravel(), gy.ravel()]

    Z_lin = clf_lin.decision_function(grid).reshape(gx.shape)
    Z_angle = clf_angle.decision_function(state_kernel_fn(angle_state_fn, grid, X_train)).reshape(gx.shape)
    Z_iqp = clf_iqp.decision_function(state_kernel_fn(iqp_state_fn, grid, X_train)).reshape(gx.shape)
    Z_rbf = clf_rbf.decision_function(grid).reshape(gx.shape)

    X_norm = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
    K_lin_vis = X_norm @ X_norm.T
    K_rbf_vis = rbf_kernel(X_train, X_train, gamma=clf_rbf._gamma)

    panels = [
        ("SVM lineal\n(clásica)", accs[0], Z_lin, clf_lin.support_, K_lin_vis),
        ("QSVC Angle Embedding\n(sin entrelazamiento)", accs[1], Z_angle, clf_angle.support_, K_angle_tr),
        ("QSVC ZZ / IQP\n(con entrelazamiento)", accs[2], Z_iqp, clf_iqp.support_, K_iqp_tr),
        ("SVM RBF\n(clásica estándar)", accs[3], Z_rbf, clf_rbf.support_, K_rbf_vis),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8.8))
    split_idx = np.sum(y_train == 0) - 0.5
    col_tr = _class_colors(y_train)
    col_te = _class_colors(y_test)

    for col, (title, acc, Z, sv_idx, K_mat) in enumerate(panels):
        ax = axes[0, col]
        vmax = np.max(np.abs(Z)) * 1.25
        ax.contourf(gx, gy, Z, levels=18, cmap="RdBu", vmin=-vmax, vmax=vmax, alpha=0.35)
        ax.contour(gx, gy, Z, levels=[0], colors="#222222", linewidths=1.8, linestyles="--")
        # Puntos con borde blanco para separar del aro negro de los support vectors
        ax.scatter(X_train[:, 0], X_train[:, 1], c=col_tr, edgecolors="white", linewidths=0.9, s=46, zorder=3)
        ax.scatter(X_test[:, 0], X_test[:, 1], c=col_te, edgecolors="#333333", linewidths=0.8, marker="^", s=58, zorder=4)
        ax.scatter(X_train[sv_idx, 0], X_train[sv_idx, 1], facecolors="none", edgecolors=SV_RING_COLOR,
                   linewidths=2.2, s=145, zorder=5)
        ax.set_title(f"{title}\nTest {acc:.1%} ({len(sv_idx)} SV)")
        ax.set_xlabel("$x_1$")
        if col == 0:
            ax.set_ylabel("$x_2$")

        axk = axes[1, col]
        axk.imshow(K_mat, cmap="viridis", vmin=0, vmax=1)
        axk.axhline(split_idx, color="white", ls="--", lw=1.2, alpha=0.85)
        axk.axvline(split_idx, color="white", ls="--", lw=1.2, alpha=0.85)
        axk.set_title("Matriz de kernel $K$", fontsize=10.5)
        axk.set_xlabel("Dato $j$")
        if col == 0:
            axk.set_ylabel("Dato $i$")

    fig.legend(handles=_class_legend(include_sv=True), loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.02), frameon=True, fontsize=10)
    fig.suptitle("Comparativa en make_moons (fronteras de decisión y matrices de kernel)",
                 fontsize=13.5, fontweight="bold", y=1.06)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.show()


def plot_havlicek_comparison(
    X_train, X_test, y_train, y_test, grid_bundle, qsvc, rbf, acc_q, acc_rbf, state_kernel_fn, zz_state_fn, output_path
):
    """Compara paisaje real, QSVC y SVM RBF sobre el dataset sintético de Havlíček et al. (2019)."""
    gx, gy, grid_points, landscape = grid_bundle
    Z_q = qsvc.decision_function(state_kernel_fn(zz_state_fn, grid_points, X_train)).reshape(gx.shape)
    Z_rbf = rbf.decision_function(grid_points).reshape(gx.shape)

    col_tr = _class_colors(y_train)
    col_te = _class_colors(y_test)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.1))
    panels = [
        ("Estructura real del dataset sintético\n(Havlíček et al., 2019)", landscape),
        (f"QSVC con ZZ Feature Map\nPrecisión en test {acc_q:.1%}", Z_q),
        (f"SVM clásica con kernel RBF\nPrecisión en test {acc_rbf:.1%}", Z_rbf),
    ]
    for ax, (title, Z) in zip(axes, panels):
        vmax = np.max(np.abs(Z)) * 1.2
        ax.contourf(gx, gy, Z, levels=18, cmap="RdBu", vmin=-vmax, vmax=vmax, alpha=0.40)
        ax.contour(gx, gy, Z, levels=[0], colors="#111111", linewidths=1.8)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=col_tr, edgecolors="white", linewidths=0.9, s=48, zorder=3)
        ax.scatter(X_test[:, 0], X_test[:, 1], c=col_te, edgecolors="#222222", linewidths=0.8, marker="^", s=60, zorder=4)
        ax.set_title(title)
        ax.set_xlabel(r"$x_1 \in [0, 2\pi]$")
        ax.set_xlim(0, 2 * np.pi); ax.set_ylim(0, 2 * np.pi)

    axes[0].set_ylabel(r"$x_2 \in [0, 2\pi]$")
    fig.legend(handles=_class_legend(include_sv=False), loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.03), frameon=True, fontsize=10)
    fig.suptitle("Inductive bias en el dataset sintético de Havlíček et al. (2019)", fontsize=13.5, fontweight="bold", y=1.09)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.show()


def generate_kta_gif(
    X_train, X_test, y_train, y_test, snapshots, hist_kta, hist_acc, saved_steps, total_steps,
    qek_kernel_fn, figures_dir
):
    """Genera el GIF animado y la comparativa inicial/final de Quantum Kernel Alignment."""
    out_dir = Path(figures_dir)
    axis = np.linspace(-1.15, 1.15, 45)
    gdx, gdy = np.meshgrid(axis, axis)
    grid = pnp.array(np.c_[gdx.ravel(), gdy.ravel()], requires_grad=False)
    X_tr_p = pnp.array(X_train, requires_grad=False)
    split_idx = np.sum(y_train == -1) - 0.5
    col_tr = _class_colors(y_train)
    col_te = _class_colors(y_test)

    frames = []
    for step in saved_steps:
        p_snap, K_snap, clf_snap, kta_val, acc_val = snapshots[step]
        Z = clf_snap.decision_function(np.array(qek_kernel_fn(grid, X_tr_p, p_snap))).reshape(gdx.shape)

        fig = plt.figure(figsize=(16.8, 5.1))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.08, 1.15], hspace=0.20, wspace=0.40)
        ax_bound = fig.add_subplot(gs[:, 0])
        ax_mat = fig.add_subplot(gs[:, 1])
        ax_acc = fig.add_subplot(gs[0, 2])
        ax_kta = fig.add_subplot(gs[1, 2], sharex=ax_acc)

        vmax = max(np.max(np.abs(Z)), 0.5) * 1.2
        ax_bound.contourf(gdx, gdy, Z, levels=18, cmap="RdBu", vmin=-vmax, vmax=vmax, alpha=0.40)
        ax_bound.contour(gdx, gdy, Z, levels=[0], colors="#111111", linewidths=2.0)
        ax_bound.scatter(X_train[:, 0], X_train[:, 1], c=col_tr, edgecolors="white", linewidths=0.9, s=46, zorder=3)
        ax_bound.scatter(X_test[:, 0], X_test[:, 1], c=col_te, edgecolors="#222222", linewidths=0.8, marker="^", s=58, zorder=4)
        ax_bound.set_aspect("equal")
        ax_bound.set_title(f"Frontera QSVC (paso {step})\nPrecisión en test {acc_val:.1%}")
        ax_bound.set_xlabel("$x_1$"); ax_bound.set_ylabel("$x_2$")

        im = ax_mat.imshow(K_snap, cmap="viridis", vmin=0, vmax=1)
        ax_mat.axhline(split_idx, color="white", ls="--", lw=1.3)
        ax_mat.axvline(split_idx, color="white", ls="--", lw=1.3)
        ax_mat.set_title(f"Matriz de kernel $K(\\theta)$\nKTA = {kta_val:.3f}")
        ax_mat.set_xlabel("Dato $j$"); ax_mat.set_ylabel("Dato $i$")
        fig.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04)

        steps_x = np.arange(step + 1)
        ax_acc.plot(steps_x, np.array(hist_acc[:step + 1]) * 100, color=CURVE_BLUE, lw=2.3)
        ax_acc.scatter([step], [acc_val * 100], color=CURVE_BLUE, s=50, zorder=5)
        ax_acc.axvline(step, color="#888888", ls=":", lw=1.1, alpha=0.75)
        ax_acc.set_xlim(0, total_steps)
        ax_acc.set_ylim(55, 106)
        ax_acc.set_ylabel("Precisión (%)", color=CURVE_BLUE, fontweight="bold")
        ax_acc.set_title("Evolución sincronizada por iteración")
        ax_acc.grid(True, alpha=0.3)
        plt.setp(ax_acc.get_xticklabels(), visible=False)

        ax_kta.plot(steps_x, hist_kta[:step + 1], color=CURVE_RED, lw=2.3)
        ax_kta.scatter([step], [kta_val], color=CURVE_RED, s=50, zorder=5)
        ax_kta.axvline(step, color="#888888", ls=":", lw=1.1, alpha=0.75)
        ax_kta.set_ylim(0.0, 0.55)
        ax_kta.set_xlabel("Iteración")
        ax_kta.set_ylabel("KTA", color=CURVE_RED, fontweight="bold")
        ax_kta.grid(True, alpha=0.3)

        fig.suptitle("Quantum Kernel Alignment sobre el dataset DoubleCake", fontsize=13.5, fontweight="bold", y=1.05)
        tmp_path = out_dir / f"_tmp_kta_{step:03d}.png"
        fig.savefig(tmp_path, dpi=135, bbox_inches="tight")
        plt.close(fig)
        frames.append(tmp_path)

    imgs = [Image.open(f) for f in frames]
    imgs[0].save(
        out_dir / "05_kernel_alignment_doublecake.gif",
        save_all=True,
        append_images=imgs[1:] + [imgs[-1]] * 6,
        duration=450,
        loop=0,
    )
    imgs[-1].save(out_dir / "05_kernel_alignment_final.png")
    for f in frames:
        f.unlink()

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.2))
    stages = [("Estado inicial (paso 0)", 0), (f"Kernel alineado (paso {total_steps})", total_steps)]
    for row, (label, step) in enumerate(stages):
        p_snap, K_snap, clf_snap, kta_val, acc_val = snapshots[step]
        Z = clf_snap.decision_function(np.array(qek_kernel_fn(grid, X_tr_p, p_snap))).reshape(gdx.shape)
        vmax = max(np.max(np.abs(Z)), 0.5) * 1.2

        axes[row, 0].contourf(gdx, gdy, Z, levels=18, cmap="RdBu", vmin=-vmax, vmax=vmax, alpha=0.40)
        axes[row, 0].contour(gdx, gdy, Z, levels=[0], colors="#111111", linewidths=2.0)
        axes[row, 0].scatter(X_train[:, 0], X_train[:, 1], c=col_tr, edgecolors="white", linewidths=0.9, s=46)
        axes[row, 0].scatter(X_test[:, 0], X_test[:, 1], c=col_te, edgecolors="#222222", linewidths=0.8, marker="^", s=58)
        axes[row, 0].set_aspect("equal")
        axes[row, 0].set_title(f"{label}\nPrecisión en test {acc_val:.1%}")

        im = axes[row, 1].imshow(K_snap, cmap="viridis", vmin=0, vmax=1)
        axes[row, 1].axhline(split_idx, color="white", ls="--", lw=1.3)
        axes[row, 1].axvline(split_idx, color="white", ls="--", lw=1.3)
        axes[row, 1].set_title(f"Matriz de kernel (KTA = {kta_val:.3f})")
        fig.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)

    fig.suptitle("Quantum Kernel Alignment sobre DoubleCake (inicial frente a alineado)",
                 fontsize=13.5, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "05_kernel_alignment_antes_despues.png", dpi=160, bbox_inches="tight")
    plt.show()


def plot_exponential_concentration(qubit_counts, sample_matrices, mean_iqp, var_iqp, var_angle_scaled, output_path):
    """Grafica mapas de calor para n=2, 6, 10 qubits y la curva logarítmica de varianza."""
    fig, axes = plt.subplots(1, 4, figsize=(18.5, 4.4))

    for idx, n_qubits in enumerate([2, 6, 10]):
        im = axes[idx].imshow(sample_matrices[n_qubits], cmap="viridis", vmin=0, vmax=1)
        mean_val = mean_iqp[qubit_counts.index(n_qubits)]
        axes[idx].set_title(f"Matriz $K$ con $n = {n_qubits}$ qubits\nMedia fuera de la diagonal {mean_val:.3f}")
        axes[idx].set_xlabel("Dato $j$")
        if idx == 0:
            axes[idx].set_ylabel("Dato $i$")
        fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

    ax_v = axes[3]
    ax_v.semilogy(qubit_counts, var_iqp, "o-", color=CURVE_RED, lw=2.5, markersize=7, label="IQP entrelazado (3 capas)")
    ax_v.semilogy(qubit_counts, var_angle_scaled, "s--", color=CURVE_BLUE, lw=2.2, markersize=6, label=r"Angle reescalado ($x/\sqrt{n}$)")
    ax_v.set_xlabel("Número de qubits $n$")
    ax_v.set_ylabel(r"$\mathrm{Var}_{x \neq x'}[k(x, x')]$ (escala log)")
    ax_v.set_title("Concentración exponencial\n(Thanasilp et al., 2024)")
    ax_v.set_xticks(qubit_counts)
    ax_v.grid(True, which="both", alpha=0.3)
    ax_v.legend(fontsize=9)

    fig.suptitle("Colapso de la matriz de kernel hacia la identidad al crecer el número de qubits",
                 fontsize=13.5, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.show()


def plot_shots_and_scaling(shots_list, means, stds, exact_acc, output_path):
    """Grafica el efecto del muestreo finito (shots) y el escalamiento cuadrático O(M^2)."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    ax1 = axes[0]
    m_arr = means * 100
    s_arr = stds * 100
    ax1.semilogx(shots_list, m_arr, "o-", color=CURVE_BLUE, lw=2.5, markersize=7, label="Media con shots finitos")
    ax1.fill_between(shots_list, m_arr - s_arr, np.minimum(m_arr + s_arr, 100), color=CURVE_BLUE, alpha=0.18, label="± 1 desvío estándar")
    ax1.axhline(exact_acc * 100, color="#27ae60", ls="--", lw=2.0, label=f"Simulador exacto ({exact_acc:.1%})")
    ax1.set_xlabel("Mediciones (shots $S$) por circuito (escala log)")
    ax1.set_ylabel("Precisión en test (%)")
    ax1.set_ylim(50, 105)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_title("Efecto del ruido estadístico de medición\n(dataset sintético de Havlíček et al., 2019)")
    ax1.legend(loc="lower right", fontsize=9.5)

    ax2 = axes[1]
    m_vals = np.linspace(20, 1500, 200)
    circuit_counts = m_vals * (m_vals - 1) / 2
    ax2.plot(m_vals, circuit_counts / 1e3, color=CURVE_RED, lw=2.8, label=r"Circuitos en train $\frac{M(M-1)}{2}$")
    milestones = [
        (100, 100 * 99 / 2, (90, 150)),
        (500, 500 * 499 / 2, (230, 360)),
        (1000, 1000 * 999 / 2, (560, 650)),
    ]
    for m_h, c_h, text_pos in milestones:
        ax2.scatter([m_h], [c_h / 1e3], color="black", s=50, zorder=5)
        ax2.annotate(
            f"M = {m_h}\n({c_h/1e3:,.1f}k circuitos)",
            xy=(m_h, c_h / 1e3), xytext=text_pos,
            arrowprops=dict(arrowstyle="->", color="#444444", lw=1.3), fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff8dc", ec="#d4a017"),
        )
    ax2.set_xlabel("Muestras de train $M$")
    ax2.set_ylabel("Circuitos distintos en la QPU (miles)")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Crecimiento cuadrático de la matriz de Gram")
    ax2.legend(loc="upper left", fontsize=9.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.show()
