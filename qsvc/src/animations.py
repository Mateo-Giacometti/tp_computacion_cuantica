"""Generates evolution GIFs (Kernel Alignment, exponential concentration, shot noise)."""
import sys
from pathlib import Path

ROOT_QSVC = Path(__file__).resolve().parent.parent
if str(ROOT_QSVC) not in sys.path:
    sys.path.insert(0, str(ROOT_QSVC))

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from PIL import Image
from sklearn.svm import SVC

from src import datasets as ds, plots as pl


def _save_gif(frame_paths, output_gif, duration=120, hold_last=10):
    imgs = [Image.open(p) for p in frame_paths]
    imgs[0].save(
        output_gif,
        save_all=True,
        append_images=imgs[1:] + [imgs[-1]] * hold_last,
        duration=duration,
        loop=0,
    )
    for p in frame_paths:
        p.unlink()
    print(f"GIF guardado en: {output_gif}")


def generate_concentration_gif(figures_dir):
    """Anima cómo la matriz de kernel colapsa a la identidad para n = 2..10 qubits."""
    out_dir = Path(figures_dir)
    rng = np.random.default_rng(7)
    qubit_counts = list(range(2, 11))
    n_samples = 24
    off_diag = ~np.eye(n_samples, dtype=bool)

    matrices, means_iqp, vars_iqp, vars_ang = [], [], [], []
    for n_q in qubit_counts:
        X_rand = rng.uniform(0, 2 * np.pi, size=(n_samples, n_q))
        dev_n = qml.device("default.qubit", wires=n_q)

        @qml.qnode(dev_n)
        def iqp_state_n(x):
            qml.IQPEmbedding(x, wires=range(n_q), n_repeats=3)
            return qml.state()

        @qml.qnode(dev_n)
        def angle_state_n(x):
            qml.AngleEmbedding(x / np.sqrt(n_q), wires=range(n_q), rotation="Y")
            return qml.state()

        s_iqp = np.array([iqp_state_n(x) for x in X_rand])
        K_iqp = np.abs(s_iqp.conj() @ s_iqp.T) ** 2

        s_ang = np.array([angle_state_n(x) for x in X_rand])
        K_ang = np.abs(s_ang.conj() @ s_ang.T) ** 2

        matrices.append(K_iqp)
        means_iqp.append(float(np.mean(K_iqp[off_diag])))
        vars_iqp.append(float(np.var(K_iqp[off_diag])))
        vars_ang.append(float(np.var(K_ang[off_diag])))

    frames = []
    for idx, n_q in enumerate(qubit_counts):
        fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))

        im = axes[0].imshow(matrices[idx], cmap="viridis", vmin=0, vmax=1)
        axes[0].set_title(f"Matriz de kernel $K$ (IQP) con $n = {n_q}$ qubits\nMedia fuera de la diagonal {means_iqp[idx]:.4f}")
        axes[0].set_xlabel("Dato $j$")
        axes[0].set_ylabel("Dato $i$")
        fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="Similitud $k(x_i, x_j)$")

        qs_so_far = qubit_counts[:idx + 1]
        axes[1].semilogy(qs_so_far, vars_iqp[:idx + 1], "o-", color=pl.CURVE_RED, lw=2.5, markersize=7,
                         label="IQP entrelazado (3 capas)")
        axes[1].semilogy(qs_so_far, vars_ang[:idx + 1], "s--", color=pl.CURVE_BLUE, lw=2.2, markersize=6,
                         label=r"Angle reescalado ($x/\sqrt{n}$)")
        axes[1].scatter([n_q], [vars_iqp[idx]], color=pl.CURVE_RED, s=110, edgecolors="black", zorder=6)
        axes[1].set_xlim(1.6, 10.4)
        axes[1].set_ylim(5e-7, 2e-1)
        axes[1].set_xticks(qubit_counts)
        axes[1].set_xlabel("Número de qubits $n$")
        axes[1].set_ylabel(r"$\mathrm{Var}_{x \neq x'}[k(x, x')]$ (escala log)")
        axes[1].set_title("Concentración exponencial (Thanasilp et al., 2024)")
        axes[1].grid(True, which="both", alpha=0.3)
        axes[1].legend(loc="lower left", fontsize=9.5)

        ratio = vars_iqp[0] / vars_iqp[idx]
        axes[1].text(
            0.97, 0.52, f"Caída de varianza respecto a n = 2\n{ratio:,.0f}x menor",
            transform=axes[1].transAxes, ha="right", va="center", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="#fff8dc", ec="#d4a017"),
        )

        fig.suptitle("Colapso progresivo de la matriz de kernel hacia la identidad",
                     fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        tmp = out_dir / f"_tmp_conc_{n_q:02d}.png"
        fig.savefig(tmp, dpi=135, bbox_inches="tight")
        plt.close(fig)
        frames.append(tmp)

    _save_gif(frames, out_dir / "06_concentracion_exponencial.gif", duration=650, hold_last=6)


def generate_shots_gif(figures_dir):
    """Anima cómo la matriz de kernel y la frontera del QSVC se estabilizan al aumentar S shots."""
    out_dir = Path(figures_dir)
    X_train, X_test, y_train, y_test, (gx, gy, grid_pts, _) = ds.make_havlicek_dataset()
    order = np.argsort(y_train)
    X_train, y_train = X_train[order], y_train[order]

    dev_2q = qml.device("default.qubit", wires=2)

    @qml.qnode(dev_2q)
    def zz_state(x):
        ds.zz_feature_map(x)
        return qml.state()

    def state_kernel(A, B):
        sa = np.array([zz_state(a) for a in A])
        sb = np.array([zz_state(b) for b in B])
        return np.abs(sa.conj() @ sb.T) ** 2

    K_tr_exact = state_kernel(X_train, X_train)
    K_te_exact = state_kernel(X_test, X_train)
    K_grid_exact = state_kernel(grid_pts, X_train)

    shots_seq = [10, 20, 40, 80, 150, 300, 600, 1500, None]
    shots_x_plot = [10, 20, 40, 80, 150, 300, 600, 1500, 5000]

    rng = np.random.default_rng(12)
    upper_idx = np.triu_indices_from(K_tr_exact, k=1)
    split_idx = np.sum(y_train == -1) - 0.5
    col_tr = pl._class_colors(y_train)
    col_te = pl._class_colors(y_test)

    acc_hist, err_hist = [], []
    frames = []

    for idx, n_shots in enumerate(shots_seq):
        if n_shots is None:
            K_tr_hat, K_te_hat, K_grid_hat = K_tr_exact, K_te_exact, K_grid_exact
            label_s = "Exacto (∞ shots)"
        else:
            K_tr_hat = np.ones_like(K_tr_exact)
            samp = rng.binomial(n_shots, np.clip(K_tr_exact[upper_idx], 0, 1)) / n_shots
            K_tr_hat[upper_idx] = samp
            K_tr_hat[(upper_idx[1], upper_idx[0])] = samp
            K_te_hat = rng.binomial(n_shots, np.clip(K_te_exact, 0, 1)) / n_shots
            K_grid_hat = rng.binomial(n_shots, np.clip(K_grid_exact, 0, 1)) / n_shots
            label_s = f"S = {n_shots} shots"

        clf = SVC(kernel="precomputed", C=10.0).fit(K_tr_hat, y_train)
        acc = clf.score(K_te_hat, y_test)
        mae = float(np.mean(np.abs(K_tr_hat - K_tr_exact)))
        acc_hist.append(acc * 100)
        err_hist.append(mae)

        Z = clf.decision_function(K_grid_hat).reshape(gx.shape)

        fig = plt.figure(figsize=(16.8, 5.1))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.08, 1.05, 1.15], hspace=0.20, wspace=0.36)
        ax_mat = fig.add_subplot(gs[:, 0])
        ax_bound = fig.add_subplot(gs[:, 1])
        ax_acc = fig.add_subplot(gs[0, 2])
        ax_err = fig.add_subplot(gs[1, 2], sharex=ax_acc)

        im = ax_mat.imshow(K_tr_hat, cmap="viridis", vmin=0, vmax=1)
        ax_mat.axhline(split_idx, color="white", ls="--", lw=1.2)
        ax_mat.axvline(split_idx, color="white", ls="--", lw=1.2)
        ax_mat.set_title(f"Matriz estimada ({label_s})\nError medio |K̂ - K| = {mae:.4f}")
        ax_mat.set_xlabel("Dato $j$"); ax_mat.set_ylabel("Dato $i$")
        fig.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04)

        vmax = max(np.max(np.abs(Z)), 0.5) * 1.2
        ax_bound.contourf(gx, gy, Z, levels=18, cmap="RdBu", vmin=-vmax, vmax=vmax, alpha=0.40)
        ax_bound.contour(gx, gy, Z, levels=[0], colors="#111111", linewidths=1.8)
        ax_bound.scatter(X_train[:, 0], X_train[:, 1], c=col_tr, edgecolors="white", linewidths=0.8, s=42, zorder=3)
        ax_bound.scatter(X_test[:, 0], X_test[:, 1], c=col_te, edgecolors="#222222", linewidths=0.8, marker="^", s=54, zorder=4)
        ax_bound.set_title(f"Frontera QSVC ({label_s})\nPrecisión en test {acc:.1%}")
        ax_bound.set_xlabel(r"$x_1$"); ax_bound.set_ylabel(r"$x_2$")
        ax_bound.set_xlim(0, 2 * np.pi); ax_bound.set_ylim(0, 2 * np.pi)

        xs = shots_x_plot[:idx + 1]
        cur_x = shots_x_plot[idx]
        ax_acc.semilogx(xs, acc_hist, "o-", color=pl.CURVE_BLUE, lw=2.3, markersize=6)
        ax_acc.axvline(cur_x, color="#888888", ls=":", lw=1.1, alpha=0.75)
        ax_acc.set_xlim(8, 7000)
        ax_acc.set_ylim(78, 105)
        ax_acc.set_ylabel("Precisión (%)", color=pl.CURVE_BLUE, fontweight="bold")
        ax_acc.set_title("Estabilización al aumentar shots")
        ax_acc.grid(True, which="both", alpha=0.3)
        plt.setp(ax_acc.get_xticklabels(), visible=False)

        ax_err.semilogx(xs, err_hist, "s-", color=pl.CURVE_RED, lw=2.2, markersize=5.5)
        ax_err.axvline(cur_x, color="#888888", ls=":", lw=1.1, alpha=0.75)
        ax_err.set_ylim(-0.005, 0.115)
        ax_err.set_xticks([10, 100, 1000, 5000])
        ax_err.set_xticklabels(["10", "100", "1000", "∞"])
        ax_err.set_xlabel("Cantidad de shots $S$ (escala log)")
        ax_err.set_ylabel("Error |K̂ - K|", color=pl.CURVE_RED, fontweight="bold")
        ax_err.grid(True, which="both", alpha=0.3)

        fig.suptitle("Impacto de las mediciones finitas (shots) sobre el QSVC",
                     fontsize=13.5, fontweight="bold", y=1.06)
        tmp = out_dir / f"_tmp_shots_{idx:02d}.png"
        fig.savefig(tmp, dpi=135, bbox_inches="tight")
        plt.close(fig)
        frames.append(tmp)

    _save_gif(frames, out_dir / "07_shots_estabilizacion.gif", duration=700, hold_last=6)


def regenerate_kta_gif(figures_dir):
    """Regenera el GIF de DoubleCake con el nuevo panel de doble eje Y."""
    out_dir = Path(figures_dir)
    Xdc_train, Xdc_test, ydc_train, ydc_test = ds.make_double_cake_dataset()
    n_qubits, n_layers = 4, 4
    dev_qek = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev_qek, interface="autograd")
    def qek_states(x1, x2, params):
        for layer in range(n_layers):
            for q in range(n_qubits):
                feature = x1 if q % 2 == 0 else x2
                qml.RX(params[layer, q, 0] * feature + params[layer, q, 1], wires=q)
                qml.RY(params[layer, q, 2] * (x1 * x2) + params[layer, q, 3], wires=q)
            for q in range(n_qubits):
                qml.CRZ(params[layer, q, 4], wires=[q, (q + 1) % n_qubits])
        return qml.state()

    def qek_kernel(A, B, params):
        sa = qek_states(A[:, 0], A[:, 1], params)
        sb = qek_states(B[:, 0], B[:, 1], params)
        ov = pnp.dot(pnp.conj(sa), pnp.transpose(sb))
        return pnp.real(ov * pnp.conj(ov))

    def kta_score(params, X, y):
        K = qek_kernel(X, X, params)
        return pnp.sum(K * pnp.outer(y, y)) / (len(y) * pnp.sqrt(pnp.sum(K * K) + 1e-12))

    pnp.random.seed(42)
    params = pnp.random.uniform(-0.25, 0.25, size=(n_layers, n_qubits, 5), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.12)

    n_steps = 55
    saved_steps = [0, 2, 4, 6, 8, 10, 14, 18, 24, 32, 42, 55]
    Xtr_p, Xte_p = pnp.array(Xdc_train, requires_grad=False), pnp.array(Xdc_test, requires_grad=False)
    ytr_p = pnp.array(ydc_train, requires_grad=False)

    hist_kta, hist_acc, snapshots = [], [], {}
    for step in range(n_steps + 1):
        kta_val = float(kta_score(params, Xtr_p, ytr_p))
        K_tr = np.array(qek_kernel(Xtr_p, Xtr_p, params))
        K_te = np.array(qek_kernel(Xte_p, Xtr_p, params))
        clf = SVC(kernel="precomputed", C=5.0).fit(K_tr, ydc_train)
        acc_val = clf.score(K_te, ydc_test)

        hist_kta.append(kta_val)
        hist_acc.append(acc_val)
        if step in saved_steps:
            snapshots[step] = (params.copy(), K_tr.copy(), clf, kta_val, acc_val)
        if step < n_steps:
            params = opt.step(lambda p: -kta_score(p, Xtr_p, ytr_p), params)

    # Evitar mostrar ventanas interactivas al correr desde CLI
    plt.switch_backend("Agg")
    pl.generate_kta_gif(
        Xdc_train, Xdc_test, ydc_train, ydc_test,
        snapshots, hist_kta, hist_acc, saved_steps, n_steps, qek_kernel, out_dir
    )


if __name__ == "__main__":
    plt.switch_backend("Agg")
    fig_dir = pl.init_environment(ROOT_QSVC / "figuras")
    regenerate_kta_gif(fig_dir)
    generate_concentration_gif(fig_dir)
    generate_shots_gif(fig_dir)
    print("GIFs de evolución generados con éxito.")
