"""Reproducible synthetic datasets for the QNN notebook."""
import numpy as np
from sklearn.model_selection import train_test_split


def make_xor_quadrants(n_samples=160, noise=0.14, margin=0.22, seed=42):
    """Dataset 2D tipo XOR (cuadrantes alternados) en [-pi/2, pi/2]^2.

    La clase +1 ocupa los cuadrantes I y III (x1 * x2 > 0) y la clase -1
    ocupa los cuadrantes II y IV (x1 * x2 < 0). Es un problema clásico no
    linealmente separable que requiere capturar la interacción entre ambas
    características.
    """
    rng = np.random.default_rng(seed)
    n_per_quad = n_samples // 4
    signs = [(1, 1), (-1, -1), (1, -1), (-1, 1)]

    X_list, y_list = [], []
    for sx, sy in signs:
        x1 = sx * rng.uniform(margin, np.pi / 2 - 0.10, size=n_per_quad)
        x2 = sy * rng.uniform(margin, np.pi / 2 - 0.10, size=n_per_quad)
        pts = np.column_stack([x1, x2]) + rng.normal(0.0, noise, size=(n_per_quad, 2))
        pts = np.clip(pts, -np.pi / 2, np.pi / 2)
        label = 1 if sx * sy > 0 else -1
        X_list.append(pts)
        y_list.append(np.full(n_per_quad, label, dtype=int))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)
