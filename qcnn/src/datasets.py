"""Reproducible quantum (TFIM) and classical (digits) datasets for the QCNN notebook."""
import numpy as np
from scipy.sparse import csr_matrix, identity, kron
from scipy.sparse.linalg import eigsh
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def _op_on_sites(n_spins, ops_dict):
    """Construye el producto tensorial disperso de operadores de Pauli en una cadena."""
    i2 = identity(2, format="csr", dtype=np.float64)
    mat = ops_dict.get(0, i2)
    for site in range(1, n_spins):
        mat = kron(mat, ops_dict.get(site, i2), format="csr")
    return mat


def make_tfim_dataset(n_spins=8, h_min=0.1, h_max=2.0, n_grid=39):
    """Estados fundamentales exactos del modelo de Ising con campo transversal (TFIM).

    Hamiltoniano con condiciones de borde periódicas:
        H(h) = -sum_i Z_i Z_{i+1} - h * sum_i X_i

    Con h < 1 el sistema está en la fase ordenada (+1) y con h > 1 en la fase
    desordenada (-1). Devuelve un conjunto chico de entrenamiento lejos de la
    transición y toda la grilla de campos h para evaluar la generalización.
    """
    z_mat = csr_matrix([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
    x_mat = csr_matrix([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)

    h_zz = sum(-_op_on_sites(n_spins, {i: z_mat, (i + 1) % n_spins: z_mat}) for i in range(n_spins))
    h_x = sum(-_op_on_sites(n_spins, {i: x_mat}) for i in range(n_spins))

    def ground_state(h_val):
        _, vecs = eigsh(h_zz + h_val * h_x, k=1, which="SA")
        psi = vecs[:, 0]
        # Fija el signo global para continuidad entre valores de h
        return -psi if psi[np.argmax(np.abs(psi))] < 0 else psi

    hs_grid = np.linspace(h_min, h_max, n_grid)
    states_grid = np.array([ground_state(h) for h in hs_grid], dtype=np.float64)

    # 12 estados de entrenamiento simétricos respecto de h = 1, dejando [0.70, 1.30] sin ver
    hs_train = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65,
                         1.35, 1.45, 1.55, 1.65, 1.75, 1.85], dtype=np.float64)
    states_train = np.array([ground_state(h) for h in hs_train], dtype=np.float64)
    y_train = np.where(hs_train < 1.0, 1.0, -1.0)

    return hs_train, states_train, y_train, hs_grid, states_grid


def load_digits_01_pca8(n_train=60, seed=42):
    """Dígitos 0 vs. 1 de scikit-learn (8x8) reducidos a 8 features mediante PCA.

    Sigue el esquema estándar de Hur, Kim y Park (2022) para evaluar una QCNN
    de 8 qubits sobre imágenes clásicas con codificación RY(pi * x_i).
    """
    digits = load_digits()
    mask = (digits.target == 0) | (digits.target == 1)
    X_img = digits.images[mask]
    X_flat = digits.data[mask]
    y = np.where(digits.target[mask] == 0, 1.0, -1.0)

    X_tr_raw, X_te_raw, img_tr, img_te, y_tr, y_te = train_test_split(
        X_flat, X_img, y, test_size=0.30, random_state=seed, stratify=y
    )

    pca = PCA(n_components=8, random_state=seed)
    scaler = MinMaxScaler((0.0, 1.0))
    X_tr_8 = scaler.fit_transform(pca.fit_transform(X_tr_raw))
    X_te_8 = np.clip(scaler.transform(pca.transform(X_te_raw)), 0.0, 1.0)

    return (
        X_tr_8[:n_train],
        X_te_8,
        y_tr[:n_train],
        y_te,
        img_te,
    )
