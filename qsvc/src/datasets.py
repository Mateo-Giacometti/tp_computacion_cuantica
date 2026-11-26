"""Reproducible synthetic dataset generators and shot simulation helpers."""
import numpy as np
import pennylane as qml
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


def make_1d_blocks(n_total=60, seed=42):
    """Puntos en [-pi, pi] con la clase +1 en el centro y -1 en ambos extremos."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-np.pi, np.pi, size=(n_total, 1))
    y = np.where(np.abs(X[:, 0]) < np.pi / 2, 1, -1)

    mask = np.abs(np.abs(X[:, 0]) - np.pi / 2) > 0.12
    X, y = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.35, random_state=seed, stratify=y
    )
    order = np.argsort(y_train)
    return X_train[order], X_test, y_train[order], y_test


def load_moons_dataset(n_samples=160, noise=0.15, seed=0):
    """Dataset make_moons escalado a [0, pi] y ordenado por clase en train."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    X = MinMaxScaler((0, np.pi)).fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    order = np.argsort(y_train)
    return X_train[order], X_test, y_train[order], y_test


def zz_feature_map(x):
    """Circuito ZZ Feature Map de dos qubits (Havlíček et al., 2019)."""
    for _ in range(2):
        for i in range(2):
            qml.Hadamard(wires=i)
            qml.RZ(2.0 * x[i], wires=i)
        qml.CNOT(wires=[0, 1])
        qml.RZ(2.0 * (np.pi - x[0]) * (np.pi - x[1]), wires=1)
        qml.CNOT(wires=[0, 1])


def make_havlicek_dataset(n_per_class=45, margin=0.30, seed=123):
    """Dataset sintético de Havlíček et al. (2019) separable con ZZ Feature Map."""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def label_circuit(x):
        zz_feature_map(x)
        qml.RY(1.1, wires=0)
        qml.RX(-0.8, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RZ(0.9, wires=1)
        qml.RY(-0.7, wires=0)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    axis = np.linspace(0, 2 * np.pi, 55)
    gx, gy = np.meshgrid(axis, axis)
    grid_points = np.c_[gx.ravel(), gy.ravel()]
    landscape = np.array([float(label_circuit(pt)) for pt in grid_points]).reshape(gx.shape)

    rng = np.random.default_rng(seed)
    pos_pts, neg_pts = [], []
    while len(pos_pts) < n_per_class or len(neg_pts) < n_per_class:
        cand = rng.uniform(0, 2 * np.pi, size=2)
        val = float(label_circuit(cand))
        if val >= margin and len(pos_pts) < n_per_class:
            pos_pts.append(cand)
        elif val <= -margin and len(neg_pts) < n_per_class:
            neg_pts.append(cand)

    X = np.vstack([pos_pts, neg_pts])
    y = np.array([1] * n_per_class + [-1] * n_per_class)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=7, stratify=y
    )
    return X_train, X_test, y_train, y_test, (gx, gy, grid_points, landscape)


def make_double_cake_dataset(n_per_sector=10, n_sectors=4, seed=42):
    """Dataset DoubleCake con dos anillos concéntricos y sectores alternados."""
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    rings = [(0.18, 0.52), (0.66, 1.02)]
    for ring_idx, (r_min, r_max) in enumerate(rings):
        for sec in range(n_sectors):
            ang_min = sec * (2 * np.pi / n_sectors) + 0.12
            ang_max = (sec + 1) * (2 * np.pi / n_sectors) - 0.12
            r_vals = np.sqrt(rng.uniform(r_min**2, r_max**2, size=n_per_sector))
            theta_vals = rng.uniform(ang_min, ang_max, size=n_per_sector)
            pts = np.c_[r_vals * np.cos(theta_vals), r_vals * np.sin(theta_vals)]
            label = 1 if (sec + ring_idx) % 2 == 0 else -1
            X_list.append(pts)
            y_list.append(np.full(n_per_sector, label))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    order = np.argsort(y_train)
    return X_train[order], X_test, y_train[order], y_test


def simulate_finite_shots(K_train, K_test, y_train, y_test, shots_list, repeats=25, C=10.0, seed=99):
    """Simula la estimación binomial de matrices de kernel con S shots finitos."""
    rng = np.random.default_rng(seed)
    upper_idx = np.triu_indices_from(K_train, k=1)
    means, stds = [], []
    for n_shots in shots_list:
        scores = []
        for _ in range(repeats):
            K_tr_hat = np.ones_like(K_train)
            samples = rng.binomial(n_shots, np.clip(K_train[upper_idx], 0.0, 1.0)) / n_shots
            K_tr_hat[upper_idx] = samples
            K_tr_hat[(upper_idx[1], upper_idx[0])] = samples

            K_te_hat = rng.binomial(n_shots, np.clip(K_test, 0.0, 1.0)) / n_shots
            clf = SVC(kernel="precomputed", C=C).fit(K_tr_hat, y_train)
            scores.append(clf.score(K_te_hat, y_test))
        means.append(float(np.mean(scores)))
        stds.append(float(np.std(scores)))
    return np.array(means), np.array(stds)
