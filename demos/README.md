# Demos de Quantum Machine Learning

Tres scripts cortos en PennyLane para mostrar en la presentación. Corren en
una notebook común (simulador, sin hardware cuántico) y guardan sus figuras en
`salidas/`.

| Script | Qué muestra | Tiempo aprox. | Resultado de referencia |
| --- | --- | --- | --- |
| `01_clasificador_reuploading.py` | Un solo qubit aprende a separar puntos dentro/fuera de un círculo (data re-uploading). GIF de la frontera de decisión. | 5 s | ~91 % en test |
| `02_mesetas_esteriles.py` | Varianza del gradiente vs. número de qubits: cae exponencialmente con circuitos profundos y costo global. | 35 s | 6e-2 → 4e-4 de 2 a 10 qubits |
| `03_kernel_cuantico.py` | SVM con kernel cuántico (tipo ZZ feature map) vs. SVM clásica RBF en el dataset "moons". | 5 s | QSVC ~85 %, RBF ~90 % |

## Cómo correrlas

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python 01_clasificador_reuploading.py
python 02_mesetas_esteriles.py
python 03_kernel_cuantico.py
```

## Referencias

- Pérez-Salinas et al., *Data re-uploading for a universal quantum classifier*, Quantum 4, 226 (2020). https://arxiv.org/abs/1907.02085
- McClean et al., *Barren plateaus in quantum neural network training landscapes*, Nat. Commun. 9, 4812 (2018). https://arxiv.org/abs/1803.11173
- Havlíček et al., *Supervised learning with quantum-enhanced feature spaces*, Nature 567, 209 (2019). https://arxiv.org/abs/1804.11326
