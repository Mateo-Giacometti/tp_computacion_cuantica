"""Ejemplo de generación por lotes de fórmulas y textos con LaTeX para Google Slides.

Compila todos los bloques definidos en BLOCKS en una única ejecución de lualatex
y convierte cada página del PDF resultante a PNG con fondo transparente a 400 DPI
mediante Ghostscript.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
DPI = 400

# Cada tupla define (nombre_de_salida_sin_extension, cuerpo_latex).
# Se incluyen tres ejemplos representativos:
# 1. Ecuación aislada en una línea.
# 2. Problema de optimización en dos líneas.
# 3. Párrafo en Carlito (equivalente a Calibri) con simbología matemática embebida.
BLOCKS: list[tuple[str, str]] = [
    (
        "ejemplo_ecuacion_simple",
        r"""
{\color{navy}\fontsize{18bp}{22bp}\selectfont
$\displaystyle \mathrm{KTA}(\bm{\theta}) = \frac{\mathbf{y}^{T} K(\bm{\theta})\,\mathbf{y}}{M\,\|K(\bm{\theta})\|_{F}}$}
""",
    ),
    (
        "ejemplo_optimizacion_multilinea",
        r"""
{\color{navy}\fontsize{18bp}{22bp}\selectfont
$\begin{gathered}
\min_{\bm{\theta},\, b}\; \frac{1}{2}\,\|\bm{\theta}\|^2 \\[8bp]
{\fontsize{14.5bp}{17bp}\selectfont \text{sujeto a}\quad y_i\!\left(\bm{\theta}^{T}\phi(\mathbf{x}_i) + b\right) \ge 1, \quad i = 1,\dots,M}
\end{gathered}$}
""",
    ),
    (
        "ejemplo_texto_inline_latex",
        r"""
\begin{minipage}{362bp}\raggedright\hyphenpenalty=10000\exhyphenpenalty=10000
\fontsize{13bp}{16.2bp}\selectfont\color{body}
En un dataset de train con etiquetas $y \in \{-1, +1\}$, una SVM de clasificación busca el hiperplano $\bm{\theta}^{T}\mathbf{x} + b = 0$ de máximo margen, determinado por los support vectors.
\end{minipage}
""",
    ),
]

TEX_PREAMBLE = r"""\documentclass[multi=texpage,border=1.5pt]{standalone}
\usepackage{amsmath,amssymb,bm}
\usepackage{xcolor}
\usepackage{fontspec}
\setsansfont{Carlito}[
  BoldFont=Carlito Bold,
  ItalicFont=Carlito Italic,
  BoldItalicFont=Carlito Bold Italic
]
\renewcommand{\familydefault}{\sfdefault}
\definecolor{navy}{HTML}{10182B}
\definecolor{body}{HTML}{3B4252}
\definecolor{blueacc}{HTML}{4353E0}
\definecolor{orangeacc}{HTML}{A4520A}
\definecolor{muted}{HTML}{6B7285}
\newenvironment{texpage}{}{}
\begin{document}
"""


def find_ghostscript() -> str:
    """Devuelve el ejecutable de Ghostscript disponible en el sistema."""
    for candidate in ("gs", "gswin64c", "gswin32c"):
        if shutil.which(candidate):
            return candidate
    raise RuntimeError(
        "No se encontró Ghostscript en el PATH. Instalar 'ghostscript' antes de ejecutar el script."
    )


def build_assets(blocks: list[tuple[str, str]], output_dir: Path, dpi: int = DPI) -> dict[str, dict]:
    """Compila los bloques LaTeX a PDF y genera un PNG transparente por bloque."""
    if not shutil.which("lualatex"):
        raise RuntimeError(
            "No se encontró 'lualatex' en el PATH. Instalar TeX Live o MiKTeX con soporte LuaLaTeX."
        )

    gs_bin = find_ghostscript()
    output_dir.mkdir(parents=True, exist_ok=True)

    pages = [f"\\begin{{texpage}}\n{body.strip()}\n\\end{{texpage}}" for _, body in blocks]
    tex_source = TEX_PREAMBLE + "\n".join(pages) + "\n\\end{document}\n"

    tex_path = output_dir / "blocks.tex"
    tex_path.write_text(tex_source, encoding="utf-8")

    print(f"Compilando {len(blocks)} bloques con lualatex...")
    compile_res = subprocess.run(
        ["lualatex", "-interaction=nonstopmode", f"-output-directory={output_dir}", str(tex_path)],
        capture_output=True,
        text=True,
    )
    if compile_res.returncode != 0:
        raise RuntimeError(
            f"Error al compilar LaTeX (código {compile_res.returncode}):\n{compile_res.stdout[-2000:]}"
        )

    pdf_path = output_dir / "blocks.pdf"
    page_pattern = str(output_dir / "page_%02d.png")
    subprocess.run(
        [
            gs_bin,
            "-q",
            "-dNOPAUSE",
            "-dBATCH",
            "-sDEVICE=pngalpha",
            f"-r{dpi}",
            "-dTextAlphaBits=4",
            "-dGraphicsAlphaBits=4",
            f"-sOutputFile={page_pattern}",
            str(pdf_path),
        ],
        check=True,
    )

    manifest: dict[str, dict] = {}
    for index, (name, _) in enumerate(blocks, start=1):
        raw_png = output_dir / f"page_{index:02d}.png"
        target_png = output_dir / f"{name}.png"
        os.replace(raw_png, target_png)

        with Image.open(target_png) as img:
            w_px, h_px = img.size

        w_pt = round(w_px / dpi * 72.0, 2)
        h_pt = round(h_px / dpi * 72.0, 2)
        manifest[name] = {
            "file": target_png.name,
            "width_px": w_px,
            "height_px": h_px,
            "width_pt": w_pt,
            "height_pt": h_pt,
            "dpi": dpi,
        }
        print(f"  [{index:02d}] {target_png.name:36s} {w_px:4d}x{h_px:4d} px -> {w_pt:6.2f} x {h_pt:5.2f} pt")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nAssets generados en {output_dir} (metadatos en {manifest_path.name}).")
    return manifest


if __name__ == "__main__":
    build_assets(BLOCKS, OUTPUT_DIR, DPI)
