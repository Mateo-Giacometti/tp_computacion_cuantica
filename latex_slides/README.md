# Generación de fórmulas en LaTeX para Google Slides

Este directorio contiene el instructivo y un script de ejemplo para generar ecuaciones y bloques de texto con matemática embebida en formato PNG con fondo transparente, listos para insertar en Google Slides.

El flujo utiliza compilación real con `lualatex` y rasterizado a `400 DPI` mediante Ghostscript (`gs`). Este enfoque evita solapamientos de subíndices o paréntesis, mantiene la misma escala tipográfica que el texto nativo de la presentación y permite combinar prosa en fuente `Carlito` (métricamente equivalente a `Calibri`) con simbología matemática estándar de LaTeX.

## Requisitos e instalación

El proceso requiere una distribución de LaTeX con soporte para `lualatex`, los paquetes `standalone`, `amsmath`, `amssymb`, `bm`, `xcolor` y `fontspec`, la fuente `Carlito` y Ghostscript (`gs`).

### Linux (Ubuntu / Debian)

```bash
sudo apt update
sudo apt install texlive-luatex texlive-latex-extra texlive-fonts-extra fonts-crosextra-carlito ghostscript
```

### macOS

```bash
brew install --cask mactex
brew install ghostscript
```

Para igualar la tipografía `Calibri` en macOS se puede instalar la fuente `Carlito` o configurar `\setsansfont{Calibri}` si Microsoft Office está instalado en el sistema.

### Windows

Instalar [TeX Live](https://tug.org/texlive/) o [MiKTeX](https://miktex.org/) junto con [Ghostscript](https://ghostscript.com/). En Windows la fuente `Calibri` viene preinstalada, por lo que puede indicarse directamente en `\setsansfont{Calibri}`. En caso de ejecutar Ghostscript desde consola en Windows, el binario suele llamarse `gswin64c`.

### Python

El script de ejemplo solo requiere Python 3 y `Pillow` para leer las dimensiones en píxeles de cada imagen generada y calcular su tamaño exacto en puntos (`pt`).

```bash
pip install pillow
```

## Estructura de la carpeta

- `README.md` documenta el procedimiento completo.
- `generate_latex_assets.py` compila en una sola corrida un conjunto de fórmulas y bloques de texto con matemática embebida definidos en la lista `BLOCKS`, y guarda los PNG resultantes en `output/` junto con un archivo `manifest.json` que informa las dimensiones exactas en puntos (`pt`).

## Uso rápido

Desde la raíz del repositorio o dentro de esta carpeta, ejecutar el script de ejemplo.

```bash
python latex_slides/generate_latex_assets.py
```

El script genera un documento `.tex` multipágina usando `\documentclass[multi=texpage,border=1.5pt]{standalone}`, lo compila con `lualatex` y convierte cada página del PDF a un PNG transparente de `400 DPI` mediante Ghostscript.

Al finalizar, imprime en consola el tamaño exacto de cada imagen en píxeles y en puntos (`pt`), y guarda esa misma información en `latex_slides/output/manifest.json`.

## Cómo agregar nuevas fórmulas o bloques de texto

Editar la lista `BLOCKS` en `generate_latex_assets.py`. Cada entrada es una tupla `(nombre_archivo, codigo_latex)`.

### 1. Ecuación aislada (display)

Para fórmulas centradas independientes se utiliza modo matemático con `\displaystyle` cuando hay sumatorias, integrales o fracciones, y el color principal del deck (`navy`).

```latex
{\color{navy}\fontsize{18bp}{22bp}\selectfont
$\displaystyle \mathrm{KTA}(\bm{\theta}) = \frac{\mathbf{y}^{T} K(\bm{\theta})\,\mathbf{y}}{M\,\|K(\bm{\theta})\|_{F}}$}
```

### 2. Sistema o ecuación en varias líneas

Para problemas de optimización o derivaciones en dos filas conviene envolver el contenido en `\begin{gathered} ... \end{gathered}` para controlar el espaciado vertical con `\\[6bp]`.

```latex
{\color{navy}\fontsize{18bp}{22bp}\selectfont
$\begin{gathered}
\min_{\bm{\theta},\, b}\; \frac{1}{2}\,\|\bm{\theta}\|^2 \\[8bp]
{\fontsize{14.5bp}{17bp}\selectfont \text{sujeto a}\quad y_i\!\left(\bm{\theta}^{T}\phi(\mathbf{x}_i) + b\right) \ge 1, \quad i = 1,\dots,M}
\end{gathered}$}
```

### 3. Párrafo o encabezado con matemática embebida

Cuando un texto de una tarjeta incluye símbolos como $U(\mathbf{x})$, $\phi(\mathbf{x})$ o $\bm{\theta}$, se encapsula dentro de un entorno `minipage` con el ancho exacto en puntos (`bp`) que ocupará en la diapositiva.

```latex
\begin{minipage}{362bp}\raggedright\hyphenpenalty=10000\exhyphenpenalty=10000
\fontsize{13bp}{16.2bp}\selectfont\color{body}
En un dataset de train con etiquetas $y \in \{-1, +1\}$, una SVM de clasificación busca el hiperplano $\bm{\theta}^{T}\mathbf{x} + b = 0$ de máximo margen.
\end{minipage}
```

Las directivas `\raggedright`, `\hyphenpenalty=10000` y `\exhyphenpenalty=10000` alinean el texto a la izquierda e impiden que LaTeX corte palabras con guiones al final de cada renglón, replicando el comportamiento visual de los cuadros de texto de Google Slides.

## Paleta de colores definida en el script

Los colores declarados en el preámbulo coinciden con la identidad visual de la presentación.

| Nombre en LaTeX | Código HEX | Uso habitual |
| :--- | :--- | :--- |
| `navy` | `#10182B` | Títulos y ecuaciones principales |
| `body` | `#3B4252` | Texto de cuerpo en tarjetas |
| `blueacc` | `#4353E0` | Acentos, etiquetas y restricciones destacadas |
| `orangeacc` | `#A4520A` | Énfasis secundario y bloques de CPU o cierre |
| `muted` | `#6B7285` | Pies de fórmula y notas secundarias |

## Inserción en Google Slides

1. Abrir la presentación en Google Slides e ir a `Insertar → Imagen → Subir desde la computadora`.
2. Seleccionar el archivo PNG generado dentro de `latex_slides/output/`.
3. Para mantener la escala tipográfica exacta en los bloques de texto con matemática embebida, verificar en `Opciones de formato → Tamaño y rotación` que el ancho y el alto en puntos coincidan con los valores `w_pt` y `h_pt` informados en `manifest.json` (equivalentes a `pixeles / 400 * 72`).
4. Para ecuaciones aisladas en paneles o recuadros, escalar únicamente desde las esquinas manteniendo fija la relación de aspecto.

## Alternativa manual para una sola fórmula

Si se necesita generar una única ecuación sin Python, guardar el código en `formula.tex` con `\documentclass[border=1.5pt]{standalone}` y ejecutar en terminal los siguientes comandos.

```bash
lualatex -interaction=nonstopmode formula.tex
gs -q -dNOPAUSE -dBATCH -sDEVICE=pngalpha -r400 -dTextAlphaBits=4 -dGraphicsAlphaBits=4 -sOutputFile=formula.png formula.pdf
```
