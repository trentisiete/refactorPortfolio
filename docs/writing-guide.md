# Guía de Escritura de Artículos en Markdown

Esta guía explica cómo escribir artículos `.md` para este proyecto Astro sin errores de renderizado.

---

## Frontmatter (Obligatorio)

Cada artículo debe empezar con este bloque YAML:

```yaml
---
title: "Título del artículo"
excerpt: "Descripción breve del artículo."
date: 2026-03-10
lang: es          # es | en | de
tags: ["tag1", "tag2"]
draft: true       # true = no se publica
---
```

> ⚠️ Si `draft: true`, el artículo NO aparecerá en la web.
>
> El tiempo de lectura se calcula automáticamente a partir del contenido; no se añade al frontmatter.

---

## Texto y Formato Básico

```markdown
Texto normal.

**Negrita**, *cursiva*, ***negrita cursiva***, ~~tachado~~.

> Cita o bloqueo de texto.

- Lista no ordenada
- Otro elemento

1. Lista ordenada
2. Otro elemento

[Texto del enlace](https://ejemplo.com)
```

---

## Ecuaciones Matemáticas (KaTeX)

El proyecto usa `remark-math` + `rehype-katex`. Soporta LaTeX estándar.

### ✅ Inline math

Escribe entre un solo `$`:

```markdown
La fórmula $E = mc^2$ es famosa.
```

### ✅ Display math (ecuaciones centradas)

Usa `$$` en su propia línea. **REGLAS CRÍTICAS:**

```markdown
Texto anterior.

$$
E = mc^2
$$

Texto siguiente.
```

### ❌ Errores comunes con ecuaciones

```markdown
<!-- ❌ NO: $$ pegado al texto -->
La fórmula es: $$
E = mc^2
$$

<!-- ❌ NO: Línea en blanco entre $$ y el contenido -->
$$

E = mc^2

$$

<!-- ❌ NO: \begin{equation} dentro de $$ -->
$$
\begin{equation}
E = mc^2
\end{equation}
$$

<!-- ❌ NO: \label{} — no se soporta -->
$$
E = mc^2 \label{eq:einstein}
$$
```

### Resumen de reglas para `$$`

| Regla | Descripción |
|---|---|
| Línea en blanco **antes** de `$$` | Obligatorio |
| Línea en blanco **después** de `$$` | Obligatorio |
| **Sin** línea en blanco entre `$$` y el contenido | Obligatorio |
| `$$` en su propia línea | Obligatorio |
| No usar `\begin{equation}` | Usar solo `$$` |
| No usar `\label{}` | No se soporta |

### Comandos LaTeX soportados

Todo lo que soporte KaTeX funciona. Ejemplos comunes:

```markdown
$\nabla_x \log p_t(x)$         <!-- gradiente -->
$\mathbb{E}[X]$                <!-- esperanza -->
$\mathcal{N}(0, I)$            <!-- distribución normal -->
$\frac{a}{b}$                  <!-- fracción -->
$\sum_{i=1}^{n} x_i$           <!-- sumatorio -->
$\int_0^T f(t) dt$             <!-- integral -->
$\sqrt{x^2 + y^2}$             <!-- raíz cuadrada -->
$x \sim p_{data}(x)$           <!-- tilde (distribuido como) -->
$\theta^*$                     <!-- superíndice -->
$\bar{W}(t)$                   <!-- barra superior -->
```

---

## Citas y Referencias (Footnotes)

Usa la sintaxis nativa de Markdown para notas al pie.

### ✅ En el texto

```markdown
Según Song et al. [^song2021], los modelos de difusión...
```

### ✅ Definición al final del archivo

```markdown
[^song2021]: Song, Y. et al. (2021). Score-Based Generative Modeling 
through Stochastic Differential Equations. ICLR.
```

### ❌ Errores comunes con footnotes

```markdown
<!-- ❌ NO: Formato de Pandoc/BibTeX -->
[@song2021]
[^@song2021]

<!-- ❌ NO: Sin definición al final del archivo -->
Texto con referencia [^mireferenciasindefenir].
```

### Reglas para footnotes

| Regla | Descripción |
|---|---|
| Formato `[^clave]` en el texto | Obligatorio |
| Definición `[^clave]: Texto...` al final | Obligatorio |
| Cada definición en su propia línea | Obligatorio |
| Línea en blanco entre definiciones | Recomendado |
| NO usar `[@clave]` ni `[^@clave]` | Esos son formatos de Pandoc |

---

## Imágenes y Figuras

### ✅ Imagen simple

```markdown
![Descripción de la imagen](ruta/a/imagen.png)
```

### ✅ Figura con caption (HTML)

```markdown
<figure>
<img src="mi_imagen.png" />
<figcaption>Descripción de la figura.</figcaption>
</figure>
```

### ❌ NO usar

```markdown
<!-- ❌ NO: atributos de LaTeX -->
<figure data-latex-placement="htbp">

<!-- ❌ NO: clases de LaTeX -->
<figure id="fig:algo" data-latex-placement="H">
```

---

## Tablas

Usa la sintaxis estándar de Markdown con `|`:

### ✅ Correcto

```markdown
| Columna 1 | Columna 2 | Columna 3 |
|---|---|---|
| Dato A | Dato B | Dato C |
| Dato D | Dato E | Dato F |
```

### ❌ NO usar tabla estilo Pandoc

```markdown
<!-- ❌ NO: Formato de tabla de Pandoc -->
  Columna 1         Columna 2
  --------          ---------
  Dato A            Dato B

  : Caption de la tabla
```

---

## Encabezados

### ✅ Correcto

```markdown
# Título principal (solo uno por artículo)
## Sección
### Subsección
#### Sub-subsección
```

### ❌ NO usar IDs de Pandoc

```markdown
<!-- ❌ NO -->
## Mi sección {#sec:mi_seccion}
## Mi sección {#sec:mi_seccion .unnumbered}
```

Los IDs se generan automáticamente a partir del texto del encabezado.

---

## Cross-references (Enlaces internos)

### ✅ Enlace a un encabezado

```markdown
Como se explica en la [sección anterior](#mi-sección).
```

El ID se genera automáticamente: el texto del heading se convierte a minúsculas, 
los espacios se reemplazan por `-` y se eliminan caracteres especiales.

### ❌ NO usar atributos de Pandoc

```markdown
<!-- ❌ NO -->
[Figura 1](#fig:mi_figura){reference-type="ref" reference="fig:mi_figura"}
```

---

## Código

### ✅ Inline

```markdown
Usa el comando `npm run dev` para iniciar.
```

### ✅ Bloque con sintaxis highlighting

````markdown
```python
def hello():
    print("Hola mundo")
```
````

Lenguajes soportados: `python`, `javascript`, `typescript`, `bash`, `css`, `html`, `json`, `yaml`, `markdown`, etc.

---

## Caracteres Especiales

### ❌ Evitar

| Carácter | Problema | Alternativa |
|---|---|---|
| `\"` | Se renderiza como `\"` | Usa `"` directamente |
| `\[texto\]` | Se convierte en math inline | Usa `texto` sin backslashes |
| `<br>` en medio de párrafos | Puede romper el parser | Usa una línea en blanco |

---

## Checklist antes de publicar

- [ ] El frontmatter tiene `title`, `date`, `lang` y `draft: false`
- [ ] Las ecuaciones `$$` tienen línea en blanco antes y después
- [ ] Las ecuaciones `$$` **no** tienen línea en blanco entre `$$` y el contenido
- [ ] Las citas usan formato `[^clave]` con definiciones al final
- [ ] Las tablas usan formato pipe `|`
- [ ] Los encabezados no tienen `{#id}`
- [ ] No hay `\begin{equation}`, `\label{}`, ni atributos de Pandoc
- [ ] El artículo compila: `npm run build` sin errores

---

## Stack técnico del proyecto

| Componente | Tecnología |
|---|---|
| Framework | Astro v5 |
| Markdown → HTML | remark + rehype |
| Ecuaciones | remark-math + rehype-katex (KaTeX) |
| Tablas/Footnotes | remark-gfm |
| CSS de artículos | `/styles/article.css` + `/styles/markdown.css` |

---

## Sistema de idiomas y traducción automática

El contenido vive en carpetas por idioma; la carpeta define el idioma y el
nombre de archivo (compartido entre idiomas) vincula las traducciones:

```
src/content/articles/es/mi-articulo.md    <- escrito a mano
src/content/articles/en/mi-articulo.md    <- generado automáticamente
src/content/articles/de/mi-articulo.md    <- generado automáticamente
```

### Flujo de publicación

1. Escribe el artículo en la carpeta de tu idioma (es, en o de da igual).
2. Cuando esté listo, pon `draft: false` y haz `git commit`.
3. El hook `pre-commit` ejecuta `scripts/translate.mjs`: traduce a los otros
   dos idiomas con la API de Claude y añade las traducciones al mismo commit.
4. `git push` y listo.

Los borradores (`draft: true`) no se traducen. Los archivos generados llevan
`translated: true` y `sourceHash` en el frontmatter: si editas la fuente, se
regeneran en el siguiente commit; si no, no se vuelve a llamar a la API.
Si algún día retocas una traducción a mano, quita su `translated: true` y el
sistema no la volverá a tocar.

### Requisitos

- `.env` en la raíz con `ANTHROPIC_API_KEY=...` (no se sube al repo).
- Los hooks se activan solos con `npm install` (postinstall). En un clon
  nuevo sin install: `git config core.hooksPath .githooks`.

### Comandos útiles

```bash
npm run translate:check   # ver qué está pendiente de traducir (sin API)
npm run translate         # traducir ahora, sin esperar al commit
```

Si el hook no encuentra la API key o falla la red, avisa pero nunca bloquea
el commit; lo pendiente se traduce en el siguiente commit o con `npm run translate`.
