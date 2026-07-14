---
title: "Modelos sustitutos para la búsqueda de entradas óptimas"
topic: "Trabajo de Fin de Grado"
date: 2026-07-10
excerpt: "Procesos gaussianos, optimización bayesiana y Expected Improvement para priorizar nuevas evaluaciones cuando experimentar es costoso."
# tags: ["modelos sustitutos", "procesos gaussianos", "optimización bayesiana"]
draft: false
lang: "es"
kind: "article"
readingTime: 125
---

Este artículo recoge mi Trabajo de Fin de Grado en Ciencia e Ingeniería de Datos. Estudia cómo utilizar procesos gaussianos como modelos sustitutos para aprender con pocas evaluaciones, cuantificar la incertidumbre y decidir qué configuraciones conviene probar a continuación.

El código, los experimentos y los materiales de reproducción están disponibles en GitHub:

<https://github.com/trentisiete/surrogate_models>

---

## Resumen

En numerosos contextos experimentales, la principal dificultad no reside únicamente en analizar los resultados obtenidos, sino en decidir qué experimentos realizar a continuación cuando cada evaluación supone un coste elevado. En este contexto, el Grupo de Ingredientes Alimentarios Saludables de la UAM nos planteó un problema orientado a optimizar la selección de nuevas configuraciones experimentales en la alimentación de larvas *Hermetia illucens*.

La herramienta idónea para estos problemas son los modelos sustitutos, que consisten en construir un modelo de aprendizaje automático que aproxime la relación entre las características de una dieta y los resultados que cabría esperar de ella. De este modo, antes de realizar un nuevo ensayo, el modelo puede ayudar a estimar qué combinaciones de alimentos parecen más prometedoras y con qué grado de confianza se realizan tales predicciones. En particular, los modelos sustitutos estarán basados en procesos gaussianos, ya que permiten combinar predicción e incertidumbre en un mismo marco.

El TFG se desarrolla en tres etapas. En primer lugar, se establece el marco teórico necesario para entender los modelos sustitutos, los procesos gaussianos y el modo en que la incertidumbre puede utilizarse para proponer nuevas evaluaciones. En segundo lugar, se realizan experimentos sintéticos sobre funciones conocidas, lo que permite estudiar de forma controlada el comportamiento de estas herramientas: partir de pocos datos iniciales, ajustar un modelo, seleccionar nuevos puntos mediante *Expected Improvement* y comprobar si el proceso mejora el mejor resultado encontrado. Estos experimentos muestran que el procedimiento propuesto mejora el diseño inicial en todos los benchmarks analizados, aunque su rendimiento depende de diversos factores.

Finalmente, la metodología se aplica al caso real de dietas para *Hermetia illucens*. En este escenario, el modelo se evalúa dejando fuera dietas completas durante el entrenamiento y comprobando si es capaz de predecir sus resultados. Esta forma de evaluación se eligió porque se aproxima al uso real que se busca: estimar el comportamiento de una dieta todavía no ensayada y escoger el modelo que mejor generaliza a nuevas propuestas. Los resultados indican que, en los tres objetivos finalmente analizados, el proceso gaussiano mejora al modelo base, aunque la incertidumbre predictiva aparece parcialmente calibrada.

En conjunto, el trabajo muestra que los modelos sustitutos no reemplazan la validación experimental, pero sí pueden reducir el espacio de búsqueda, cuantificar incertidumbre y priorizar nuevas configuraciones candidatas cuando el presupuesto experimental es limitado.

## Introducción

Muchos problemas reales de optimización consisten en encontrar una configuración de entrada que produzca una respuesta favorable. Sin embargo, en numerosos contextos de ingeniería, ciencia de datos o experimentación física, evaluar cada configuración puede ser costoso, lento o limitado. En estos casos, no es viable explorar exhaustivamente todo el espacio de búsqueda ni aplicar directamente métodos que requieran muchas evaluaciones de la función objetivo. El problema deja de ser únicamente encontrar el mejor punto, y pasa a ser decidir cómo aprender lo máximo posible a partir de un número reducido de observaciones.

Este escenario aparece cuando la función que se desea optimizar se comporta como una caja negra: se pueden observar salidas para ciertas entradas, pero no se dispone de una expresión analítica sencilla ni de derivadas que permitan guiar directamente la búsqueda. Además, las observaciones pueden estar afectadas por ruido o variabilidad experimental. Por tanto, cualquier estrategia de optimización debe equilibrar dos necesidades: aproximar el comportamiento de la función con los datos disponibles y decidir qué nuevas configuraciones conviene evaluar.

En la literatura de optimización basada en modelos sustitutos, este tipo de problemas se aborda construyendo aproximaciones baratas de evaluar a partir de un conjunto limitado de observaciones. Este enfoque se ha utilizado especialmente en contextos de diseño e ingeniería, donde las evaluaciones pueden proceder de simulaciones de alta fidelidad, ensayos físicos o procesos experimentales costosos. De forma general, el procedimiento combina un diseño inicial de experimentos, el ajuste de un modelo aproximado, su validación y, cuando es posible, una fase secuencial de enriquecimiento del conjunto de datos mediante nuevos puntos de evaluación.

Dentro de esta familia de métodos, los procesos gaussianos ocupan un papel destacado porque no solo proporcionan una predicción puntual, sino también una estimación de incertidumbre asociada a cada predicción. Esta incertidumbre permite distinguir entre regiones bien explicadas por los datos disponibles y regiones todavía poco exploradas. Por ello, resulta especialmente útil en estrategias de *infill*, donde funciones de adquisición como Expected Improvement (EI) combinan explotación de zonas prometedoras y exploración de zonas inciertas para proponer nuevas evaluaciones.

En este contexto, el presente TFG estudia el uso de procesos gaussianos como modelos sustitutos probabilísticos en dos escenarios complementarios. En primer lugar, se utilizan benchmarks sintéticos, donde la función objetivo es conocida y puede evaluarse de forma controlada, para analizar el ciclo completo de optimización secuencial mediante GP + EI. En segundo lugar, se incluye como prueba aplicada un caso real basado en datos experimentales de dietas para *Hermetia illucens*. En este segundo escenario, las observaciones proceden de ensayos ya realizados y obtener nuevas muestras requeriría experimentación adicional, por lo que el objetivo no es afirmar una dieta óptima definitiva, sino evaluar si el modelo puede generalizar a dietas no vistas y proponer candidatos razonables para una posible evaluación futura.

Por tanto, el trabajo no pretende presentar los modelos sustitutos como sustitutos de la validación experimental. Su papel se entiende como una herramienta de apoyo: permiten reducir el espacio de búsqueda, cuantificar incertidumbre y priorizar nuevas evaluaciones cuando el presupuesto experimental es limitado. Esta distinción es especialmente importante en el caso real, donde las predicciones del modelo deben interpretarse como hipótesis experimentales y no como conclusiones biológicas cerradas.

### Objetivos

El objetivo general de este TFG es desarrollar y evaluar un marco de optimización basada en modelos sustitutos probabilísticos, centrado en procesos gaussianos (Gaussian Processes, GP), para aproximar funciones objetivo con evaluaciones limitadas y apoyar la búsqueda de configuraciones prometedoras.

Para alcanzar este objetivo general, se plantean los siguientes objetivos específicos:

1.  Estudiar los fundamentos de los modelos sustitutos y, en particular, de los procesos gaussianos como modelos probabilísticos capaces de proporcionar predicción puntual e incertidumbre.

2.  Implementar un marco experimental basado en procesos gaussianos y EI para analizar el ciclo completo de optimización secuencial en funciones benchmark sintéticas.

3.  Evaluar en los benchmarks si el proceso de *infill* permite mejorar el mejor valor encontrado respecto al diseño inicial, y analizar cómo influyen factores como el kernel, el tamaño inicial, el ruido y la dimensionalidad del problema.

4.  Comparar la calidad predictiva del GP con su utilidad para guiar la optimización, diferenciando entre métricas de error, métricas probabilísticas y mejora del *incumbent*.

5.  Aplicar el enfoque a un caso real de dietas para *Hermetia illucens*, construyendo una representación supervisada a partir de variables de formulación, composición nutricional y tipo de subproducto.

6.  Evaluar la capacidad de generalización del modelo en el caso real mediante un protocolo *Leave-One-Diet-Out*, evitando fuga de información entre réplicas de una misma dieta.

7.  Analizar si la incertidumbre predictiva del GP puede servir como señal de apoyo para interpretar la fiabilidad de las predicciones y priorizar nuevas formulaciones candidatas.

8.  Plantear una fase prospectiva de pre-*infill* que permita identificar candidatos para futuras evaluaciones experimentales.

Estos objetivos están diseñados para cubrir tanto la parte metodológica como la parte aplicada del trabajo. Los benchmarks permiten comprobar el comportamiento del enfoque en un entorno controlado, mientras que el caso real permite estudiar sus posibilidades y limitaciones cuando los datos son escasos, agrupados por dieta y procedentes de ensayos experimentales del caso real.

### Estructura del documento

El resto del documento se organiza del siguiente modo. El Capítulo 2 introduce el marco teórico necesario para el trabajo, incluyendo la optimización con evaluaciones costosas, los modelos sustitutos, los procesos gaussianos, el diseño experimental, las funciones de adquisición y las métricas de evaluación. El Capítulo 3 presenta la experimentación realizada, primero sobre benchmarks sintéticos y después sobre el caso real de dietas para insectos. Finalmente, el Capítulo 4 resume las principales conclusiones obtenidas y plantea posibles líneas de trabajo futuro.

## Fundamentos teóricos de la optimización basada en modelos sustitutos

Muchos problemas reales de optimización en ingeniería y ciencia de datos pueden formularse mediante una función objetivo $f : X \to \mathbb{R}$ cuyo valor solo puede obtenerse evaluando un procedimiento externo, como una simulación numérica compleja o un experimento físico. En este contexto, se dice que la función es de *caja negra* cuando, a efectos prácticos, no se dispone de una expresión analítica manejable de $f$, ni de derivadas, ni de propiedades estructurales que permitan aplicar directamente métodos clásicos de optimización. En otras palabras, la función solo puede consultarse en puntos concretos del dominio y devolver una salida asociada a cada entrada.

Además, en muchos de estos problemas cada evaluación de $f(x)$ tiene un coste elevado, ya sea en tiempo de cómputo o uso de recursos. Así, la dificultad no reside únicamente en que la función sea desconocida internamente, sino también en que solo puede evaluarse un número reducido de veces.

Con el fin de formalizar este problema, introducimos un espacio de búsqueda continuo $\mathcal{X}\subset\mathbb{R}^{d}$ que contiene todas las posibles configuraciones de las variables de entrada, junto con una función objetivo $f:\mathcal{X}\to\mathbb{R}$ que modela el comportamiento del sistema. El propósito de la *optimización global* es encontrar un minimizador global

$$
\bm{x}^{\star} \in \arg\min_{\bm{x}\in\mathcal{X}} f(\bm{x}).
$$

Nuestra hipótesis de trabajo es que el coste $c(\bm{x})$ de evaluar la función objetivo es elevado. En consecuencia, el número de evaluaciones que pueden realizarse queda limitado por un presupuesto máximo $B$, de modo que solo se dispone de un conjunto reducido de observaciones:

$$
\mathcal{D}_n = \{(\bm{x}_i, y_i)\}_{i=1}^{n}, \qquad n \leq B,
$$

donde $y_i$ representa el valor observado al evaluar la función en el punto $\bm{x}_i$.

Incluso en aquellos casos en los que el coste individual de evaluación fuera asumible, la búsqueda exhaustiva seguiría siendo inviable cuando la dimensión del espacio $\mathcal{X}$ es elevada. En efecto, el número de puntos necesarios para explorar el dominio con una densidad comparable crece exponencialmente con la dimensión, fenómeno conocido como *maldición de la dimensionalidad*. Por tanto, la dificultad del problema no proviene únicamente del coste por evaluación, sino también del tamaño efectivo del espacio de búsqueda.

Además del coste, en muchos problemas las evaluaciones están afectadas por *ruido*. En experimentos físicos, el ruido proviene de errores de medida, variabilidad del entorno o diferencias entre réplicas. En simulaciones complejas, puede aparecer *ruido numérico* debido a múltiples causas como fenómenos estocásticos internos. Una forma estándar de modelar esta situación es asumir que la observación es una perturbación de la función subyacente:

$$
y_i = f(\bm{x}_i) + \varepsilon_i,
\qquad
\mathbb{E}[\varepsilon_i]=0,
\qquad
\mathrm{Var}(\varepsilon_i)=\sigma_\varepsilon^2.
$$

Estas dos restricciones (presupuesto limitado y/o ruido) hacen que muchos optimizadores clásicos resulten poco adecuados: los métodos basados en gradientes requieren derivadas (normalmente no disponibles en caja negra) y pueden ser inestables bajo ruido. En consecuencia, el problema se transforma en una búsqueda bajo escasez de datos: con pocas evaluaciones debemos (i) aproximar $f$ lo suficiente como para guiar la búsqueda y (ii) decidir adaptativamente dónde invertir las siguientes evaluaciones para mejorar la solución global.

Esta motivación conduce de forma natural al enfoque de *optimización basada en modelos sustitutos* (SBO) que consiste en construir un modelo estadístico $\hat{f}$ a partir de $\mathcal{D}_n$ y emplearlo para proponer nuevas evaluaciones, reduciendo de manera drástica el número de consultas directas a la función costosa (Forrester et al. 2008; Jiang et al. 2020). En las siguientes secciones se introducen estos modelos y cómo se integran.

### Modelos sustitutos: definición y taxonomía

En este contexto de optimización con evaluaciones costosas, un *modelo sustituto* (o *metamodelo*) se define como una aproximación $\hat{f}$ aprendida a partir de un conjunto finito de observaciones $\mathcal{D}_n=\{(\bm{x}_i,y_i)\}_{i=1}^n,$ con el objetivo de emular la respuesta del sistema real a un coste computacional muy inferior al de evaluar directamente la función objetivo original (Forrester et al. 2008; Jiang et al. 2020).
Un modelo sustituto puede emplearse con dos fines principales:

1.  **Predicción rápida y análisis del espacio de diseño.** Una vez entrenado, $\hat{f}$ permite evaluar muchas configuraciones de entrada para comprender tendencias, analizar sensibilidad o explorar regiones de interés sin ejecutar el sistema costoso.

2.  **Optimización con presupuesto limitado.** En SBO, el sustituto se utiliza para guiar la selección de nuevas evaluaciones del sistema real, reduciendo el número de ejecuciones costosas necesarias para aproximarse a la solución óptima (Jiang et al. 2020).

Existe gran variedad de modelos que pueden actuar como sustitutos. Una taxonomía orientada a la práctica, es la siguiente:

- **Según la naturaleza de la predicción:**

  **Deterministas**

  Producen una predicción puntual $\hat{f}(\bm{x})$.

  **Probabilísticos**

  Cuantifican incertidumbre (p. ej. mediante una distribución predictiva). Esta distinción es relevante cuando se quieren criterios de muestreo adaptativo basados en exploración/explotación (Forrester et al. 2008; Jiang et al. 2020).

- **Según la flexibilidad del modelo:**

  **Paramétricos**

  (p. ej. regresiones polinómicas). Simples e interpretables, pero potencialmente rígidos si la respuesta presenta multimodalidad o alta no linealidad.

  **No paramétricos o de alta flexibilidad**

  (p. ej. métodos kernel, árboles, *ensembles*). Capaces de capturar respuestas complejas, a costa de más parámetros y riesgo de sobreajuste con pocos datos (Forrester et al. 2008).

- **Según cómo se integra con la optimización:**

  **Offline**

  Se entrena una única vez con $\mathcal{D}_n$ y se utiliza para proponer candidatos (útil cuando obtener nuevas evaluaciones es inviable en el corto plazo).

  **Secuencial/online**

  El sustituto se actualiza iterativamente añadiendo nuevas evaluaciones seleccionadas de forma informada (infill/adaptive sampling), hasta agotar presupuesto o alcanzar un criterio de parada (Forrester et al. 2008; Jiang et al. 2020).

A pesar de la diversidad de modelos, el proceso general de modelado sustituto sigue un flujo de trabajo estándar:

1.  **Definir variables y dominio.** Se establecen las variables de decisión (entradas) y sus rangos, fijando el dominio $\mathcal{X}$.

2.  **Muestreo inicial (DoE).** Se selecciona un conjunto inicial de puntos $\{\bm{x}_i\}_{i=1}^n$ que cubra el espacio de forma razonablemente uniforme. En problemas continuos es habitual usar muestreos *space-filling*, como Latin Hypercube Sampling (LHS) o SOBOL (Forrester et al. 2008).

3.  **Evaluación del sistema real.** Se calculan las salidas $y_i$ con el sistema real y se construye el dataset inicial $\mathcal{D}_n$.

4.  **Ajuste y validación del sustituto.** Se elige una familia de modelos, se estiman sus parámetros y se evalúa su calidad predictiva con prácticas estándar (p. ej. validación cruzada), teniendo presente compromisos entre complejidad y robustez (Forrester et al. 2008).

5.  **Refinamiento secuencial: muestreo adaptativo o proceso de *infill***. Cuando el presupuesto lo permite, se añaden nuevos puntos (*infill points*) seleccionados por un criterio que prioriza: (i) regiones donde el modelo es probablemente inexacto (exploración), y/o (ii) regiones prometedoras respecto al óptimo (explotación). Este ciclo se repite hasta alcanzar un criterio de parada (en base a presupuesto máximo, mejora marginal, o precisión objetivo) (Forrester et al. 2008; Jiang et al. 2020).

En este TFG, los modelos sustitutos se emplean principalmente con fines de optimización bajo presupuesto limitado, sin perder su utilidad como herramienta de análisis e interpretación del problema. Esta idea se estudiará en dos escenarios complementarios: un conjunto de *benchmarks*, donde la función objetivo puede evaluarse de forma controlada para analizar el ciclo secuencial completo (pasando de $n=0$ o $n$ con muestreo inicial, a $B$ muestras), y un *caso real*, donde el modelo se entrenará con los datos disponibles para evaluar su rendimiento y proponer configuraciones candidatas $\bm{x}^\star$ para una posible validación futura.

### El Proceso Gaussiano como modelo sustituto

Siguiendo principalmente a Rasmussen y Williams (Rasmussen and Williams 2006), en esta subsección se introduce el GP como modelo bayesiano de regresión no paramétrica. Su interés en el contexto de los modelos sustitutos probabilísticos reside en que, además de proporcionar una predicción media, ofrece una cuantificación analítica de la incertidumbre, esencial en optimización bayesiana (BO). Para ello, se presentará primero la definición del GP como distribución sobre funciones y, después, su interpretación a partir de la regresión lineal bayesiana en espacio de características. La Figura 2.1 ilustra esta idea mostrando el paso de una distribución prior sobre funciones a una distribución posterior tras observar datos.

![(a) Muestras de funciones generadas por el prior del GP. (b) Distribución posterior tras condicionar en dos observaciones; la línea negra representa la media predictiva y la banda sombreada el intervalo predictivo del 95 %.](/assets/articles/surrogate_models/fig_gp_prior_posterior.png)

#### Definición y consistencia

Formalmente, un GP es un proceso estocástico definido sobre el espacio de entrada $\mathcal{X}$ tal que, para cualquier conjunto finito de puntos $X=\{x_1,\dots,x_n\}\subset\mathcal{X}$, el vector de valores de la función

$$
\mathbf{f}_X = \bigl(f(x_1),\dots,f(x_n)\bigr)^\top
$$

sigue una distribución normal multivariante. En consecuencia, existe una función de media $m:\mathcal{X}\to\mathbb{R}$ y una función de covarianza $k:\mathcal{X}\times\mathcal{X}\to\mathbb{R}$ tales que

$$
\mathbf{f}_X \sim \mathcal{N}\!\bigl(\mathbf{m}_X,\mathbf{K}_{XX}\bigr),
$$

donde

$$


\mathbf{m}_X =
\begin{bmatrix}
m(x_1)\\
\vdots\\
m(x_n)
\end{bmatrix},
\qquad
(\mathbf{K}_{XX})_{ij}=k(x_i,x_j).

$$

De forma abreviada, se escribe

$$
f(x)\sim \mathcal{GP}\!\bigl(m(x),k(x,x')\bigr).
$$

La expresión anterior define el prior del GP sobre cualquier conjunto finito de puntos. Por tanto, especificar un proceso gaussiano equivale a fijar dos objetos: una función de media $m(x)$, que recoge la tendencia global esperada de la función, y una función de covarianza $k(x,x')$, también llamada kernel, que determina cómo se relacionan los valores de la función en distintos puntos del dominio.

La elección de la función de media forma parte de la especificación del prior. En general, un GP no requiere media nula; sin embargo, una elección habitual es tomar $m(x)=0.$ Esta hipótesis resulta razonable cuando no se dispone de información previa clara sobre una tendencia determinista global, ya que permite delegar la mayor parte de la estructura del modelo en la función de covarianza. Además, aparece de forma natural si se parte de un modelo lineal bayesiano centrado,

$$
f(x)=\phi(x)^\top w,
\qquad
w\sim\mathcal{N}(0,\Sigma_p),
$$

entonces la media inducida sobre la función satisface

$$
\mathbb{E}[f(x)] = \phi(x)^\top\mathbb{E}[w] = 0.
$$

Si existiera una tendencia conocida o relevante, podría incorporarse mediante una función de media no nula o mediante funciones base explícitas.

Una vez fijada la media, el segundo elemento necesario para definir el prior es la función de covarianza $k(x,x')$, que induce la matriz de covarianza $\mathbf{K}_{XX}$, definida en la ecuación anterior, que debe ser simétrica y semidefinida positiva para representar una covarianza válida. Intuitivamente, el kernel codifica las hipótesis a priori sobre la regularidad, suavidad y escala de variación de la función desconocida.

Además de definir media y covarianza, las distribuciones gaussianas asociadas a distintos subconjuntos finitos del dominio deben ser coherentes entre sí. Esta coherencia se obtiene gracias a una propiedad fundamental de la distribución gaussiana multivariante: la consistencia bajo marginalización.

**Proposición: marginalización de una gaussiana multivariante.**

Sea

$$
\begin{bmatrix}
\mathbf{f}_A\\
\mathbf{f}_B
\end{bmatrix}
\sim
\mathcal{N}\!\left(
\begin{bmatrix}
\boldsymbol{\mu}_A\\
\boldsymbol{\mu}_B
\end{bmatrix},
\begin{bmatrix}
\Sigma_{AA} & \Sigma_{AB}\\
\Sigma_{BA} & \Sigma_{BB}
\end{bmatrix}
\right),
$$

donde $\mathbf{f}_A$ y $\mathbf{f}_B$ son dos subvectores del vector gaussiano conjunto. Entonces, la distribución marginal de $\mathbf{f}_A$ viene dada por

$$
\mathbf{f}_A \sim \mathcal{N}(\boldsymbol{\mu}_A,\Sigma_{AA}).
$$

En esta partición por bloques, $\Sigma_{AA}$ es la submatriz de covarianza asociada exclusivamente a las componentes de $\mathbf{f}_A$, mientras que $\Sigma_{AB}$ y $\Sigma_{BA}$ recogen las covarianzas cruzadas entre $\mathbf{f}_A$ y $\mathbf{f}_B$.

Esta propiedad garantiza que, aunque el GP esté definido sobre un dominio potencialmente infinito, la inferencia puede realizarse de forma exacta utilizando únicamente un número finito de puntos. En particular, basta con considerar la distribución conjunta de los valores observados y, en su caso, de los puntos de predicción, ya que cualquier subconjunto de variables del proceso sigue teniendo una distribución gaussiana coherente con la especificación global.

La definición anterior introduce al GP directamente en el espacio de funciones (*function-space view*): se especifica un prior mediante su media y su función de covarianza, y la consistencia por marginalización garantiza la coherencia de todas las distribuciones finitas inducidas. Sin embargo, esta formulación todavía no muestra con claridad de dónde surge el kernel ni cómo debe interpretarse el aprendizaje del modelo. Para ello, resulta útil volver a una construcción más familiar: la regresión lineal bayesiana en un espacio de características, desde la cual el GP aparece al marginalizar los pesos.

#### Del espacio de pesos al espacio de funciones

Para comprender con mayor precisión el origen del kernel y el significado del aprendizaje en un GP, resulta útil partir de la regresión lineal bayesiana clásica y observar cómo se transita desde un modelo definido por parámetros explícitos (*weight-space view*) hacia una formulación basada únicamente en covarianzas entre datos (*function-space view*). Consideremos una función latente modelada como una combinación lineal de funciones base:

$$
f(\bm{x}) = \bm{\phi}(\bm{x})^\top \bm{w},
$$

donde $\bm{\phi}(\bm{x})\in\mathbb{R}^N$ representa el vector de características asociado a la entrada $\bm{x}$ y $\bm{w}\in\mathbb{R}^N$ es el vector de pesos. En lugar de asumir que existe un único valor verdadero y desconocido de $\bm{w}$ (enfoque frecuentista), el enfoque bayesiano introduce incertidumbre sobre dichos parámetros mediante un prior gaussiano: $\bm{w}\sim\mathcal{N}(\bm{0},\bm{\Sigma}_p).$ Las observaciones no se consideran iguales a la función latente, sino realizaciones ruidosas de la misma. Para cada dato observado se asume

$$
y_i = f(\bm{x}_i)+\varepsilon_i,
\qquad
\varepsilon_i\sim\mathcal{N}(0,\sigma_n^2),
$$

con ruido gaussiano independiente e idénticamente distribuido. Si definimos $\bm{y}=[y_1,\dots,y_n]^\top$ y la matriz de diseño en el espacio de características como

$$
\bm{\Phi}=
\begin{bmatrix}
\bm{\phi}(\bm{x}_1) & \cdots & \bm{\phi}(\bm{x}_n)
\end{bmatrix}
\in\mathbb{R}^{N\times n},
$$

entonces el modelo probabilístico para las observaciones puede escribirse como

$$
\bm{y}\mid X,\bm{w}\sim\mathcal{N}(\bm{\Phi}^\top\bm{w},\sigma_n^2\bm{I}).
$$

En la vista en espacio de pesos, aprender a partir de los datos equivale a actualizar de forma bayesiana la distribución sobre $\bm{w}$. Aplicando la regla de Bayes,

$$
p(\bm{w}\mid X,\bm{y}) \propto p(\bm{y}\mid X,\bm{w})\,p(\bm{w}),
$$

y dado que tanto la verosimilitud como el prior son gaussianos, la posterior vuelve a ser gaussiana. En concreto,

$$
p(\bm{w}\mid X,\bm{y})=
\mathcal{N}(\bar{\bm{w}},\bm{A}^{-1}),
\qquad
\bm{A}=\sigma_n^{-2}\bm{\Phi}\bm{\Phi}^\top+\bm{\Sigma}_p^{-1},
\qquad
\bar{\bm{w}}=\sigma_n^{-2}\bm{A}^{-1}\bm{\Phi}\bm{y}.
$$

Esta expresión resume qué significa aprender en la *weight-space view*: los datos no fijan un único vector de pesos, sino que actualizan la distribución de probabilidad sobre ellos. La media posterior $\bar{\bm{w}}$ representa la localización más plausible de los pesos tras observar los datos, mientras que $\bm{A}^{-1}$ cuantifica la incertidumbre residual sobre dichos parámetros.

A partir de esta posterior puede obtenerse ya una distribución predictiva para un nuevo punto $\bm{x}_*$. En efecto, si $f_* = f(\bm{x}_*)=\bm{\phi}(\bm{x}_*)^\top \bm{w},$ entonces la predictiva se obtiene marginalizando la incertidumbre sobre $\bm{w}$:

$$
p(f_*\mid \bm{x}_*,X,\bm{y})
=
\int p(f_*\mid \bm{x}_*,\bm{w})\,p(\bm{w}\mid X,\bm{y})\,d\bm{w}.
$$

Como $f_*$ es una transformación lineal de una variable gaussiana, esta distribución es también gaussiana:

$$
p(f_*\mid \bm{x}_*,X,\bm{y})
=
\mathcal{N}\!\Bigl(
\bm{\phi}(\bm{x}_*)^\top \bar{\bm{w}},
\;
\bm{\phi}(\bm{x}_*)^\top \bm{A}^{-1}\bm{\phi}(\bm{x}_*)
\Bigr).
$$

Esta formulación muestra de forma explícita cómo la incertidumbre sobre los pesos se propaga a la predicción. Sin embargo, también revela una limitación: la inferencia depende de $\bm{A}^{-1}$, una matriz de tamaño $N\times N$, donde $N$ es la dimensión del espacio de características. Si dicho espacio es muy grande o incluso infinito, esta representación deja de ser operativa. El paso decisivo consiste entonces en reescribir el modelo únicamente en términos de productos internos entre características.

De hecho, bajo el prior gaussiano sobre $\bm{w}$, la función latente evaluada en cualquier punto $\bm{x}$ es también una variable aleatoria gaussiana, al ser combinación lineal de variables gaussianas. Por tanto, para cualquier conjunto finito de entradas, la distribución conjunta de los valores de la función queda completamente determinada por su media y su covarianza. En particular,

$$
\mathbb{E}[f(\bm{x})]
=
\mathbb{E}[\bm{\phi}(\bm{x})^\top\bm{w}]
=
\bm{\phi}(\bm{x})^\top\mathbb{E}[\bm{w}]
=
0,
$$

y, para dos puntos arbitrarios $\bm{x}$ y $\bm{x}'$,

$$
\operatorname{cov}\bigl(f(\bm{x}),f(\bm{x}')\bigr)
=
\mathbb{E}\!\left[
\bm{\phi}(\bm{x})^\top \bm{w}\bm{w}^\top \bm{\phi}(\bm{x}')
\right]
=
\bm{\phi}(\bm{x})^\top \bm{\Sigma}_p \bm{\phi}(\bm{x}').
$$

La dependencia respecto a los pesos y a su covarianza puede agruparse en una única función definida sobre pares de entradas: $k(\bm{x},\bm{x}') = \bm{\phi}(\bm{x})^\top \bm{\Sigma}_p \bm{\phi}(\bm{x}').$ Así, la incertidumbre originalmente modelada sobre los parámetros $\bm{w}$ se traslada a la estructura de covarianza entre valores de la función.

Desde esta perspectiva, para cualquier conjunto finito $X=\{\bm{x}_1,\dots,\bm{x}_n\}$, el vector de valores latentes $\bm{f}_X = \bigl(f(\bm{x}_1),\dots,f(\bm{x}_n)\bigr)^\top$ puede obtenerse marginalizando los pesos:

$$
p(\bm{f}_X\mid X)
=
\int p(\bm{f}_X\mid X,\bm{w})\,p(\bm{w})\,d\bm{w}.
$$

Dado que $\bm{f}_X=\bm{\Phi}^\top \bm{w}$ es una transformación lineal de una variable gaussiana, su distribución marginal es también gaussiana, con

$$
\mathbb{E}[\bm{f}_X]=\bm{0},
\qquad
\operatorname{cov}(\bm{f}_X)=\bm{\Phi}^\top \bm{\Sigma}_p \bm{\Phi}=\bm{K},
\quad \text{con } K_{ij}=k(\bm{x}_i,\bm{x}_j).
$$

Por tanto,

$$
\bm{f}_X \sim \mathcal{N}(\bm{0}_X,\bm{K}_X).
$$

Es decir, al marginalizar los pesos se obtiene directamente una distribución gaussiana sobre los valores de la función en cualquier conjunto finito de entradas. Esta es precisamente la formulación del GP en el espacio de funciones: el modelo deja de expresarse en términos explícitos de $\bm{w}$ y pasa a quedar determinado únicamente por su función de media y por su función de covarianza.

###### El truco del núcleo

Para conectar el truco del núcleo con su interpretación funcional, resulta útil recurrir a la teoría de espacios de Hilbert con kernel reproductor. En esta parte se sigue la presentación de los métodos kernel de Bishop (Bishop 2006) y la interpretación de RKHS y regularización de Hastie, Tibshirani y Friedman (Hastie et al. 2009).

La expresión anterior $k(\bm{x},\bm{x}') = \bm{\phi}(\bm{x})^\top \bm{\Sigma}_p \bm{\phi}(\bm{x}')$ muestra que la covarianza entre dos evaluaciones de la función no depende de los pesos $\bm{w}$ de forma explícita, sino únicamente de cómo se relacionan las entradas $\bm{x}$ y $\bm{x}'$ en el espacio de características inducido por $\bm{\phi}$ y ponderado por $\bm{\Sigma}_p$. Este es el punto en el que aparece el truco del núcleo.

Dado que $\bm{\Sigma}_p$ es semidefinida positiva, puede escribirse en términos de una raíz matricial $\bm{\Sigma}_p^{1/2}$. Definiendo entonces un nuevo mapa de características $\bm{\psi}(\bm{x})=\bm{\Sigma}_p^{1/2}\bm{\phi}(\bm{x}),$ la covarianza anterior puede reescribirse como

$$
k(\bm{x},\bm{x}')
=
\bm{\psi}(\bm{x})^\top \bm{\psi}(\bm{x}').
$$

Por tanto, el kernel puede interpretarse como un producto interno en un espacio de características transformado. Esta observación tiene una consecuencia fundamental: si las expresiones del modelo dependen únicamente de productos internos entre entradas transformadas, entonces no es necesario construir ni manipular de forma explícita el vector de características $\bm{\phi}(\bm{x})$. Basta con disponer de una función $k(\bm{x},\bm{x}')$ que devuelva directamente ese producto interno.

En consecuencia, el modelo puede operar implícitamente en espacios de características de dimensión muy alta, e incluso infinita, sin abandonar una formulación computable. En el caso de los procesos gaussianos, esto significa que la inferencia y la predicción pueden expresarse únicamente en términos de la matriz kernel $\bm{K}$, cuyos elementos son $K_{ij}=k(\bm{x}_i,\bm{x}_j).$ La complejidad algebraica deja así de depender de la construcción explícita del espacio de características y se traslada al manejo de matrices de covarianza definidas sobre los datos observados.

Desde un punto de vista conceptual, el truco del kernel no es solo una simplificación computacional. También permite separar dos niveles de modelado: por un lado, la elección de una función kernel $k$, que codifica hipótesis a priori sobre suavidad, regularidad o estructura de la función desconocida; por otro, la representación concreta en términos de funciones base, que puede permanecer implícita. Esta reformulación justifica que, una vez introducido el GP desde la regresión lineal bayesiana, el análisis posterior pueda desarrollarse enteramente en términos de funciones de covarianza.

###### RKHS y teorema de representación.

La interpretación anterior puede precisarse mediante el concepto de espacio de Hilbert con kernel reproductor (*Reproducing Kernel Hilbert Space*, RKHS). Dado un kernel simétrico y semidefinido positivo $k$, existe un espacio de Hilbert $\mathcal{H}_k$ formado por funciones $f:\mathcal{X}\to\mathbb{R}$ en el que cada sección $k(\cdot,\bm{x})$ pertenece al espacio y satisface la propiedad reproductora:

$$
f(\bm{x})=\langle f,\;k(\cdot,\bm{x})\rangle_{\mathcal{H}_k}.
$$

Además, el propio kernel verifica

$$
k(\bm{x},\bm{x}')=
\langle k(\cdot,\bm{x}),\,k(\cdot,\bm{x}')\rangle_{\mathcal{H}_k}.
$$

Esta propiedad permite interpretar el kernel como un producto interno en un espacio de características posiblemente de dimensión muy alta o infinita. En el espacio original $\mathcal{X}$, la relación entre las entradas y la salida puede ser no lineal; sin embargo, mediante un levantamiento adecuado a un espacio funcional más rico, dicha relación puede tratarse mediante operaciones lineales en ese nuevo espacio. En el marco de RKHS, este levantamiento viene dado de forma canónica por $\Phi_k(\bm{x})=k(\cdot,\bm{x}),$ de modo que

$$
k(\bm{x},\bm{x}')=
\langle \Phi_k(\bm{x}),\Phi_k(\bm{x}')\rangle_{\mathcal{H}_k}.
$$

Por tanto, el kernel no solo actúa como una función de covarianza o como un producto interno implícito, sino también como el objeto que define la geometría del espacio funcional en el que vive la solución.

Esta interpretación refuerza la idea central del truco del núcleo: no es necesario construir de forma explícita el levantamiento $\Phi_k(\bm{x})$, que puede ser desconocido o incomputable en la práctica, ya que el kernel permite operar directamente con los productos internos relevantes. Así, el modelo puede capturar relaciones no lineales en el espacio original trabajando de forma implícita en un espacio de características más rico.

El resultado estructural clave en este contexto es el teorema de representación (Schölkopf et al. 2001). Para un problema de aprendizaje regularizado de la forma

$$
J[f]=Q\bigl(y_1,\dots,y_n,f(\bm{x}_1),\dots,f(\bm{x}_n)\bigr)
+\frac{\lambda}{2}\|f\|_{\mathcal{H}_k}^2,
$$

donde $Q$ mide el ajuste a los datos observados, todo minimizador $f\in \mathcal{H}_k$ puede escribirse como una combinación finita de kernels centrados en los datos de entrenamiento:

$$
f(\bm{x})=\sum_{i=1}^n \alpha_i\,k(\bm{x},\bm{x}_i).
$$

Este resultado es especialmente importante porque muestra que, aunque $\mathcal{H}_k$ pueda ser de dimensión infinita, la solución inducida por un conjunto finito de datos queda contenida en el subespacio generado por $\{k(\cdot,\bm{x}_1),\dots,k(\cdot,\bm{x}_n)\}.$

En el caso de los procesos gaussianos, esta misma estructura reaparece en la media posterior predictiva, que puede escribirse como una combinación lineal de evaluaciones del kernel en los puntos observados. Así, la visión probabilística del GP y la visión funcional basada en RKHS convergen en una misma idea: el aprendizaje depende de los datos únicamente a través de la geometría inducida por el kernel.

#### Funciones de covarianza (Kernels)

El kernel encapsula las suposiciones a priori sobre la suavidad, escala y estructura de correlación de la función latente. Para que defina una matriz de covarianza válida, debe ser simétrico y semidefinido positivo. En este trabajo se consideran los siguientes kernels:

- **Lineal (Dot Product).** Se define como

  $$
  k_{\mathrm{lin}}(\bm{x},\bm{x}')
      =
      \sigma_0^2 + \bm{x}^\top \bm{x}'.
  $$

  Este kernel corresponde a una regresión lineal bayesiana en el espacio original de entradas. El término $\bm{x}^\top\bm{x}'$ mide la similitud lineal entre dos puntos, mientras que $\sigma_0^2\geq 0$ actúa como un término de sesgo o intercepto, permitiendo desplazar la función respecto al origen. Es útil cuando se espera que la respuesta presente una tendencia global aproximadamente lineal.

- **Squared Exponential (SE/RBF).** Se expresa como

  $$
  k_{\mathrm{SE}}(\bm{x},\bm{x}')
      =
      \sigma_f^2
      \exp\left(
      -\frac{1}{2}(\bm{x}-\bm{x}')^\top
      \bm{M}
      (\bm{x}-\bm{x}')
      \right).
  $$

  El parámetro $\sigma_f^2>0$ es la varianza de la señal y controla la amplitud vertical de las funciones generadas por el prior. La matriz $\bm{M}$ controla la escala de variación respecto a las variables de entrada. En el caso isotrópico, $\bm{M}=\ell^{-2}\bm{I},$ donde $\ell>0$ es la longitud característica o *length-scale*.

  En el caso ARD (*Automatic Relevance Determination*), se utiliza habitualmente $\bm{M}=\mathrm{diag}(\ell_1^{-2},\dots,\ell_d^{-2}),$ permitiendo una escala distinta para cada dimensión. Valores pequeños de $\ell$ permiten variaciones rápidas de la función, mientras que valores grandes inducen funciones más suaves y con correlaciones de mayor alcance.

  El kernel SE/RBF induce funciones infinitamente diferenciables, por lo que representa una hipótesis de suavidad muy fuerte. Esta propiedad puede ser adecuada para respuestas muy regulares, pero también puede resultar excesiva en fenómenos físicos con cambios locales, irregularidades o menor grado de diferenciabilidad, donde el modelo puede suavizar en exceso la respuesta real.

- **Matérn.** El kernel Matérn generaliza al SE/RBF introduciendo un parámetro de suavidad $\nu>0$, que controla el grado de diferenciabilidad de las funciones generadas. Si $r=\|\bm{x}-\bm{x}'\|$ denota la distancia euclídea entre dos entradas y $\ell>0$ la longitud característica, el kernel Matérn se define como

  $$

      k_{\text{Mat\'ern}}(r)=
      \sigma_f^2
      \frac{2^{1-\nu}}{\Gamma(\nu)}
      \left(\frac{\sqrt{2\nu}\, r}{\ell}\right)^\nu
      K_\nu\left(\frac{\sqrt{2\nu}\, r}{\ell}\right).

  $$

  En esta expresión, $\Gamma(\nu)$ es la función Gamma, que generaliza el factorial a valores reales positivos, y $K_\nu(\cdot)$ es la función de Bessel modificada de segunda especie (NIST Digital Library of Mathematical Functions 2010). Al igual que en el kernel SE, $\sigma_f^2$ controla la amplitud de la función y $\ell$ determina la escala espacial de correlación. Estos parámetros no se calculan mediante una fórmula cerrada única, sino que se tratan como parámetros del kernel y se ajustan a partir de los datos, por ejemplo maximizando la log-verosimilitud marginal (LML).

  Los casos $\nu=3/2$ y $\nu=5/2$ son especialmente comunes porque producen expresiones cerradas sencillas y permiten controlar la suavidad de forma más realista que el SE. En particular, $\nu=3/2$ induce funciones menos suaves, mientras que $\nu=5/2$ permite funciones más regulares, pero sin imponer la suavidad infinita del kernel SE/RBF. Por ello, el kernel Matérn suele ser adecuado cuando se quiere modelar una función continua pero no necesariamente infinitamente diferenciable.

- **Ruido Blanco (White Kernel).**

  $$
  k_{\mathrm{White}}(\bm{x},\bm{x}')
      =
      \sigma_n^2 \delta_{\bm{x},\bm{x}'}.
  $$

  En esta expresión, $\sigma_n^2>0$ representa la varianza del ruido de observación y $\delta_{\bm{x},\bm{x}'}$ es la delta de Kronecker, que vale $1$ cuando las entradas coinciden y $0$ en caso contrario. Este kernel no modela una estructura suave de la función latente, sino variabilidad independiente asociada a las observaciones.

  Si el modelo observado se escribe como

  $$
  y_i = f(\bm{x}_i)+\varepsilon_i,
      \qquad
      \varepsilon_i\sim\mathcal{N}(0,\sigma_n^2),
  $$

  entonces la covarianza de las observaciones no es únicamente la covarianza de la función latente, sino

  $$
  \operatorname{cov}(y_i,y_j)
      =
      k_f(\bm{x}_i,\bm{x}_j)
      +
      \sigma_n^2\delta_{ij}.
  $$

  Por tanto, el kernel de ruido blanco se añade habitualmente a otro kernel, como RBF o Matérn, para representar que cada observación puede contener error de medida o ruido numérico independiente, sin introducir correlación espacial entre puntos distintos.

**Isotropía y Automatic Relevance Determination (ARD).**
Un kernel se dice *isotrópico* si la correlación depende solo de la distancia euclídea $\|\bm{x}-\bm{x}'\|$, asumiendo que la función varía de la misma forma en todas las direcciones. Sin embargo, en problemas de ingeniería, es habitual que algunas variables de entrada influyan mucho más que otras en la respuesta.

Para capturar esto, utilizamos **ARD** definiendo la matriz de escalas $\bm{M} = \mathrm{diag}(\ell_1^{-2}, \dots, \ell_d^{-2})$. Los parámetros $\ell_d$ son las **escalas de longitud características** (length-scales), que admiten una interpretación física directa:

- Si $\ell_d$ es pequeña, la función varía rápidamente en esa dimensión (variable muy relevante).

- Si $\ell_d$ es muy grande ($\ell_d \to \infty$), la función es casi constante en esa dirección (variable irrelevante), eliminando efectivamente la dependencia.

#### Predicción e incertidumbre

Dado un conjunto de observaciones ruidosas $\mathcal{D}_n=\{(\bm{x}_i,y_i)\}_{i=1}^n,$ se asume el modelo de observación

$$
y_i=f(\bm{x}_i)+\varepsilon_i,
\qquad
\varepsilon_i\sim\mathcal{N}(0,\sigma_n^2),
$$

con ruido gaussiano independiente. En esta subsección se considera la predicción del valor latente de la función en un nuevo punto de prueba $\bm{x}_*$, que denotamos por $f_* := f(\bm{x}_*).$

Por simplicidad, y de acuerdo con la justificación de media nula introducida en la Sección 2.2.1, se toma $m(\bm{x})=0$. Esta hipótesis puede relajarse incorporando una función de media no nula cuando exista información previa sobre una tendencia global de la función. Bajo esta convención, y usando la consistencia de las distribuciones gaussianas finitas descrita en la proposición anterior, la distribución conjunta de las observaciones $\bm{y}$ y del valor latente $f_*$ viene dada por

$$

\begin{bmatrix}
\bm{y} \\
f_*
\end{bmatrix}
\sim
\mathcal{N}\left(
\bm{0},
\begin{bmatrix}
\bm{K}+\sigma_n^2\bm{I} & \bm{k}_* \\
\bm{k}_*^\top & k_{**}
\end{bmatrix}
\right).


$$

En esta expresión, $\bm{K}=K(X,X)\in\mathbb{R}^{n\times n}$ es la matriz kernel entre los puntos de entrenamiento, con entradas $K_{ij}=k(\bm{x}_i,\bm{x}_j).$ El vector $\bm{k}_*=K(X,\bm{x}_*)\in\mathbb{R}^n$ recoge las covarianzas entre los puntos observados y el punto de prueba:

$$
\bm{k}_*
=
\begin{bmatrix}
k(\bm{x}_1,\bm{x}_*) \\
\vdots \\
k(\bm{x}_n,\bm{x}_*)
\end{bmatrix},
$$

mientras que $k_{**}=k(\bm{x}_*,\bm{x}_*)$ es la varianza a priori de la función latente en $\bm{x}_*$. El término $\sigma_n^2\bm{I}$ aparece únicamente en el bloque correspondiente a las observaciones $\bm{y}$, ya que estas incluyen ruido de medida, mientras que $f_*$ representa el valor latente no ruidoso de la función.

Aplicando las reglas de condicionamiento de una gaussiana multivariante (Rasmussen and Williams 2006), se obtiene la distribución posterior predictiva

$$
p(f_*\mid X,\bm{y},\bm{x}_*)
=
\mathcal{N}(\bar{f}_*,\mathbb{V}[f_*]),
$$

donde

$$
\begin{aligned}
\bar{f}_*
&=
\bm{k}_*^\top
(\bm{K}+\sigma_n^2\bm{I})^{-1}
\bm{y},

\\
\mathbb{V}[f_*]
&=
k_{**}
-
\bm{k}_*^\top
(\bm{K}+\sigma_n^2\bm{I})^{-1}
\bm{k}_*.

\end{aligned}
$$

La primera ecuación proporciona la media posterior predictiva, es decir, la estimación promedio del valor latente de la función en $\bm{x}_*$ bajo el modelo. La segunda cuantifica la incertidumbre posterior sobre ese valor latente. Es importante distinguir esta incertidumbre posterior sobre $f_*$ de la varianza del ruido $\sigma_n^2$. La primera refleja incertidumbre del modelo sobre la función latente y puede reducirse al incorporar observaciones informativas. Sin embargo, cuando las observaciones son muy ruidosas, dicha incertidumbre puede permanecer elevada incluso en zonas ya evaluadas, especialmente si no existen réplicas suficientes o si el ruido domina la señal. Si se quisiera predecir una nueva observación ruidosa $y_*$, y no solo el valor latente $f_*$, la varianza predictiva debería incluir además el término de ruido $\sigma_n^2$.

#### Entrenamiento: inferencia de parámetros

En los modelos paramétricos de estimación puntual, el entrenamiento suele consistir en seleccionar un valor concreto de los parámetros del modelo, por ejemplo mediante máxima verosimilitud (MLE) o mediante la minimización de una función de pérdida regularizada. Desde una perspectiva bayesiana, en cambio, los parámetros pueden tratarse como variables aleatorias: se define un prior sobre ellos y, tras observar los datos, se obtiene una distribución posterior. Si posteriormente se desea escoger un único valor representativo de esos parámetros, una opción posible es el estimador MAP. No obstante, en regresión con procesos gaussianos es habitual marginalizar los pesos o valores latentes y ajustar los parámetros mediante la maximización de la verosimilitud marginal.

En el caso de los procesos gaussianos, el entrenamiento adopta una forma distinta a la de los modelos paramétricos clásicos. Como se ha visto en la Sección 2.2.2.0.1, los pesos $\bm{w}$ de la representación en espacio de características quedan marginalizados y su efecto pasa a estar codificado en el kernel. Por tanto, entrenar un GP no consiste en fijar una familia de covarianza y estimar los hiperparámetros que la definen, junto con el nivel de ruido del modelo (Rasmussen and Williams 2006). En la práctica, una estrategia habitual consiste en maximizar la verosimilitud marginal de los datos,

$$
\hat{\bm{\theta}}
=
\arg\max_{\bm{\theta}}
\log p(\bm{y}\mid X,\bm{\theta}).
$$

Este procedimiento se conoce como maximización de la verosimilitud marginal o *type-II maximum likelihood*. Si además se introduce un prior explícito $p(\bm{\theta})$ sobre los parámetros, entonces el criterio correspondiente sería un MAP sobre $\bm{\theta}$:

$$
\hat{\bm{\theta}}_{\mathrm{MAP}}
=
\arg\max_{\bm{\theta}}
\left[
\log p(\bm{y}\mid X,\bm{\theta})
+
\log p(\bm{\theta})
\right].
$$

En este trabajo se presenta la formulación estándar basada en la LML, sin incluir un prior adicional sobre $\bm{\theta}$. La LML mide la probabilidad de observar los datos de entrenamiento bajo el modelo definido por los parámetros $\bm{\theta}$. En regresión con GP y ruido gaussiano, se obtiene marginalizando los valores latentes de la función:

$$
p(\bm{y}\mid X,\bm{\theta})
=
\int p(\bm{y}\mid \bm{f},X,\bm{\theta})p(\bm{f}\mid X,\bm{\theta})\,d\bm{f}.
$$

Como tanto el prior sobre $\bm{f}$ como la verosimilitud son gaussianos, esta integral puede resolverse de forma analítica y conduce a:

$$

\log p(\bm{y}\mid X,\bm{\theta})
=
\underbrace{-\frac{1}{2}\bm{y}^\top \bm{K}_y^{-1}\bm{y}}_{\text{Ajuste a los datos}}
\quad
\underbrace{-\frac{1}{2}\log|\bm{K}_y|}_{\text{Penalizacion por complejidad}}
\quad
\underbrace{-\frac{n}{2}\log 2\pi}_{\text{Constante}},


$$

donde $\bm{K}_y=\bm{K}_f+\sigma_n^2\bm{I}$ es la matriz de covarianza de las observaciones ruidosas.

Los tres términos de la expresión anterior tienen interpretaciones complementarias:

1.  **Ajuste a los datos.** El término $-\frac{1}{2}\bm{y}^\top \bm{K}_y^{-1}\bm{y}$ mide la compatibilidad entre las observaciones y la estructura de covarianza inducida por el kernel. Puede interpretarse como una distancia de Mahalanobis respecto a la media del modelo. Penaliza configuraciones de parámetros bajo las cuales los datos observados resultan poco probables.

2.  **Penalización por complejidad.** El término $-\frac{1}{2}\log|\bm{K}_y|$ depende de la matriz de covarianza y, por tanto, de los parámetros del kernel y del ruido. Actúa como una penalización de complejidad: modelos demasiado flexibles pueden asignar probabilidad a muchos patrones posibles, diluyendo la probabilidad asignada al conjunto concreto de datos observados. Por el contrario, modelos excesivamente rígidos pueden no explicar bien los datos.

3.  **Constante de normalización.** El término $-\frac{n}{2}\log 2\pi$ depende únicamente del número de observaciones y no afecta a la optimización de $\bm{\theta}$, aunque forma parte de la densidad gaussiana completa.

Esta descomposición está relacionada con el compromiso entre ajuste y complejidad del modelo, aunque no debe confundirse literalmente con la descomposición clásica sesgo-varianza-ruido. En GP, el ruido aparece explícitamente en $\bm{K}_y$ mediante $\sigma_n^2\bm{I}$, mientras que la flexibilidad del modelo queda controlada por los parámetros del kernel.

**Interpretación como regularización bayesiana**
La maximización de la LML incorpora de forma natural una forma de regularización bayesiana. El término de ajuste favorece parámetros que explican bien las observaciones, mientras que el término asociado al determinante penaliza modelos cuya flexibilidad reparte la masa de probabilidad entre demasiadas configuraciones posibles. Este efecto se relaciona con la llamada Navaja de Ockham bayesiana: entre modelos capaces de explicar los datos, la evidencia tiende a favorecer aquellos que lo hacen sin introducir complejidad innecesaria.

No obstante, esta regularización no debe interpretarse como una garantía absoluta contra el sobreajuste ni como una sustitución universal de la validación del modelo. La LML evalúa los datos bajo las hipótesis del GP y del kernel elegido. Si la familia de kernels está mal especificada, otros criterios, como la validación cruzada o la comparación con kernels alternativos, pueden ser necesarios para evaluar la capacidad predictiva del modelo. Además, la superficie de la LML puede presentar múltiples óptimos locales, por lo que el resultado puede depender de la inicialización del optimizador y de la parametrización del kernel.

Conviene matizar que la regularización no es exclusiva de un enfoque frecuentista. Por ejemplo, la regresión Ridge puede verse tanto como la minimización de una pérdida penalizada como el estimador MAP de una regresión lineal bayesiana con prior gaussiano sobre los pesos. En efecto, si $p(\bm{w})=\mathcal{N}(\bm{0},\sigma_w^2\bm{I})$ y el ruido de observación tiene varianza $\sigma_D^2$, el MAP conduce a una penalización

$$
\lambda\|\bm{w}\|^2,
\qquad
\lambda=\frac{\sigma_D^2}{\sigma_w^2}.
$$

Por tanto, el término de regularización puede interpretarse como una preferencia previa por modelos con pesos pequeños. La diferencia en el GP no es que desaparezca toda forma de regularización, sino que esta queda incorporada en la elección del prior funcional, es decir, en el kernel y sus parámetros.

### Diseño experimental y muestreo inicial

El rendimiento de la SBO depende en gran medida de la calidad del conjunto inicial de observaciones $\mathcal{D}_n$. Si los primeros puntos se concentran en una región reducida del dominio $\mathcal{X}\subset\mathbb{R}^d$, el modelo sustituto dispone de poca información sobre el resto del espacio de búsqueda. En un GP, esto afecta directamente a la media y a la varianza predictiva, que pueden quedar muy condicionadas por una cobertura inicial pobre. Por ello, antes de iniciar la fase secuencial de optimización, es habitual emplear estrategias de DoE orientadas a cubrir el dominio de forma representativa.

El muestreo aleatorio uniforme es la alternativa más simple, pero puede generar agrupaciones locales y regiones poco exploradas, especialmente cuando el presupuesto de evaluaciones es limitado. Para reducir este problema se utilizan diseños de tipo *space-filling*, cuyo objetivo es distribuir los puntos iniciales de manera más uniforme dentro del dominio (Forrester et al. 2008). Entre las alternativas más habituales se encuentran el *Latin Hypercube Sampling* (LHS) y las secuencias de Sobol.

El *Latin Hypercube Sampling* divide el rango de cada variable en $n$ intervalos de igual probabilidad y selecciona los puntos de forma que, en cada dimensión, exista exactamente una muestra por intervalo. Esto garantiza una buena estratificación marginal, aunque no necesariamente una cobertura geométrica óptima en el espacio multidimensional.

En este trabajo se emplean secuencias de Sobol, una familia de secuencias *quasi-Monte Carlo* de baja discrepancia (Sobol' 1967). A diferencia del muestreo pseudoaleatorio, estas secuencias son deterministas y están diseñadas para cubrir el hipercubo unidad de forma más uniforme. La discrepancia mide, de forma intuitiva, la diferencia entre la distribución empírica de los puntos generados y una distribución uniforme ideal; por tanto, una secuencia de baja discrepancia tiende a dejar menos huecos y menos agrupamientos que un muestreo aleatorio puro.

Esta propiedad resulta útil en la construcción inicial del modelo sustituto, ya que permite obtener una primera cobertura estable del dominio antes de aplicar criterios adaptativos de selección de nuevas evaluaciones. Además, al ser deterministas, las secuencias de Sobol facilitan la reproducibilidad del diseño inicial.

Respecto al tamaño del diseño inicial, una regla práctica habitual en optimización con modelos sustitutos es tomar un número de puntos proporcional a la dimensión del problema, por ejemplo $n=10d$, donde $d$ es la dimensión del vector de entrada (Jones et al. 1998). En problemas con evaluaciones muy costosas, este número puede reducirse para reservar más presupuesto a la fase secuencial de optimización.

### Selección de nuevas evaluaciones

Una vez construido un modelo sustituto inicial a partir del DoE, surge el reto de cómo utilizarlo para encontrar el óptimo global del problema original. En la literatura existen tres enfoques principales para abordar esta tarea (Jiang et al. 2020):

1.  **Heurísticas puras:** Algoritmos como los genéticos o evolutivos que evalúan directamente la función de caja negra. Requieren miles de evaluaciones de alta fidelidad, lo cual resulta inviable bajo presupuestos estrictos.

2.  **Metaheurísticas asistidas por sustitutos:** Utilizan el modelo sustituto para pre-evaluar y filtrar individuos dentro de una población evolutiva antes de ejecutar la simulación real. Aunque reducen el coste, su naturaleza poblacional sigue demandando un número considerable de muestras.

3.  **Optimización Basada en Modelos Sustitutos (SBO):** Es el enfoque más eficiente para problemas costosos. Se apoya en un modelo probabilístico para guiar la búsqueda secuencialmente, evaluando un único punto (o un pequeño lote) en cada iteración, maximizando la información obtenida.

#### Planteamiento: optimización secuencial con presupuesto limitado

El flujo de trabajo en SBO, y particularmente en BO, guarda una estrecha relación con paradigmas de aprendizaje automático como el *Active Learning*, donde el modelo selecciona de forma activa qué nuevas observaciones serían más informativas para mejorar su aprendizaje (Settles 2009). No obstante, en el contexto de la optimización basada en modelos sustitutos suele denominarse más específicamente proceso de *infill*, ya que el objetivo no es únicamente mejorar la precisión global del modelo, sino guiar la búsqueda del óptimo bajo un presupuesto limitado de evaluaciones $B$ optimizando una *función de adquisición*.

Esta función, mucho más barata de evaluar, decide de forma inteligente cuál es el candidato óptimo $x^*$ para la siguiente iteración. Una vez determinado, se evalúa en el sistema real (costoso), se añade a $\mathcal{D}_n$, y se reentrena el modelo sustituto (actualizando sus parámetros). Este ciclo se repite hasta agotar el presupuesto de evaluaciones $B$ o hasta satisfacer un criterio de convergencia.

#### Exploración vs explotación

El diseño de la función de adquisición define el comportamiento del algoritmo, debiendo equilibrar dos fuerzas contrapuestas (Jiang et al. 2020):

- **Explotación (Búsqueda local):** Priorizar las regiones del espacio de búsqueda donde el modelo sustituto predice que el valor de la función objetivo es muy bueno (mínimo, en el caso de minimización). Una estrategia puramente explotadora converge rápido, pero es altamente susceptible a quedar atrapada en mínimos locales.

- **Exploración (Búsqueda global):** Priorizar regiones con alta incertidumbre, es decir, donde el modelo tiene pocos datos y la varianza predictiva es alta. Una estrategia puramente exploradora actúa como un muestreo pseudoaleatorio, desperdiciando evaluaciones en zonas poco prometedoras.

#### Optimización Bayesiana y funciones de adquisición

En BO, la selección de nuevas evaluaciones se realiza mediante funciones de adquisición, que utilizan la distribución predictiva del modelo sustituto para decidir qué punto conviene evaluar a continuación (Jones et al. 1998). Aunque existen diversos criterios, como *Probability of Improvement* (PI), *EI* o *Lower Confidence Bound* (LCB), en este trabajo nos centraremos en el *EI*, uno de los criterios más utilizados en BO por su capacidad para combinar de forma explícita explotación e incertidumbre.

**Probability of Improvement (PI)**
El criterio PI fue uno de los primeros desarrollados. Su objetivo es maximizar la probabilidad de que un nuevo punto $\bm{x}$ mejore la mejor solución observada hasta el momento, $f_{\min}$. Si asumimos que la predicción en $\bm{x}$ sigue una distribución normal $\mathcal{N}(\mu(\bm{x}), \sigma^2(\bm{x}))$, la probabilidad de mejora es:

$$
PI(\bm{x})
=
P(Y(\bm{x}) < f_{\min})
=
\Phi\left(
\frac{f_{\min}-\mu(\bm{x})}{\sigma(\bm{x})}
\right),
$$

donde $\Phi$ es la función de distribución acumulada de la normal estándar.

La limitación principal de PI es que solo tiene en cuenta la probabilidad de mejorar el valor actual, pero no la magnitud esperada de dicha mejora. Por ello, puede asignar mucho valor a puntos con alta probabilidad de producir una mejora muy pequeña, sesgando la búsqueda hacia una explotación local excesiva.

**Expected Improvement (EI)**
El *Expected Improvement* (Mejora Esperada) resuelve esta limitación considerando no solo si un punto puede mejorar el mejor valor observado, sino cuánto se espera que lo mejore (Jones et al. 1998). Para un problema de minimización, definimos la función de mejora frente a la mejor observación actual $f_{\min}$ como:

$$
I(\bm{x})
=
\max(0, f_{\min}-Y(\bm{x})).
$$

De forma equivalente, puede escribirse la función a trozos:

$$
I(\bm{x})
=
\begin{cases}
f_{\min}-Y(\bm{x}), & \text{si } Y(\bm{x})<f_{\min},\\
0, & \text{si } Y(\bm{x})\geq f_{\min}.
\end{cases}
$$

Dado que $Y(\bm{x})$ es una variable aleatoria modelada por el GP, la mejora esperada se obtiene integrando la mejora sobre la densidad predictiva:

$$
\mathbb{E}[I(\bm{x})]
=
\int_{-\infty}^{f_{\min}}
(f_{\min}-y)
\frac{1}{\sigma(\bm{x})}
\phi\left(
\frac{y-\mu(\bm{x})}{\sigma(\bm{x})}
\right)
\,dy,
$$

donde $\phi(\cdot)$ es la función de densidad de la normal estándar. Haciendo el cambio de variable

$$
z=\frac{y-\mu(\bm{x})}{\sigma(\bm{x})},
\qquad
dy=\sigma(\bm{x})\,dz,
$$

el límite superior pasa a ser $Z=\frac{f_{\min}-\mu(\bm{x})}{\sigma(\bm{x})}.$ Por tanto,

$$
\mathbb{E}[I(\bm{x})]
=
\int_{-\infty}^{Z}
\left(f_{\min}-\mu(\bm{x})-\sigma(\bm{x})z\right)\phi(z)\,dz.
$$

Separando los términos:

$$
\mathbb{E}[I(\bm{x})]
=
(f_{\min}-\mu(\bm{x}))\int_{-\infty}^{Z}\phi(z)\,dz
-
\sigma(\bm{x})\int_{-\infty}^{Z}z\phi(z)\,dz.
$$

Como

$$
\int_{-\infty}^{Z}\phi(z)\,dz=\Phi(Z),
\qquad
\int_{-\infty}^{Z}z\phi(z)\,dz=-\phi(Z),
$$

se obtiene la formulación cerrada:

$$


\mathbb{E}[I(\bm{x})] =
\begin{cases}
(f_{\min} - \mu(\bm{x})) \Phi(Z) + \sigma(\bm{x}) \phi(Z),
& \text{si } \sigma(\bm{x}) > 0, \\[0.2em]
0,
& \text{si } \sigma(\bm{x}) = 0,
\end{cases}
\qquad
Z = \frac{f_{\min} - \mu(\bm{x})}{\sigma(\bm{x})}.

$$

La expresión cerrada de EI combina de forma explícita las dos estrategias de búsqueda:

- El primer término, $(f_{\min} - \mu(\bm{x})) \Phi(Z)$, domina cuando el modelo predice un valor bajo de $\mu(\bm{x})$, es decir, un mejor resultado esperado en un problema de minimización. Este término impulsa la **explotación** de zonas prometedoras.

- El segundo término, $\sigma(\bm{x}) \phi(Z)$, domina cuando la incertidumbre $\sigma(\bm{x})$ es grande, favoreciendo la **exploración** de regiones donde el modelo todavía tiene poca información.

Matemáticamente, esta síntesis de estrategias se demuestra de forma analítica calculando las derivadas parciales de la función EI respecto a los momentos predictivos del modelo. Por un lado, la derivada respecto a la media predictiva es:

$$
\frac{\partial E[I(\bm{x})]}{\partial \mu(\bm{x})} = -\Phi(Z)
$$

Dado que la función de distribución acumulada $\Phi(Z)$ toma valores estrictamente positivos, esta derivada es siempre negativa. Esto demuestra que la función EI es **monótonamente decreciente** respecto a $\mu(\bm{x})$. En la práctica, significa que cuanto menor sea el valor predicho por el modelo para una región, y por tanto mejor en un problema de minimización, mayor será su expectativa de mejora, lo que empuja al algoritmo a *explotar* las zonas prometedoras.

Por otro lado, la derivada respecto a la incertidumbre predictiva resulta en:

$$
\frac{\partial E[I(\bm{x})]}{\partial \sigma(\bm{x})} = \phi(Z).
$$

Puesto que la función de densidad de la normal estándar $\phi(Z)$ es siempre mayor que cero, esta derivada es estrictamente positiva. Por consiguiente, EI es **monótonamente creciente** respecto a $\sigma(\bm{x})$. Su significado práctico es que, ante dos localizaciones con la misma predicción media, el algoritmo siempre asignará un mayor valor de adquisición a la que posea mayor incertidumbre, formalizando así la *exploración*. Un mayor desconocimiento amplía la varianza de la campana de Gauss predictiva, aumentando el área de probabilidad que cae por debajo del umbral $f_{\min}$.

En los puntos de diseño ya observados del conjunto de datos (asumiendo observaciones sin ruido), la certidumbre del GP es absoluta, es decir, $\sigma(\bm{x}) = 0$. Al evaluar este límite, la mejora esperada se anula ($E[I(\bm{x})] = 0$). Esta propiedad es crítica, ya que previene de forma inherente el estancamiento del algoritmo, garantizando que no se desperdicie el valioso presupuesto computacional evaluando exactamente el mismo punto dos veces.

![(a) Media predictiva del GP condicionada a las observaciones disponibles. (b) Valores de EI asociados al modelo del panel (a). (c) Media predictiva del GP tras incorporar una nueva observación mediante *infill*. (d) Valores de EI actualizados para la siguiente iteración.](/assets/articles/surrogate_models/app_forrester_ei_decision_synthesis_step5.png)

En conjunto, estas propiedades hacen que EI sea adecuada como criterio de selección dentro del proceso de *infill*: permite proponer nuevas evaluaciones combinando la búsqueda en regiones prometedoras con la exploración de zonas donde el modelo todavía presenta incertidumbre. Este mecanismo secuencial será utilizado en la parte experimental del trabajo y se resume esquemáticamente en la Figura 2.2.

Una vez agotado el presupuesto máximo de evaluaciones (B), o alcanzado un criterio de convergencia predefinido, el proceso de enriquecimiento se da por terminado.

### Métricas de evaluación

Para validar el rendimiento de los modelos sustitutos, no es suficiente evaluar la precisión de sus predicciones puntuales. En el caso de modelos probabilísticos como los Procesos Gaussianos, resulta fundamental cuantificar la calidad de su incertidumbre predictiva. Por ello, las métricas de evaluación se dividen en dos categorías: métricas predictivas clásicas y métricas probabilísticas.

#### Métricas predictivas (MAE, RMSE, R$^2$)

Estas métricas evalúan la divergencia entre la observación real $y_i$ y la media predictiva del modelo $\mu_i$ (o $\hat{y}_i$) sobre un conjunto de tamaño $N$.

- **MAE (Mean Absolute Error):** Viene dada por $\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \mu_i|$. Representa la magnitud promedio de los errores en las predicciones, sin considerar su dirección. Crece de forma lineal, lo que lo hace más robusto frente a valores atípicos que el RMSE.

- **RMSE (Root Mean Squared Error):** Se define como $\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \mu_i)^2}$. Al elevar las diferencias al cuadrado, penaliza de forma severa los errores grandes (valores atípicos o predicciones desastrosas) en comparación con los errores pequeños.

- **Coeficiente de determinación ($R^2$):** Se expresa como $R^2 = 1 - \frac{\sum_{i=1}^{N} (y_i - \mu_i)^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}$. Mide la proporción de la varianza total de la variable dependiente que es explicada por el modelo.

#### Métricas probabilísticas (NLL, NLPD)

Mientras que métricas como RMSE, MAE o $R^2$ evalúan únicamente la calidad de la predicción puntual, las métricas probabilísticas tienen en cuenta la distribución predictiva completa del modelo. En el caso de un GP, esto significa evaluar no solo si la media predictiva $\mu_i$ se aproxima al valor observado $y_i$, sino también si la incertidumbre asignada por el modelo es coherente con los errores que comete.

Esta distinción es importante en modelos sustitutos probabilísticos, ya que una buena media predictiva no garantiza una buena cuantificación de incertidumbre. Un modelo puede acertar de media, pero ser demasiado confiado; o puede cubrir casi todos los puntos reales mediante intervalos excesivamente amplios y, por tanto, poco informativos. Por ello, además de las métricas predictivas clásicas, se consideran métricas basadas en densidad predictiva y cobertura de intervalos (Rasmussen and Williams 2006; Gneiting and Raftery 2007; Kuleshov et al. 2018).

###### Negative Log-Likelihood y Negative Log Predictive Density.

Conviene distinguir dos cantidades relacionadas, pero usadas en contextos distintos. Durante el entrenamiento del GP, es habitual minimizar el negativo de la LML, también llamado *negative log marginal likelihood*: $-\log p(\bm{y}\mid X,\bm{\theta}),$ que se corresponde con el signo contrario de la expresión de la LML definida anteriormente. Esta cantidad se utiliza para ajustar los parámetros $\bm{\theta}$ del kernel y del ruido.

En cambio, para evaluar el rendimiento probabilístico sobre un conjunto de prueba, resulta más apropiado usar la *Negative Log Predictive Density* (NLPD). Esta métrica evalúa, punto a punto, la probabilidad que el modelo entrenado asigna al valor realmente observado. Si para cada punto de test $\bm{x}_i$ la distribución predictiva es $p(y_i\mid \bm{x}_i,\mathcal{D}_n)=\mathcal{N}(\mu_i,s_i^2),$ entonces la NLPD media sobre un conjunto de prueba de $N$ muestras se define como

$$

\mathrm{NLPD}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left[
\frac{1}{2}\log(2\pi s_i^2)
+
\frac{(y_i-\mu_i)^2}{2s_i^2}
\right].


$$

Aquí $s_i^2$ representa la varianza de la distribución predictiva usada para evaluar $y_i$. Si se evalúan observaciones ruidosas, esta varianza debe incluir tanto la incertidumbre sobre la función latente como la varianza del ruido de observación. Si, por el contrario, se evalúa directamente una función latente sin ruido, se utilizaría únicamente la varianza posterior sobre $f(\bm{x}_i)$. Valores más bajos de NLPD indican una mejor distribución predictiva. La expresión anterior penaliza notablemente dos situaciones indeseables:

1.  **Sobreconfianza (*overconfidence*).** Si el modelo asigna una varianza $s_i^2$ muy pequeña pero la media $\mu_i$ se aleja del valor real $y_i$, el término cuadrático $\frac{(y_i-\mu_i)^2}{2s_i^2}$ crece con rapidez. Esto penaliza modelos que producen intervalos estrechos pero fallan con errores grandes.

2.  **Subconfianza (*underconfidence*).** Si el modelo asigna varianzas muy grandes de forma sistemática, puede cubrir muchos valores reales, pero el término $\frac{1}{2}\log(2\pi s_i^2)$ aumenta. Esto penaliza distribuciones predictivas excesivamente anchas y poco informativas.

Por tanto, la NLPD no mide únicamente el error de la media, sino el equilibrio entre precisión y calibración de la incertidumbre.

## Experimentación y resultados

Este capítulo traslada el marco teórico anterior al diseño experimental del trabajo. El objetivo es evaluar el uso de GPs como modelos sustitutos probabilísticos para aproximar una relación entre configuraciones de entrada y una respuesta de interés, especialmente en escenarios donde las evaluaciones son limitadas o costosas.

La experimentación se organiza en dos escenarios. En primer lugar, se utilizan funciones benchmark sintéticas, donde la función objetivo es conocida y puede evaluarse en nuevos puntos. Este escenario permite estudiar el ciclo completo de optimización basada en modelos sustitutos: diseño inicial, ajuste del GP, selección secuencial de puntos mediante un criterio de *infill* y actualización del conjunto observado. En segundo lugar, se plantea un caso real basado en datos experimentales de dietas para insectos. En este caso, las observaciones proceden de ensayos ya realizados y se encuentran agrupadas por dieta, por lo que el objetivo principal no es ejecutar nuevas evaluaciones físicas, sino validar si el modelo generaliza a dietas no vistas.

Por tanto, ambos escenarios comparten el mismo marco general de modelado sustituto, pero no el mismo protocolo experimental: los benchmarks permiten analizar el proceso secuencial de búsqueda en un entorno controlado, mientras que el caso real evalúa la viabilidad del enfoque antes de plantear una fase prospectiva de recomendación de nuevas configuraciones.

### Marco común de modelado sustituto

En ambos escenarios se parte de un conjunto finito de observaciones

$$
\mathcal{D}_n = \{(\bm{x}_i, y_i)\}_{i=1}^{n},
$$

donde $\bm{x}_i \in \mathcal{X}$ representa un conjunto de datos de entrada e $y_i$ la respuesta observada. A partir de estos datos se entrena un modelo sustituto cuyo objetivo es aproximar la relación entre entradas y salidas, proporcionando predicciones para entradas no incluidas en el conjunto de observaciones y, en el caso de los procesos gaussianos, una estimación de incertidumbre.

#### Datos, modelos y preprocesamiento

Las entradas se organizan en una matriz de diseño $X \in \mathbb{R}^{n \times d}$, donde $n$ es el número de observaciones y $d$ la dimensión del espacio de entrada. Las respuestas se recogen en un vector $\bm{y} \in \mathbb{R}^{n}$. En los benchmarks, cada observación corresponde a una evaluación controlada de una función sintética. En el caso real, cada observación corresponde a una medición experimental asociada a una dieta, por lo que las particiones de entrenamiento y evaluación deben respetar dicha agrupación para evitar fuga de información entre réplicas de una misma dieta.

El diseño experimental compara una línea base constante y varias configuraciones de GP. Antes del ajuste, las variables de entrada se estandarizan mediante *z-score*: cada columna se centra con la media del conjunto de entrenamiento y se divide por su desviación típica. Esta transformación se estima de nuevo dentro de cada partición o iteración experimental y después se aplica al conjunto de evaluación correspondiente, evitando así utilizar información de test durante el ajuste. En el caso real, las transformaciones adicionales necesarias para representar las dietas se describen en la Sección 3.3.

El diseño compara una línea base constante y varias familias de GP. El Cuadro 3.1 resume qué modelos se usan en cada experimento y qué papel cumplen dentro del análisis. La línea base se interpreta como referencia mínima; los GP constituyen los modelos sustitutos probabilísticos analizados. Esta descripción define el protocolo experimental; las clases y paquetes concretos empleados para ejecutarlo se describen por separado.

| **Familia** | **Uso** | **Papel en el análisis** |
|:---|:---|:---|
| Dummy | Benchmarks y caso real | Predicción constante calculada a partir de las respuestas del conjunto de entrenamiento. Sirve como referencia mínima frente a modelos que sí utilizan las variables de entrada. |
| GP lineal | Benchmarks y caso real | GP con kernel lineal y término de ruido blanco. Representa una hipótesis de relación aproximadamente lineal entre entradas y respuesta. |
| GP RBF | Benchmarks y caso real | GP con kernel RBF y término de ruido blanco. Introduce una hipótesis suave y no lineal con una escala de longitud común. |
| GP Matérn $3/2$ | Benchmarks y caso real | GP con kernel Matérn $3/2$ y término de ruido blanco. Permite funciones menos suaves que RBF. |
| GP Matérn $5/2$ | Benchmarks y caso real | GP con kernel Matérn $5/2$ y término de ruido blanco. Supone una respuesta más suave que Matérn $3/2$, pero menos restrictiva que RBF. |
| GP compuesto | Caso real | GP con kernel aditivo lineal + Matérn $5/2$ + ruido blanco. Se usa para combinar una tendencia global con una componente no lineal suave en el conjunto experimental de dietas. |
| Variante ARD | Según protocolo | Variante con una escala de longitud por variable. En benchmarks se aplica a RBF y Matérn $5/2$ cuando la dimensión lo permite; en el caso real se prueba solo sobre el mejor kernel previo de cada combinación de objetivo y representación de entrada. |

*Familias de modelos consideradas por experimento.*

#### Predicción probabilística e incertidumbre

Para una nueva configuración $\bm{x}$, el GP devuelve una media predictiva $\mu(\bm{x})$ y una desviación típica $\sigma(\bm{x})$. La media se utiliza como predicción puntual y la desviación típica como medida de incertidumbre. En los benchmarks, esta incertidumbre se usa de forma activa dentro de EI para seleccionar nuevas evaluaciones. En el caso real, se emplea como indicador de fiabilidad de las predicciones, sin ejecutar todavía una fase física de *infill*.

#### Entorno de implementación

Los experimentos se implementaron en Python, usando `pandas` y `numpy` para la manipulación de datos, `scikit-learn` para el ajuste y validación de modelos, `scipy` y `scikit-optimize` para utilidades numéricas y optimización, y `matplotlib`, `seaborn` y `plotly` para la generación de figuras.

En la implementación, los modelos se encapsularon en las clases `DummySurrogateRegressor` y `GPSurrogateRegressor`, esta última construida sobre un `Pipeline` con `StandardScaler` y `GaussianProcessRegressor`.

Las ejecuciones se realizaron en un equipo con Windows 10.0.26200 de 64 bits, procesador Intel(R) Core(TM) Ultra 7 155H, 22 hilos lógicos y 32 GB de RAM.

### Experimento 1: benchmarks sintéticos

Como se ha adelantado, el primer experimento utiliza funciones benchmark sintéticas, en las que la función objetivo puede evaluarse en nuevos puntos del dominio. Esto permite estudiar el ciclo completo de optimización basada en modelos sustitutos: diseño inicial, ajuste del GP, selección de nuevas evaluaciones mediante EI y actualización secuencial del conjunto observado. La lectura se centra en la evolución del mejor valor encontrado, la capacidad predictiva y la incertidumbre conforme se incorporan nuevas observaciones.

#### Benchmarks seleccionados

Se consideran cuatro benchmarks en optimización global y modelado sustituto (Surjanovic and Bingham 2013; Forrester et al. 2008; Dixon and Szegö 1978; Morris et al. 1993), con una progresión razonable de dificultad, desde problemas de baja dimensión e interpretación visual hasta funciones de mayor dimensión que deben analizarse principalmente mediante métricas. Los dominios de búsqueda se definen en la implementación y se utilizan de forma común para el muestreo inicial, el conjunto de test y la optimización del criterio de adquisición. Toda esta información se resume en el Cuadro 3.2.

| **Benchmark** | **Dim.** | **Dominio** | **Papel en el experimento** |
|:---|:--:|:---|:---|
| Forrester | 1 | $x \in [0,1]$ | Caso unidimensional para visualizar el ajuste del GP, la incertidumbre y el comportamiento de EI. |
| Branin | 2 | $x_1 \in [-5,10]$, $x_2 \in [0,15]$ | Función bidimensional multimodal, útil para analizar la selección de puntos en un dominio representable. |
| Hartmann6 | 6 | $x_i \in [0,1]$, $i=1,\dots,6$ | Benchmark de mayor dimensión y multimodalidad, evaluado principalmente mediante métricas. |
| Borehole | 8 | Dominio heterogéneo de ocho variables | Problema de inspiración ingenieril para estudiar el comportamiento del método en dimensión superior. |

*Benchmarks seleccionados para el experimento sintético.*

#### Diseño inicial, presupuesto y configuraciones GP

Para cada benchmark se genera un diseño inicial dentro del dominio de búsqueda, que representa la información disponible antes de iniciar el proceso secuencial de *infill*. Se comparan dos estrategias de muestreo inicial, Sobol y muestreo aleatorio uniforme, y tres tamaños iniciales:

$$
n_{\text{train}} \in \{1,\; 1 \times d,\; 4 \times d\},
$$

donde $d$ es la dimensión del benchmark. El caso $n_{\text{train}}=1$ permite observar el comportamiento del sistema desde una situación extrema con una única observación inicial, mientras que $1 \times d$ y $4 \times d$ representan diseños escalados con la dimensión del problema.

A partir del diseño inicial, el proceso de *infill* añade $5 \times d$ nuevos puntos, sujeto a un máximo de 50 observaciones durante la trayectoria. También se consideran tres condiciones de ruido: evaluación sin ruido y ruido gaussiano con niveles $0.5$ y $1.0$. La configuración general se resume en el Cuadro 3.3, que fija las condiciones comunes empleadas después para agregar y comparar los resultados.

El dominio detallado de Borehole, el flujo completo del proceso de *infill* y la configuración específica de semillas se incluyen como material complementario en el Anexo 5. En particular, el dominio de Borehole se recoge en el Cuadro 5.1, el esquema del proceso secuencial en la Figura 5.1 y las semillas empleadas en el Cuadro 5.2.

| **Elemento** | **Configuración** |
|:---|:---|
| Benchmarks | Forrester, Branin, Hartmann6 y Borehole. |
| Muestreo inicial | Sobol y aleatorio uniforme. |
| Tamaño inicial | $n_{\text{train}} = 1$, $1 \times d$ y $4 \times d$. |
| Presupuesto de *infill* | $5 \times d$ puntos añadidos al diseño inicial, con un máximo de 50 observaciones durante el proceso. |
| Ruido | Sin ruido y ruido gaussiano con niveles $0.5$ y $1.0$. |
| Modelos GP | Familias lineal, RBF, Matérn $3/2$ y Matérn $5/2$, descritas en el Cuadro 3.1; en dimensión mayor que uno se evalúan además variantes ARD para RBF y Matérn $5/2$. |
| Conjunto de test | Conjunto independiente generado dentro del dominio de cada benchmark. |
| Semillas | Se utilizan semillas fijas para garantizar la reproducibilidad del experimento. La configuración detallada se recoge en el Anexo 5.3. |

*Configuración general del experimento de benchmarks.*

En cada iteración, el GP se ajusta con el conjunto de observaciones disponible $\mathcal{D}_n$. A continuación, se calcula EI sobre el dominio de búsqueda y se selecciona el siguiente punto como

$$
\bm{x}_{\mathrm{next}}
    =
    \operatorname*{arg\,max}_{\bm{x}\in\mathcal{X}} (EI(\bm{x})).
$$

Aunque la función objetivo se plantea como un problema de minimización, EI se maximiza porque representa la mejora esperada respecto al mejor valor observado hasta el momento. En la implementación, la maximización de EI se aproxima mediante Differential Evolution; los detalles del procedimiento de optimización de la función de adquisición se recogen en el Anexo 5.2.

La configuración computacional completa de este experimento, incluyendo el tamaño del conjunto de test, el parámetro de exploración de EI, el presupuesto del optimizador y las semillas empleadas, se documenta en el Anexo 8.1.

#### Métricas del experimento benchmark

Las métricas predictivas y probabilísticas empleadas para evaluar el GP, como MAE, RMSE, $R^2$, NLPD y cobertura del intervalo predictivo, siguen las definiciones introducidas en la Sección 2.5. En este experimento se utilizan para analizar si el proceso de *infill* mejora la capacidad predictiva del modelo, pero la lectura principal no es solo predictiva: también interesa comprobar si EI permite encontrar valores menores de la función objetivo.

Desde el punto de vista de la optimización, la magnitud principal es la mejora relativa del mínimo encontrado durante el proceso de *infill*. El mejor valor disponible hasta una iteración $t$ se denomina *incumbent*. Cuando el óptimo global $f^\star$ es conocido, esta mejora se expresa como reducción relativa del *gap* al óptimo. Para ello, se define $g_t = f_{\mathrm{best},t}^{\mathrm{clean}} - f^\star,$ donde $f_{\mathrm{best},t}^{\mathrm{clean}}$ es el mejor valor limpio encontrado hasta la iteración $t$. A partir de esta distancia se calcula

$$
I_t = \frac{g_0 - g_t}{g_0},
$$

donde $g_0$ corresponde al *gap* inicial del diseño de partida. Valores próximos a cero indican poca mejora respecto al diseño inicial, mientras que valores próximos a uno indican una reducción casi completa de la distancia inicial al óptimo.

En Borehole, al no disponer de un óptimo global definido en la implementación, no se calcula un *gap* al óptimo. En este caso, la misma lectura se aproxima mediante la reducción relativa del mejor valor limpio encontrado respecto al diseño inicial. Por tanto, en el texto se usa "mejora relativa del mínimo encontrado" como denominación general, y "reducción del *gap*" únicamente cuando existe un óptimo de referencia.

Además de la mejora final, se considera la mejora acumulada durante la trayectoria. Esta magnitud resume la evolución de la mejora relativa a lo largo del proceso de *infill* y permite distinguir entre configuraciones que encuentran buenas soluciones pronto y configuraciones que solo mejoran al final del presupuesto disponible. Por ello, la mejora final mide el resultado alcanzado al terminar el proceso, mientras que la mejora acumulada aporta una lectura temporal de la búsqueda.

La capacidad predictiva se analiza mediante el MAE relativo respecto al estado inicial y mediante la comparación frente al modelo Dummy. El MAE relativo permite comprobar si las observaciones añadidas reducen el error del GP sobre un conjunto de test independiente.

Las métricas probabilísticas, en particular NLPD y cobertura del 95 %, se interpretan como diagnóstico complementario de incertidumbre. Valores bajos de NLPD y coberturas próximas al nivel nominal indican distribuciones predictivas más coherentes, aunque no garantizan por sí solas una mejor búsqueda del óptimo.

#### Resultados del experimento benchmark

Los resultados se organizan en tres niveles. Primero se analiza si el proceso de *infill* basado en EI mejora el mejor valor encontrado de la función objetivo. Después se estudia si las nuevas observaciones también mejoran la capacidad predictiva del GP. Finalmente, se resumen los factores experimentales, como el tamaño inicial, el ruido, la dimensionalidad, el kernel y el uso de ARD, que más condicionan el rendimiento observado.

##### Evolución del mínimo encontrado durante el proceso de infill

La métrica principal desde el punto de vista de la optimización es la mejora relativa del mínimo encontrado, definida en la Sección 3.2.3. En este análisis, *step*=0 representa el mejor punto disponible en el diseño inicial, antes de añadir nuevas evaluaciones. A partir de ahí, cada iteración incorpora el punto seleccionado por EI y actualiza el mejor valor encontrado.

El Cuadro 3.4 resume el mejor modelo de cada benchmark según la mejora final del mínimo encontrado. También se incluye la mejora acumulada, que permite distinguir si la mejora aparece pronto durante la trayectoria o si se concentra al final del presupuesto de *infill*.

La mejora acumulada se calcula a partir del área bajo la trayectoria normalizada de mejora relativa. Por tanto, no solo mide el resultado final, sino también la rapidez con la que aparece la mejora. Un ratio cercano a uno indica que la mayor parte de la mejora se obtiene pronto, mientras que un ratio menor sugiere una mejora más tardía.

Los resultados muestran que EI consigue mejoras positivas en los cuatro benchmarks, aunque con distinta intensidad. Branin presenta la mejora final más clara, con una reducción media del *gap* cercana a $0.90$. Forrester también muestra una mejora elevada, mientras que en Borehole esta lectura corresponde a la reducción relativa del mejor valor limpio respecto al diseño inicial, al no disponer de un óptimo de referencia. Hartmann6 resulta el escenario más exigente: la mejora existe, pero es más moderada y requiere un diseño inicial más informativo, coherente con su mayor dimensionalidad.

| Benchmark | Mejor modelo | Mejora final | Mejora acum. | Ratio | $n_{\text{train}}$ | Lectura |
|:---|:---|:--:|:--:|:--:|:--:|:---|
| Borehole | $GP\_Linear$ | $0.720$ | $0.694$ | $0.963$ | $1$ | Mejora temprana y fuerte. |
| Branin | $GP\_Matern52\_ARD$ | $0.898$ | $0.596$ | $0.663$ | $1$ | Mejora final muy alta. |
| Forrester | $GP\_Matern52$ | $0.737$ | $0.390$ | $0.530$ | $4$ | Mejora fuerte, pero más tardía. |
| Hartmann6 | $GP\_RBF$ | $0.434$ | $0.304$ | $0.701$ | $24$ | Mejora moderada en mayor dimensión. |

*Resumen del mejor modelo de optimización por benchmark. La mejora final corresponde a la mejora relativa del mínimo encontrado al terminar el proceso de infill; en los benchmarks con óptimo conocido equivale a la reducción relativa del gap al óptimo. La mejora acumulada resume la evolución durante la trayectoria.*

![Evolución de la mejora relativa del mínimo encontrado para el mejor modelo de cada benchmark. El valor *step*=0 corresponde al mejor punto del diseño inicial, antes de añadir puntos mediante EI. Las curvas muestran la media y las bandas sombreadas la desviación típica.](/assets/articles/surrogate_models/evolution_incumbent_best_model_by_benchmark.png)

En Borehole, la mejora final y la mejora acumulada son muy próximas, lo que indica que el proceso encuentra regiones favorables desde fases tempranas. En Branin, la mejora final es muy alta, aunque parte del progreso aparece a lo largo de la trayectoria. Forrester presenta una mejora final fuerte, pero menos inmediata. En Hartmann6, el progreso es más gradual, lo que confirma que el presupuesto disponible resulta más restrictivo en problemas de mayor dimensión.

La Figura 3.1 confirma esta lectura temporal. En Borehole y Branin, gran parte de la mejora aparece durante las primeras iteraciones y después las curvas tienden a estabilizarse. En Forrester, la trayectoria depende más del diseño inicial y la mejora aparece de forma menos inmediata. En Hartmann6, el avance es más progresivo, lo que refuerza la importancia de disponer de más información inicial en problemas de mayor dimensión. Como material complementario, la Figura 5.2 muestra la mejora relativa final del mínimo encontrado agregada por benchmark.

En conjunto, el análisis de la mejora relativa del mínimo encontrado confirma que EI cumple su función principal dentro del ciclo de optimización: seleccionar nuevas evaluaciones que mejoran el mejor valor disponible respecto al diseño inicial.

##### Evolución de la capacidad predictiva del GP

Además del éxito en optimización, se analiza si el proceso de *infill* mejora la capacidad predictiva del GP. Para ello se utiliza principalmente el MAE relativo respecto al estado inicial. Esta lectura debe separarse de la mejora relativa del mínimo encontrado: una configuración puede predecir mejor en promedio sobre el conjunto de test sin ser necesariamente la más útil para encontrar mínimos.

El Cuadro 3.5 resume el mejor modelo predictivo de cada benchmark y su comparación frente a Dummy. Dummy no participa en el proceso de *infill*, pero sirve como referencia para comprobar si el GP aporta valor frente a una predicción trivial.

| Benchmark | Mejor modelo MAE | Mejora MAE | Ganancia vs Dummy | \% mejor que Dummy |
|:----------|:-----------------|:----------:|:-----------------:|:------------------:|
| Borehole  | $GP\_RBF\_ARD$   |  $0.574$   |      $0.847$      |     $100.0\%$      |
| Branin    | $GP\_RBF$        |  $0.175$   |      $0.280$      |      $72.2\%$      |
| Forrester | $GP\_RBF$        |  $0.281$   |      $0.291$      |      $66.7\%$      |
| Hartmann6 | $GP\_Linear$     |  $0.466$   |      $0.306$      |      $94.4\%$      |

*Resumen de la mejora predictiva del GP y comparación frente a Dummy.*

El GP aporta valor predictivo frente a Dummy en todos los benchmarks, aunque con distinta intensidad. La mejora es especialmente clara en Borehole y Hartmann6, mientras que en Branin y Forrester es más moderada e irregular. Sin embargo, los mejores modelos predictivos no coinciden siempre con los mejores modelos de optimización. Por ejemplo, en Hartmann6 el mejor modelo según MAE es $GP\_Linear$, pero el mejor modelo para mejorar el mínimo encontrado es $GP\_RBF$. De forma similar, $GP\_RBF$ destaca en MAE para Branin y Forrester, mientras que los mejores modelos de optimización son $GP\_Matern52\_ARD$ y $GP\_Matern52$, respectivamente.

Las métricas probabilísticas se interpretan como diagnóstico complementario. En conjunto, el proceso de *infill* tiende a reducir simultáneamente el MAE y el NLPD respecto al estado inicial, aunque la calibración de la incertidumbre no es uniforme entre benchmarks. En Borehole y Branin se observa subcobertura en todos los modelos, lo que sugiere intervalos predictivos demasiado estrechos. En Forrester y Hartmann6 la cobertura se aproxima más al valor nominal en algunas configuraciones. Por tanto, la incertidumbre del GP resulta útil para guiar EI y diagnosticar la fiabilidad de las predicciones, pero no debe confundirse con una garantía directa de éxito en optimización.

Estos resultados confirman que la precisión global, la calibración probabilística y la utilidad para guiar la búsqueda del óptimo son aspectos relacionados, pero no equivalentes. Las figuras complementarias del Anexo 5 muestran con más detalle esta separación: RBF y Matérn concentran los comportamientos más estables en varios benchmarks, mientras que el kernel lineal resulta menos robusto y solo destaca en casos concretos; véanse las Figuras 5.3 y 5.4.

##### Influencia de las condiciones experimentales

Finalmente, se resume el efecto agregado de las condiciones experimentales evaluadas. Dado el elevado número de combinaciones, el objetivo no es describir cada configuración de forma aislada, sino identificar qué factores parecen condicionar más el rendimiento del proceso GP + EI.

| **Orden** | **Factor** | **Resultado observado** | **Conclusión** |
|:---|:---|:---|:---|
| 1 | Benchmark y dimensionalidad | La mejor mejora final del *incumbent* cambia mucho entre funciones: Branin alcanza $0.898$, mientras que Hartmann6 queda en $0.434$. | La dificultad del paisaje objetivo domina el resultado. La dimensionalidad y la estructura de la función condicionan más que cualquier decisión aislada del pipeline. |
| 2 | $n_{\text{train}}$ inicial | El efecto del tamaño inicial es visible dentro de un mismo benchmark: el rango de mejora llega a $0.457$ en Forrester y $0.365$ en Branin. | La información inicial condiciona la trayectoria de EI. Con pocos puntos hay más margen de mejora relativa, pero también más inestabilidad; por eso debe leerse junto al valor final alcanzado. |
| 3 | Kernel | Las diferencias entre kernels no siempre son concluyentes. En Borehole varios modelos quedan prácticamente empatados, mientras que en Branin, Forrester y Hartmann6 destacan sobre todo variantes RBF y Matérn. | El kernel importa, pero no debe leerse como una competición con un único ganador. RBF y Matérn ofrecen el comportamiento más consistente, aunque la elección final depende de la geometría del benchmark y del criterio analizado. |
| 4 | ARD | En mejora del *incumbent*, ARD frente al kernel base da $\Delta=0.002$, con $46\%$ de pares favorables ($n=108$). En MAE el efecto es mayor: $\Delta=0.109$, $63\%$ favorable. | ARD no mejora de forma sistemática la optimización, aunque puede ayudar a la calidad predictiva. Su utilidad depende del benchmark y no debe presentarse como ventaja general. |
| 5 | Sampler | Sobol frente a random da $\Delta=0.035$ en *incumbent*, pero solo $38\%$ de pares favorables ($n=186$). En MAE, el delta es negativo. | El sampler tiene un efecto secundario e irregular. Puede cambiar trayectorias concretas, pero no aparece como factor dominante frente al benchmark, el tamaño inicial o el kernel. |
| 6 | Ruido | Comparando sin ruido frente a ruido gaussiano, el *incumbent* mejora en media: $\Delta=0.110$ para $\sigma=0.5$ y $\Delta=0.125$ para $\sigma=1.0$. Sin embargo, los porcentajes favorables son moderados. | El ruido reduce la estabilidad del proceso y puede degradar la búsqueda, pero su efecto no es uniforme. Depende del benchmark y de si se analiza optimización o calidad predictiva. |

*Lectura cualitativa de los factores que condicionan el rendimiento en benchmarks.*

El Cuadro 3.6 no debe interpretarse como un ranking estadístico estricto, sino como una síntesis empírica de los patrones observados. La métrica principal para ordenar los factores es la mejora relativa final del mínimo observado, es decir, cuánto mejora el mejor valor encontrado tras el proceso de *infill*. Cuando se indican deltas, estos proceden de comparaciones emparejadas: por ejemplo, $\Delta>0$ en "Sobol -- random" significa que Sobol obtiene mayor mejora media que random bajo las mismas condiciones restantes. La conclusión principal es que el rendimiento del ciclo GP + EI está condicionado ante todo por el propio benchmark. En una lectura práctica, el foco debe ponerse primero en la dificultad geométrica del problema, el tamaño del diseño inicial y la familia de kernel; ARD, el sampler y el ruido se interpretan como factores secundarios, capaces de modificar trayectorias concretas pero no de compensar por sí solos un problema mal condicionado o con poca información inicial. El análisis de los efectos de estas condiciones se recoge como material complementario en la Figura 5.5.

### Experimento 2: caso real con dietas para insectos

El segundo experimento traslada el marco de modelos sustitutos a un caso real basado en datos experimentales de cría de insectos. En este contexto, el interés se centra en la fase larvaria, ya que es la fase utilizada posteriormente como fuente de alimentación. El problema experimental consiste en estudiar cómo distintas formulaciones de dieta afectan al crecimiento de las larvas y a su composición final.

El diseño experimental parte de una dieta control, compuesta únicamente por salvado de trigo. Esta dieta representa la referencia habitual de cría y permite comparar el efecto de introducir dietas no convencionales. Las dietas no convencionales mantienen el salvado de trigo como base, pero sustituyen una parte por un subproducto alimentario con valor nutricional. En el estudio principal se consideran tres subproductos: hoja de olivo, cascarilla de quinoa y orujo de oliva.

Desde el punto de vista del modelado, el objetivo no es únicamente ajustar un modelo a una tabla de datos, sino evaluar si un GP puede capturar relaciones útiles entre la composición de la dieta y la respuesta observada en las larvas. Estas relaciones son relevantes porque los ensayos físicos son limitados y cada nueva formulación requiere una validación experimental. Por tanto, el modelo sustituto se plantea como una herramienta previa para estimar respuestas y cuantificar incertidumbre antes de una posible fase prospectiva de selección de nuevas dietas.

Las respuestas de interés combinan parámetros productivos y variables de composición. En el plano productivo, una variable relevante es `FCR`, que mide la relación entre la cantidad de alimento consumido y la ganancia de peso obtenida. Valores menores de `FCR` indican una conversión más eficiente del alimento en biomasa. En el plano composicional, se consideran variables como `PROTEINA (%)` y `QUITINA (%)`. En las primeras pruebas también se consideró `TPC` que recoge la concentración de compuestos fenólicos totales en las larvas tras la cría y se utiliza como indicador adicional asociado al efecto de la dieta pero no se analizarán sus resultados por escasez de muestras.

En consecuencia, el caso real persigue dos objetivos metodológicos. En primer lugar, comprobar si el GP es capaz de generalizar a dietas no vistas a partir de variables de composición, inclusión y tipo de subproducto. En segundo lugar, analizar si la incertidumbre predictiva puede servir como señal de fiabilidad del modelo. A diferencia del experimento con *benchmarks* sintéticos, aquí no se ejecuta un ciclo secuencial de *infill*, ya que las observaciones proceden de ensayos experimentales ya realizados.

#### Alcance del caso real y selección de *Hermetia*

El archivo experimental original contiene información para dos especies: *Hermetia illucens* y *Tenebrio molitor*. Además, recoge varios bloques de información: composición nutricional y TPC de las dietas, parámetros productivos de la cría, y composición nutricional y TPC de las larvas obtenidas. En este experimento se trabaja únicamente con el bloque experimental de *Hermetia illucens*.

Sin embargo, los datos de *Tenebrio molitor* no se han utilizado porque mezclan varios bloques experimentales, incorporan tipos adicionales de dieta y presentan diferencias de tratamiento, como el suministro de agua en algunos controles. Para mantener un caso cerrado y comparable, el análisis se limita a *Hermetia illucens*: 33 observaciones organizadas en 11 dietas con 3 réplicas por dieta. En este bloque, las variables de entrada están completas y los objetivos tienen disponibilidad suficiente: FCR cuenta con 31 valores válidos, mientras que quitina y proteína cuentan con 33.

#### Diseño experimental y dietas consideradas

El conjunto analizado contiene una dieta control, basada en salvado de trigo, y varias dietas no convencionales en las que parte de ese salvado se sustituye por hoja de olivo, orujo de oliva o cascarilla de quinoa. Para identificar las formulaciones se usan etiquetas breves que combinan el subproducto y el porcentaje de inclusión: por ejemplo, `Hoja15` indica una dieta con un 15 % de hoja de olivo. La misma regla se aplica a las dietas con orujo y quinoa.

Cada formulación se ensayó en tres lotes independientes de larvas. Por ello, cada observación representa una réplica de una dieta, no una media por dieta. En la validación, las tres réplicas de una misma formulación se mantienen juntas: cuando una dieta se deja fuera, ninguna de sus réplicas participa en el entrenamiento. El Cuadro 3.7 resume el alcance del caso real.

| **Elemento** | **Descripción** |
|:---|:---|
| Bloque analizado | Ensayos de *Hermetia illucens* |
| Observaciones | 33 réplicas experimentales |
| Dietas evaluadas | 11 formulaciones |
| Réplicas por dieta | 3 lotes independientes |
| Dieta de referencia | Salvado de trigo sin inclusión de subproductos |
| Subproductos principales | Hoja de olivo, orujo de oliva y cascarilla de quinoa |
| Agrupación para validación | Réplicas de una misma dieta |
| Objetivos modelados | FCR, quitina y proteína larvaria |
| Nivel de trabajo | Réplica experimental |

*Resumen de los datos utilizados en el caso real.*

La representación de entrada describe cada dieta mediante variables de formulación y composición. Se comparan dos conjuntos: uno reducido, con porcentaje de inclusión, proteína, fibra, grasa y TPC medios de la dieta; y otro completo, que añade cenizas, carbohidratos y ratios nutricionales derivados. En ambos casos se incluye el tipo de subproducto, de forma que el modelo distingue entre control, hoja de olivo, orujo de oliva y quinoa sin depender solo del nombre de la dieta.

De este modo, el modelo no utiliza únicamente el nombre de la dieta como identificador, sino una descripción estructurada de cada formulación. Esta representación permite estudiar si las variables disponibles contienen información suficiente para anticipar la respuesta de dietas completas no vistas durante el entrenamiento.

#### Limpieza y preprocesamiento

El dataset utilizado en el caso real se obtiene a partir del archivo experimental original mediante un proceso de limpieza, combinación y transformación de tablas. El objetivo de esta etapa es pasar de varias hojas de Excel, con información distribuida por tipo de análisis, a una tabla única a nivel de réplica experimental. En esa tabla, cada fila debe contener tanto la respuesta observada como las variables necesarias para describir la dieta evaluada.

El proceso seguido puede resumirse en los siguientes pasos.

****Paso 1. Limpieza de las hojas originales.****

En primer lugar, se normalizan las hojas correspondientes a composición de dieta, parámetros productivos, composición larvaria y TPC. Durante esta etapa se corrigen nombres de columnas duplicados, se homogeneizan tratamientos y se estructuran las columnas de medias y desviaciones. También se normalizan los nombres de las variables que se utilizarán posteriormente en el modelado.

****Paso 2. Enumeración de réplicas.****

Para poder unir correctamente las distintas fuentes de información, se asigna un identificador de réplica dentro de cada dieta. Este paso es necesario porque los parámetros productivos y la composición de larvas se registran a nivel de lote o réplica de cría. De este modo, cada observación final puede asociarse a una dieta concreta y a una réplica experimental concreta.

****Paso 3. Construcción del dataset de productividad.****

Una vez limpias las tablas originales, se construye el dataset de productividad de *Hermetia*. Para ello, se combinan los parámetros productivos con la composición larvaria usando como claves la dieta, el tratamiento, la réplica y la especie. Después, se añade la composición nutricional de la dieta y la información de TPC. El resultado es una tabla a nivel de réplica que integra información productiva, composición de larva y descripción de la dieta.

****Paso 4. Extracción de metadatos de dieta.****

A partir del nombre textual de cada dieta se generan variables estructuradas, como `diet_name`, `byproduct_type` e `inclusion_pct`. Esta transformación permite que el modelo no dependa únicamente del nombre de la dieta, sino de variables interpretables: tipo de subproducto, porcentaje de inclusión y composición nutricional. Además, se calculan ratios nutricionales derivados que se emplean en el modo completo de variables.

****Paso 5. Selección del objetivo y filtrado de valores ausentes.****

El modelado se realiza de forma independiente para cada variable objetivo. Para cada objetivo se eliminan únicamente las filas cuyo valor de respuesta está ausente. Esta decisión evita descartar observaciones válidas para el resto de objetivos. Así, `FCR` se evalúa con 31 observaciones válidas, mientras que `QUITINA (%)` y `PROTEINA (%)` se evalúan con las 33 observaciones disponibles.

****Paso 6. Construcción de $X$, $\bm{y}$ y grupos.****

Para cada objetivo se construye la matriz de entrada $X$, el vector de respuesta $\bm{y}$ y el vector de grupos. El vector $\bm{y}$ contiene los valores observados del objetivo seleccionado. El vector de grupos se construye a partir de `diet_name`, de forma que todas las réplicas de una misma dieta compartan el mismo identificador de grupo. Esta agrupación será la base del protocolo LODO descrito en la Sección 3.3.4.

****Paso 7. Codificación de variables categóricas.****

Las variables de entrada se forman combinando las variables numéricas seleccionadas con una codificación binaria del tipo de subproducto. En concreto, `byproduct_type` se transforma en variables indicadoras y se concatena con las variables numéricas del modo reducido o completo. Todas las columnas resultantes se convierten a formato numérico antes del ajuste del modelo.

****Paso 8. Escalado dentro del `Pipeline`.****

El escalado de variables se realiza dentro del `Pipeline` de cada modelo. Por tanto, los parámetros de normalización se estiman únicamente con los datos de entrenamiento de cada partición y se aplican después sobre la dieta retenida en test.

En el CSV final utilizado, las variables de entrada seleccionadas no presentan valores ausentes. En conjunto, este preprocesamiento transforma los datos experimentales originales en una estructura supervisada y agrupada por dieta. Esta organización permite evaluar el modelo en una situación más exigente que una partición aleatoria por filas: predecir la respuesta de dietas completas no observadas durante el entrenamiento.

#### Protocolo de evaluación Leave-One-Diet-Out

La evaluación del caso real se realiza mediante un protocolo *Leave-One-Diet-Out* (LODO), motivado por la estructura agrupada del dataset. Las observaciones no son muestras independientes sin relación entre sí, sino réplicas experimentales asociadas a una misma dieta. Por tanto, una partición aleatoria por filas podría situar réplicas de una misma dieta simultáneamente en entrenamiento y test, generando una estimación demasiado optimista de la capacidad predictiva del modelo.

Sea $\mathcal{G}=\{g_1,\dots,g_K\}$ el conjunto de dietas disponibles. En el fold externo $k$, el conjunto de test queda formado por todas las observaciones pertenecientes a la dieta $g_k$:

$$
\mathcal{D}_{\text{test}}^{(k)}
    =
    \{(\bm{x}_i,y_i): g_i = g_k\}.
$$

El conjunto de entrenamiento contiene el resto de dietas:

$$
\mathcal{D}_{\text{train}}^{(k)}
    =
    \{(\bm{x}_i,y_i): g_i \in \mathcal{G}\setminus\{g_k\}\}.
$$

De este modo, todas las réplicas de la dieta retenida quedan excluidas del ajuste. En el dataset de *Hermetia*, al disponer de 11 dietas, el protocolo genera 11 folds externos.

*Esquema del protocolo LODO: en cada fold externo se reserva una dieta completa para medir el rendimiento final; la selección de hiperparámetros se realiza mediante validación interna sobre las dietas restantes.*

La selección de hiperparámetros se realiza de forma anidada. Para cada fold externo, la dieta retenida se reserva exclusivamente para la evaluación final. Sobre las dietas restantes se ejecuta una validación interna, también basada en grupos, y se selecciona la configuración con menor MAE. A continuación, el modelo se reentrena con todas las observaciones del entrenamiento externo y se evalúa sobre la dieta retenida.

Este diseño evalúa una tarea más exigente que interpolar entre réplicas de dietas ya observadas. El modelo debe predecir la respuesta de una dieta completa que no ha participado ni en el ajuste final ni en la selección de hiperparámetros. Por ello, los resultados obtenidos con LODO se interpretan como una estimación de la capacidad de generalización a dietas no vistas dentro del espacio experimental disponible.

#### Modelos, ajuste y métricas

En esta etapa se reutilizan las familias del Cuadro 3.1 que corresponden al caso real: Dummy, GP lineal, GP RBF, GP Matérn $3/2$, GP Matérn $5/2$ y GP compuesto. Todos los modelos se evalúan bajo el protocolo LODO anidado descrito en la Sección 3.3.4, de forma independiente para cada objetivo y para cada representación de entrada. Para evitar una exploración excesiva de variantes, ARD no se aplica a todas las familias desde el inicio, sino solo al mejor kernel previo de cada combinación de objetivo y representación de entrada; en el protocolo final, esta variante se limita a RBF o Matérn, no al kernel compuesto.

La configuración necesaria para reproducir ambos experimentos se resume de forma conjunta en el Anexo 8.

En el caso real, el GP compuesto se concreta mediante una suma de kernels:

$$
k(\bm{x},\bm{x}')
=
C_1\,k_{\text{Lineal}}(\bm{x},\bm{x}')
+
C_2\,k_{\text{Matern }5/2}(\bm{x},\bm{x}')
+
k_{\text{White}}(\bm{x},\bm{x}'),
$$

donde $C_1$ y $C_2$ son constantes de escala positivas. La componente lineal recoge una tendencia global de la respuesta respecto a las variables de entrada; la componente Matérn $5/2$ permite desviaciones locales suaves; y el término de ruido blanco representa variabilidad experimental no explicada por las covariables disponibles.

El ajuste se organiza en tres niveles:

****Selección interna.****

En cada fold externo, los parámetros se seleccionan mediante un LODO interno aplicado únicamente sobre las dietas de entrenamiento. La métrica de selección es el MAE, por lo que se elige la configuración con menor MAE en validación interna.

****Evaluación externa.****

Una vez seleccionados los parámetros, el modelo se reentrena con todas las dietas del entrenamiento externo y se evalúa sobre la dieta retenida. Las métricas finales proceden de esta evaluación externa, no del entrenamiento. En los modelos GP se obtiene tanto la media predictiva como la desviación típica. Esta segunda magnitud permite calcular métricas asociadas a incertidumbre.

La lectura de métricas se apoya en las definiciones predictivas y probabilísticas introducidas en la Sección 2.5, pero en este caso es necesario precisar cómo se agregan bajo validación LODO. El MAE se utiliza como métrica principal para seleccionar hiperparámetros y comparar modelos, porque mantiene las unidades originales de cada objetivo y permite interpretar directamente el error medio.

En LODO, cada fold externo corresponde a una dieta retenida. Por ello, el MAE macro se interpreta como el error medio por dieta: se calcula promediando el MAE de los folds externos, de modo que cada dieta tiene el mismo peso. Esta es la lectura principal del caso real, ya que el objetivo es evaluar la generalización a dietas completas no vistas, no favorecer dietas con más réplicas válidas.

Para comparar modelos frente a la línea base, se utiliza el MAE relativo respecto a Dummy:

$$
\rho_{\mathrm{Dummy}} = \frac{\mathrm{MAE}_{\mathrm{modelo}}}{\mathrm{MAE}_{\mathrm{Dummy}}}.
$$

Valores de $\rho_{\mathrm{Dummy}}<1$ indican que el modelo reduce el error frente a la predicción trivial. Además, cuando se analiza la dificultad de cada dieta retenida, el error se normaliza por el rango del objetivo correspondiente. Esta normalización permite comparar visualmente errores de variables con escalas distintas, como FCR, quitina y proteína.

En los modelos GP también se evalúa la incertidumbre predictiva. La cobertura del intervalo aproximado del 95 % mide la proporción de observaciones que caen dentro de $\mu \pm 1.96\sigma$. Además, se compara la desviación estándar predictiva $\sigma(\bm{x})$ con el error absoluto observado $|y-\mu(\bm{x})|$, con el fin de comprobar si el modelo asigna más incertidumbre a los casos donde efectivamente comete errores mayores. Estas métricas de incertidumbre no se utilizan para seleccionar parámetros, sino como diagnóstico de fiabilidad del GP.

Finalmente, en la parte prospectiva del caso real, EI se interpreta como una puntuación de adquisición y no como una métrica de validación. El *incumbent* observado representa la mejor dieta ya medida experimentalmente para cada objetivo, mientras que el candidato con mayor EI es una formulación no ensayada que el modelo considera informativa por su combinación de media predictiva e incertidumbre. Esta distinción evita presentar una recomendación prospectiva como si fuera un resultado experimental confirmado.

#### Resultados del caso real

Los resultados del caso real se analizan con una lectura distinta a la del experimento de benchmarks. En este escenario solo se dispone de un conjunto limitado de ensayos experimentales ya realizados. Por tanto, el interés principal no es medir únicamente el error medio del modelo, sino comprobar si el GP aprende relaciones transferibles a dietas completas no observadas durante el entrenamiento.

##### Generalización estructural frente a memorización

La primera cuestión es si el modelo sustituto extrae una señal general de las variables de formulación y composición, o si solo reconoce dietas similares a las observadas. Para ello se utiliza el protocolo LODO descrito en la Sección 3.3.4, que obliga al modelo a predecir una formulación completa no vista.

![Rendimiento relativo de los modelos frente al baseline Dummy en validación LODO. Valores inferiores a $1$ indican menor MAE que Dummy y, por tanto, mejor capacidad predictiva sobre dietas no vistas.](/assets/articles/surrogate_models/fig_01_lodo_generalization_vs_dummy.png)

![Dificultad de generalización por dieta retenida en el protocolo LODO. El color representa el error normalizado por el rango del objetivo y el texto muestra el MAE medio en las unidades originales de cada variable.](/assets/articles/surrogate_models/fig_02_error_by_left_out_diet.png)

La Figura 3.3 resume el rendimiento relativo frente al modelo Dummy. El eje vertical representa el MAE normalizado respecto al Dummy, por lo que valores inferiores a $1$ indican mejora frente a la predicción trivial. Bajo esta lectura, el GP compuesto mejora al Dummy en los tres objetivos activos: la reducción relativa del error es del $15.8\%$ en FCR, del $15.4\%$ en quitina y del $36.6\%$ en proteína. Este resultado indica que el modelo no se limita a reproducir una media global, sino que aprovecha información contenida en las variables de entrada para anticipar, al menos parcialmente, la respuesta de dietas no vistas.

No obstante, esta mejora agregada no debe interpretarse como una generalización uniforme. La Figura 3.4 descompone el error por dieta retenida y muestra que la dificultad del problema depende de la formulación evaluada. En FCR, los errores son relativamente contenidos en varias dietas, aunque aparecen dietas más difíciles, como `Hoja50` y algunas formulaciones con orujo o quinoa. En quitina, la dificultad se concentra especialmente en dietas como `Orujo70`, `Orujo90` y `Control`, mientras que en proteína destacan errores elevados en dietas como `Orujo50` y algunas formulaciones con hoja.

La lectura conjunta de ambas figuras es importante. La mejora frente al Dummy muestra que existe una señal aprendible en las variables nutricionales y de formulación, pero el mapa de errores evidencia que dicha señal no se transfiere igual a todas las dietas. Esto es coherente con la naturaleza del caso real: las respuestas de las larvas no dependen únicamente de una tendencia global asociada al porcentaje de inclusión o a la composición media, sino también de efectos específicos del subproducto, variabilidad experimental y posibles interacciones no completamente representadas en las variables disponibles.

En consecuencia, el resultado no permite afirmar que el GP resuelva por completo el problema de predicción de nuevas dietas. Sí permite, en cambio, defender que el modelo compuesto captura regularidades útiles más allá de la memorización de réplicas. La validación LODO actúa aquí como una prueba de generalización estructural: si el modelo mejora al Dummy cuando se retira una dieta completa, entonces parte de la información aprendida procede de relaciones compartidas entre dietas.

##### Expresividad geométrica del GP compuesto

Una vez comprobado que el GP aporta capacidad predictiva frente al baseline, la siguiente cuestión es qué tipo de estructura de covarianza resulta más adecuada para este caso real. Como se explicó en la Sección 2.2.3, el kernel no es únicamente un componente técnico del GP, sino el elemento que define las hipótesis a priori sobre la forma de la función que se desea aproximar. En este caso, la respuesta biológica de las larvas puede contener una tendencia global asociada a la composición nutricional de la dieta, pero también desviaciones locales debidas al tipo de subproducto, al porcentaje de inclusión y a la variabilidad experimental.

La Figura 3.5 compara distintas familias de kernel utilizando el MAE medio por dieta, es decir, el MAE macro del protocolo LODO. La comparación muestra que el kernel compuesto es el modelo más competitivo en los tres objetivos considerados. Esta diferencia es especialmente clara en proteína, donde el GP compuesto reduce de forma apreciable el error frente al Dummy y frente a los kernels simples. En quitina también se observa una mejora consistente respecto a las alternativas individuales, mientras que en FCR las diferencias entre varios kernels no lineales son más reducidas, aunque el compuesto se mantiene entre las mejores opciones.

![Comparación de familias de kernel en el caso real de *Hermetia*. El eje vertical muestra el MAE medio por dieta, equivalente al MAE macro obtenido mediante validación LODO; valores menores indican mejor generalización a dietas no vistas.](/assets/articles/surrogate_models/fig_04_kernel_family_mae.png)

Esta lectura es coherente con la definición previa del GP compuesto: una estructura de covarianza que combina una tendencia global, flexibilidad local y ruido experimental. Por ello, la comparación de la Figura 3.5 no enfrenta solo nombres de modelos, sino hipótesis distintas sobre la forma de la relación entre dieta y respuesta.

No obstante, este resultado debe interpretarse con prudencia. La superioridad del GP compuesto en la Figura 3.5 no significa que el modelo capture por completo la dinámica biológica del sistema, ni que pueda extrapolar con seguridad fuera del rango experimental observado. Lo que muestra es que, dentro del espacio cubierto por las dietas disponibles y bajo validación LODO, una estructura de covarianza que combina tendencia global, variación local y ruido experimental describe mejor los datos que las familias simples consideradas por separado.

Por tanto, el GP compuesto se adopta como modelo principal no porque elimine la incertidumbre del caso real, sino porque ofrece la mejor combinación observada entre flexibilidad y robustez.

##### Calibración práctica de la incertidumbre

Además de evaluar la precisión de la media predictiva, en este caso real interesa comprobar si la incertidumbre del GP aporta información útil sobre la fiabilidad de sus predicciones. Esta cuestión es especialmente relevante porque el objetivo no es únicamente interpolar dietas ya ensayadas, sino valorar si el modelo puede orientar decisiones sobre formulaciones no observadas. En este contexto, una predicción puntual con error puede seguir siendo útil si el modelo expresa una incertidumbre elevada en las regiones donde es probable que falle. Por el contrario, un modelo que comete errores grandes con intervalos demasiado estrechos resultaría poco fiable como herramienta prospectiva.

La Figura 3.6 compara, para cada objetivo, la desviación estándar predictiva del GP con el MAE cometido en las dietas retenidas por LODO. Esta representación permite analizar de forma práctica si las zonas donde el modelo declara mayor incertidumbre coinciden con aquellas donde el error real es mayor. La relación es más clara en quitina, donde se obtiene una correlación positiva entre incertidumbre y error. En este caso, el GP tiende a asignar mayor desviación estándar a observaciones que efectivamente resultan más difíciles de predecir, por lo que la incertidumbre actúa como una señal útil de cautela.

En FCR y proteína, sin embargo, la calibración es más débil. En FCR la relación entre desviación estándar y error es prácticamente nula, lo que indica que el modelo no ordena bien los casos fáciles y difíciles según su incertidumbre. En proteína la relación también es limitada, e incluso aparecen errores relevantes en zonas donde el modelo no siempre asigna una incertidumbre proporcionalmente alta. Por tanto, la incertidumbre del GP no debe interpretarse como una medida perfectamente calibrada en todos los objetivos. En conjunto, la Figura 3.6 muestra que la incertidumbre predictiva aporta información útil, pero no uniforme, sobre la fiabilidad del modelo. Una visualización complementaria de los intervalos predictivos por dieta se recoge en el Anexo 7.5.

![Relación entre la desviación estándar predictiva del GP y el error absoluto observado en validación LODO. Cada punto corresponde a una observación evaluada en una dieta retenida; el color indica el tipo de subproducto.](/assets/articles/surrogate_models/fig_05_uncertainty_vs_error.png)

En consecuencia, la incertidumbre del GP se interpreta en este trabajo como una señal de apoyo para la toma de decisiones, no como una garantía absoluta de fiabilidad. Su papel es especialmente útil para priorizar ensayos futuros y detectar regiones donde el modelo reconoce mayor desconocimiento. Sin embargo, cualquier candidato propuesto a partir del sustituto debe entenderse como una hipótesis experimental pendiente de validación, no como una conclusión cerrada sobre el rendimiento real de una nueva dieta.

##### Cribado prospectivo mediante EI

Como cierre del caso real, se analiza si el GP entrenado puede utilizarse como herramienta de cribado para una posible fase futura de experimentación. A diferencia del experimento de benchmarks, aquí no se ejecuta un ciclo físico de *infill*: no se ensayan nuevas dietas ni se actualiza el conjunto de datos. El objetivo es únicamente comprobar si el modelo permite priorizar formulaciones candidatas dentro de un espacio factible.

Para ello se emplea EI, introducido en la Sección 2.4.3 y formulado anteriormente, adaptando la orientación de cada objetivo: FCR se minimiza, mientras que quitina y proteína se maximizan. El espacio prospectivo se construye a partir de los subproductos considerados en el caso real, hoja, orujo y quinoa, y de porcentajes de inclusión dentro del rango observado. Sobre este dominio se genera una rejilla con incrementos del $1\%$, excluyendo las formulaciones ya ensayadas.

El Cuadro 3.8 compara, para cada objetivo, el mejor punto observado experimentalmente con el candidato no ensayado priorizado por EI.

| **Objetivo** | **Incumbent observado** | **Candidato EI** | **Predicción del candidato** | **EI** | **Lectura** |
|:---|:---|:---|:---|:---|:---|
| FCR | `Quinoa30`: $1.543$ | `Quinoa16` | $1.726 \pm 0.065$ | $2.7{\cdot}10^{-5}$ | La media predicha no mejora al incumbent, por lo que el candidato debe interpretarse como exploratorio. |
| Quitina | `Orujo70`: $15.318\%$ | `Orujo89` | $12.658 \pm 1.061\%$ | $0.002$ | EI prioriza una región de orujo con alta inclusión, aunque sin superar en media al mejor valor observado. |
| Proteína | `Orujo50`: $32.147\%$ | `Orujo31` | $30.938 \pm 0.912\%$ | $0.038$ | El candidato se sitúa en una zona no ensayada de orujo, pero la media predicha sigue por debajo del incumbent. |

*Incumbents observados y candidatos prospectivos priorizados por EI.*

Los resultados muestran que los candidatos EI no sustituyen a los mejores valores observados: para FCR el incumbent sigue siendo `Quinoa30`, para quitina `Orujo70` y para proteína `Orujo50`. Además, las medias predichas de los candidatos no superan claramente a dichos incumbents. Por tanto, la utilidad de EI en este caso no reside en demostrar nuevas dietas óptimas, sino en transformar la predicción media y la incertidumbre del GP en propuestas concretas para una siguiente iteración experimental.

En conjunto, este análisis debe interpretarse como una fase de pre-*infill*. El modelo sustituto reduce el espacio de búsqueda y señala formulaciones con cierto valor informativo, pero la decisión final requeriría validación experimental. Una visualización complementaria del paisaje de EI sobre la rejilla factible se recoge en el Anexo 7.6.

#### Síntesis de resultados

En conjunto, los resultados del capítulo muestran una lectura coherente entre los dos escenarios experimentales. En los benchmarks sintéticos, el ciclo GP + EI permite mejorar el mejor valor encontrado respecto al diseño inicial, especialmente cuando la incertidumbre del modelo ayuda a dirigir nuevas evaluaciones. En el caso real, la validación LODO indica que el GP compuesto aporta capacidad predictiva frente al modelo Dummy en varios objetivos, aunque su incertidumbre no aparece perfectamente calibrada en todos ellos. Por tanto, el modelo sustituto no debe interpretarse como una herramienta capaz de reemplazar la validación experimental, sino como un mecanismo para reducir el espacio de búsqueda, cuantificar parcialmente la incertidumbre y proponer candidatos razonables para una siguiente iteración de ensayos.

## Conclusiones y trabajo futuro

El objetivo de este Trabajo de Fin de Grado (TFG) ha sido abordar un problema de toma de decisiones: determinar qué nuevas configuraciones merece la pena evaluar en un experimento real cuyo coste es elevado. Como punto de partida, se llevó a cabo un estudio exhaustivo del problema y de las herramientas más adecuadas para su resolución. Todo el conocimiento desarrollado se ha presentado de forma rigurosa y fundamentada, partiendo de los principios del aprendizaje automático y profundizando en las bases teóricas de los procesos bayesianos, la optimización bayesiana y los modelos sustitutos. El propósito de este enfoque no es sustituir el experimento real, sino aprovechar la información obtenida a partir de los datos ya observados para orientar de manera más eficiente las decisiones sobre las siguientes configuraciones que conviene evaluar.

En una fase preliminar, se han realizado experimentos sintéticos que nos han permitido estudiar en un entorno controlado, donde la función real es conocida y donde se puede comprobar si el procedimiento de búsqueda mejora el diseño inicial. En este escenario, el ciclo formado por el ajuste del proceso gaussiano, la selección de nuevos puntos mediante *Expected Improvement* y la actualización del conjunto observado ha mejorado el mejor valor encontrado en todos los benchmarks analizados. Esto confirma que, cuando el modelo dispone de una señal suficiente, puede guiar la búsqueda hacia regiones más prometedoras que las obtenidas únicamente con el diseño inicial.

Estos experimentos previos también muestran una conclusión importante: un modelo que predice bien en promedio no siempre es el que mejor ayuda a optimizar. En este tipo de problemas no solo importa reducir el error medio, sino también saber utilizar la incertidumbre para decidir dónde conviene explorar. Por ello, la utilidad de un proceso gaussiano no depende únicamente de su precisión predictiva, sino de su capacidad para combinar predicción e incertidumbre en una estrategia de selección de nuevas evaluaciones.

En el caso real de dietas para *Hermetia illucens*, el problema se ha abordado como una tarea de aprendizaje a partir de datos experimentales ya disponibles. Para evaluar y validar el modelo de una forma cercana al uso que tendría en la práctica, se ha utilizado un protocolo Leave-One-Diet-Out, dejando fuera dietas completas durante el entrenamiento y comprobando si el modelo es capaz de estimar sus resultados. Este planteamiento es más exigente que predecir réplicas aisladas, por su parecido con el caso real de querer valorar una dieta todavía no ensayada.

Bajo esta evaluación, el proceso gaussiano mejora al modelo simple de referencia en FCR, quitina y proteína. Esto indica, por tanto, que las variables de formulación y composición de la dieta contienen información útil y que el modelo no se limita únicamente a memorizar los datos disponibles. Aun así, los resultados deben interpretarse con prudencia: el conjunto de datos es pequeño, la generalización no es igual de buena para todas las dietas y la incertidumbre predictiva aparece solo parcialmente calibrada. Por tanto, las predicciones del modelo deben entenderse como una ayuda para tomar decisiones, no como una conclusión biológica definitiva.

Una lectura importante del caso real es que el trabajo se encuentra en una primera fase del ciclo experimental. Con los datos actuales, el modelo puede proponer dietas candidatas y señalar regiones que parecen interesantes, pero todavía no puede confirmar por sí solo cuál es la mejor formulación. Precisamente por eso, el proceso de *infill* resulta relevante: al seleccionar nuevas dietas mediante un criterio como Expected Improvement, evaluarlas experimentalmente y volver a entrenar el modelo con esos nuevos datos, el proceso gaussiano podría hacerse progresivamente más robusto, más útil para priorizar candidatos y, potencialmente, mejor calibrado.

Desde una perspectiva aplicada, la recomendación para un equipo científico que trabaja con piensos o dietas sería utilizar esta metodología como una herramienta de apoyo antes de diseñar la siguiente ronda de ensayos. En lugar de probar nuevas formulaciones de forma poco dirigida, el modelo permite aprovechar los experimentos ya realizados para ordenar mejor las opciones disponibles, identificar candidatos prometedores y decidir dónde merece más la pena invertir tiempo y recursos. En conjunto, el trabajo muestra que los modelos sustitutos no reemplazan la validación experimental, pero sí pueden reducir el espacio de búsqueda, cuantificar incertidumbre y ayudar a elegir con más criterio las próximas dietas que deberían probarse.

### Trabajo futuro

La primera línea de trabajo futuro consiste en cerrar el ciclo experimental del caso real. En este TFG, el caso real llega hasta una fase de pre-*infill*: el modelo se entrena con los datos disponibles, se valida mediante LODO y se utiliza para proponer candidatos prospectivos. El siguiente paso natural sería seleccionar una o varias formulaciones candidatas, evaluarlas experimentalmente y reentrenar el GP con las nuevas observaciones. De este modo, el caso real pasaría de una validación retrospectiva a un verdadero proceso secuencial de optimización basada en modelos sustitutos.

Una segunda línea de continuación sería formalizar una función de decisión multiobjetivo. En el análisis actual, FCR, quitina y proteína se estudian por separado, lo que permite interpretar cada objetivo de forma individual. Sin embargo, desde un punto de vista productivo, la elección de una dieta depende normalmente de varios criterios simultáneos, como eficiencia de conversión, composición nutricional, coste, disponibilidad del subproducto o viabilidad experimental. Definir una regla de decisión junto con criterio experto permitiría estudiar compromisos entre objetivos y no solo óptimos individuales.

También sería importante ampliar el dataset experimental. El caso real utilizado contiene 11 dietas con 3 réplicas por dieta, suficiente para una primera validación metodológica, pero todavía limitado para extraer conclusiones biológicas fuertes. Incorporar nuevas dietas, más niveles de inclusión y más réplicas permitiría mejorar la estabilidad del modelo, estudiar mejor la calibración de la incertidumbre y comprobar si las relaciones aprendidas se mantienen fuera del rango actualmente observado.

Desde el punto de vista metodológico, futuras extensiones podrían explorar modelos capaces de tratar varios objetivos de forma conjunta, como procesos gaussianos multi-salida o modelos jerárquicos. Esto permitiría aprovechar posibles relaciones entre FCR, proteína y quitina, especialmente en escenarios de pocos datos. También sería interesante estudiar modelos con ruido heterocedástico o técnicas de recalibración probabilística, dado que la incertidumbre del GP no se comporta igual en todos los objetivos.

Por último, el proceso de adquisición podría ampliarse más allá de EI. En futuras versiones podrían utilizarse criterios multiobjetivo, criterios con restricciones o estrategias de *batch infill* que propongan varias dietas para ensayar en una misma ronda experimental. Estas extensiones permitirían convertir el marco desarrollado en una herramienta más completa de apoyo a la decisión, capaz de guiar campañas experimentales sucesivas con un uso más eficiente del presupuesto disponible.

## Material complementario del experimento de benchmarks

Este anexo recoge el material metodológico y gráfico del experimento de benchmarks que se ha retirado del cuerpo principal para mantener la memoria dentro del límite de extensión. En el Capítulo 3 se conservan los elementos principales del análisis: el cuadro de benchmarks, la configuración general del experimento, el resumen del *incumbent*, la evolución del *incumbent*, el resumen predictivo y la jerarquía de factores experimentales.

### Dominio detallado del benchmark Borehole

| **Variable** | **Descripción**                      | **Dominio**       |
|:-------------|:-------------------------------------|:------------------|
| $r_w$        | Radio del pozo                       | $[0.05, 0.15]$    |
| $r$          | Radio de influencia                  | $[100, 50000]$    |
| $T_u$        | Transmisividad del acuífero superior | $[63070, 115600]$ |
| $H_u$        | Carga piezométrica superior          | $[990, 1110]$     |
| $T_l$        | Transmisividad del acuífero inferior | $[63.1, 116]$     |
| $H_l$        | Carga piezométrica inferior          | $[700, 820]$      |
| $L$          | Longitud del pozo                    | $[1120, 1680]$    |
| $K_w$        | Conductividad hidráulica del pozo    | $[9855, 12045]$   |

*Dominio del benchmark Borehole.*

### Flujo del proceso de *infill*

*Flujo del proceso de infill: diseño inicial, entrenamiento del GP, predicción de media e incertidumbre, cálculo y maximización de EI, evaluación del benchmark, actualización de los datos y medición de resultados. El ciclo se repite hasta agotar el presupuesto.*

#### Descripción detallada del proceso de *infill*

El proceso de *infill* reproduce un escenario de optimización con evaluaciones costosas, donde cada nuevo punto debe seleccionarse de forma informada. En este experimento, la función objetivo se formula como un problema de minimización, por lo que la mejora esperada se calcula respecto al mejor valor observado hasta el momento.

En cada iteración, el GP se ajusta con el conjunto de observaciones disponible $\mathcal{D}_n$ y se calcula el criterio EI sobre el dominio de búsqueda. Aunque la función objetivo se minimiza, EI se maximiza, ya que se busca el punto con mayor mejora esperada:

$$
\bm{x}_{\mathrm{next}}
=
\operatorname*{arg\,max}_{\bm{x}\in\mathcal{X}} (EI(\bm{x})).
$$

La maximización de EI se realiza mediante *Differential Evolution*, al tratarse de una estrategia robusta para funciones de adquisición potencialmente multimodales y sin necesidad de gradientes. Además, se emplea `polish=True`, aplicando un refinamiento local con L-BFGS-B tras la búsqueda global. Dado que los optimizadores de SciPy están formulados como problemas de minimización, en la implementación se optimiza $-EI(\bm{x})$, lo que equivale a maximizar $EI(\bm{x})$.

### Configuración de semillas y reproducibilidad

Esta sección recoge únicamente la asignación de semillas utilizada en las trayectorias de benchmarks. La configuración computacional completa y el procedimiento de reproducción se documentan en el Anexo 8.

| **Elemento** | **Semilla utilizada** |
|:---|:---|
| Entrenamiento inicial | 42 |
| Conjunto de test | 1042 |
| Ruido en entrenamiento y test | 42 |
| Ruido durante *infill* | 42 |
| EI, Differential Evolution y respaldo aleatorio | En cada iteración se utiliza una semilla dependiente del paso, $S + \text{step} \times 1009$, donde $S$ es la semilla base de la configuración. |
| Auditoría de validación cruzada en modo activo | Si se activa la auditoría mediante KFold, la semilla depende del paso para mantener la reproducibilidad de cada iteración. |

*Configuración de semillas empleada en el experimento de benchmarks.*

### Mejora final del mínimo encontrado por benchmark

![Mejora relativa final del mínimo encontrado durante el proceso de *infill*. En Forrester, Branin y Hartmann6 se representa la reducción relativa del *gap* al óptimo. En Borehole, al no disponer de óptimo definido en la implementación, se muestra la reducción relativa del mejor valor limpio respecto al diseño inicial. Las barras indican la media y las líneas verticales la desviación típica sobre las trayectorias agregadas.](/assets/articles/surrogate_models/evolution_incumbent_best_model_by_benchmark.png)

La Figura 5.2 resume el éxito de optimización del proceso de *infill*, no la calidad predictiva global del modelo sustituto. En Branin se obtiene la mayor reducción media del *gap* al óptimo, con GP_Matern52_ARD, seguida de Forrester con GP_Matern52. Borehole también muestra una mejora fuerte, aunque en este caso se mide la reducción del mejor valor limpio porque no se dispone de un óptimo de referencia en la implementación. Hartmann6 presenta una mejora más moderada y con mayor variabilidad.

En conjunto, la figura confirma que EI permite encontrar mejores valores de la función objetivo en todos los benchmarks, pero también muestra que la magnitud de la mejora depende mucho del problema. Por ello, el *incumbent* debe interpretarse como la métrica central del éxito de búsqueda, complementaria al MAE y al resto de métricas predictivas.

### Evolución relativa del MAE

![Evolución relativa del MAE para el mejor modelo predictivo de cada benchmark. La línea discontinua marca el valor inicial. Valores inferiores a 1 indican una reducción del error respecto al modelo entrenado únicamente con el diseño inicial.](/assets/articles/surrogate_models/evolution_relative_mae_best_model_by_benchmark.png)

La Figura 5.3 muestra la evolución del MAE relativo durante el proceso de *infill* para el mejor modelo predictivo de cada benchmark. La línea discontinua en $1$ representa el error del modelo entrenado solo con el diseño inicial; por tanto, valores inferiores a $1$ indican una reducción del MAE y valores superiores reflejan un empeoramiento temporal.

El patrón más claro aparece en Borehole, donde GP_RBF_ARD reduce el error de forma sostenida, sobre todo cuando el diseño inicial es pequeño. En Hartmann6 también se observa una mejora rápida, especialmente para $n_{\mathrm{train}}=6$, donde GP_Linear alcanza una reducción marcada del MAE. En cambio, Branin y Forrester presentan trayectorias menos monótonas: algunas configuraciones empeoran al inicio antes de estabilizarse o mejorar. Esto indica que añadir puntos mediante EI no siempre reduce el error global de forma inmediata.

En conjunto, la figura muestra que el *infill* puede mejorar la calidad predictiva del modelo sustituto, pero que esta mejora depende del benchmark, del tamaño inicial y del modelo. Además, el MAE relativo no debe confundirse con el éxito de optimización: EI selecciona puntos para mejorar la búsqueda del óptimo, no necesariamente para minimizar de forma monótona el error predictivo en todo el dominio.

### Diagnóstico predictivo y probabilístico

![Relación entre precisión puntual y diagnóstico probabilístico durante el proceso de *infill*. El eje horizontal representa el MAE relativo respecto al inicio y el eje vertical el NLPD relativo. El cuadrante inferior izquierdo corresponde al caso deseable, con mejora simultánea en error puntual y diagnóstico probabilístico. El color indica el error de calibración de la cobertura al 95 %, definido como $\left|\mathrm{coverage}_{95}-0.95\right|$.](/assets/articles/surrogate_models/mae_vs_probabilistic_diagnostics_by_step.png)

La Figura 5.4 relaciona la mejora en error puntual con la mejora del diagnóstico probabilístico durante el *infill*. El eje horizontal muestra el MAE relativo y el eje vertical el NLPD relativo, ambos respecto al inicio. Por tanto, el cuadrante inferior izquierdo representa el caso deseable: menor error medio y mejor calidad probabilística que en el diseño inicial. El color añade una tercera lectura, el error de calibración de la cobertura al $95\,\%$, donde tonos más oscuros indican intervalos mejor calibrados.

Los cuatro benchmarks terminan en la zona de mejora simultánea, aunque con matices. Borehole muestra la mejora más clara en MAE y NLPD, con un punto final cercano a $0.47$ en MAE relativo y $0.05$ en NLPD relativo; sin embargo, mantiene subcobertura, por lo que la incertidumbre no queda plenamente calibrada. Forrester también mejora de forma marcada en NLPD y de manera moderada en MAE. Branin presenta una mejora más suave y una trayectoria menos directa. Hartmann6 ofrece el comportamiento más equilibrado: reduce el MAE y mantiene una calibración cercana al $95\,\%$.

La conclusión principal es que la mejora predictiva no debe evaluarse solo con MAE. Un modelo puede reducir el error medio y seguir proporcionando intervalos de incertidumbre mal calibrados. Por ello, esta figura complementa la evolución del MAE y justifica analizar conjuntamente precisión puntual, NLPD y cobertura predictiva.

### Efecto de las condiciones experimentales

![Efectos emparejados de distintas condiciones experimentales sobre la mejora del mínimo encontrado y la mejora relativa del MAE. Un delta positivo indica que la primera condición de la comparación obtiene mayor mejora media. Las barras horizontales representan la desviación típica de los deltas emparejados.](/assets/articles/surrogate_models/factor_effects_paired_for_tfg.png)

La Figura 5.5 compara el efecto medio de distintas condiciones experimentales manteniendo emparejadas el resto de configuraciones. Los deltas son, en general, pequeños respecto a su variabilidad, lo que indica que ningún factor aislado domina de forma estable el resultado. En mejora del *incumbent*, la ausencia de ruido muestra un efecto positivo moderado frente a ruido gaussiano, mientras que Sobol frente a aleatorio y ARD frente a kernel base presentan efectos débiles.

En MAE, ARD tiende a mejorar ligeramente el resultado medio, pero con alta dispersión. El muestreo Sobol no muestra una ventaja clara frente al aleatorio y el efecto del ruido sobre el MAE es pequeño. Por tanto, la figura sugiere que el comportamiento final depende más del benchmark y de la interacción entre condiciones que de una única decisión experimental aislada.

## Análisis complementario del proceso de *infill*

Este anexo complementa el material de benchmarks del Anexo 5 con una lectura visual y estadística del proceso de *infill*. El objetivo no es introducir nuevos resultados principales, sino mostrar con más detalle cómo el modelo sustituto participa en la selección secuencial de puntos y por qué las métricas predictivas y las métricas de optimización no siempre seleccionan el mismo modelo.

### Relación entre mejora predictiva y mejora del mínimo encontrado

El Cuadro 6.1 compara, para cada benchmark, el modelo con mayor mejora del mínimo encontrado frente al modelo con mayor mejora relativa del MAE. Esta comparación es relevante porque el objetivo de un ciclo de optimización por *infill* no es necesariamente minimizar el error medio global del modelo sustituto, sino encontrar mejores valores de la función objetivo con un presupuesto limitado de evaluaciones.

| **Benchmark** | **Mejor modelo por mínimo encontrado** | **Mejora *inc.*** | **Mejor modelo por MAE** | **Mejora MAE** |
|:---|:---|:---|:---|:---|
| Borehole | GP_Linear | 0.720 | GP_RBF_ARD | 0.574 |
| Branin | GP_Matern52_ARD | 0.898 | GP_RBF | 0.175 |
| Forrester | GP_Matern52 | 0.737 | GP_RBF | 0.281 |
| Hartmann6 | GP_RBF | 0.434 | GP_Linear | 0.466 |

*Comparación entre el mejor modelo según mejora del mínimo encontrado y el mejor modelo según mejora relativa del MAE.*

En todos los benchmarks del Cuadro 6.1, el modelo que más reduce el error medio no coincide necesariamente con el modelo que consigue mayor mejora del mínimo encontrado. Este resultado justifica analizar por separado la calidad predictiva del modelo sustituto y la eficacia del proceso de búsqueda.

### Forrester como caso visual del mecanismo de *infill*

Forrester se utiliza como caso visual porque es un benchmark unidimensional. Esta propiedad permite representar simultáneamente la función real, la media posterior del GP, la incertidumbre predictiva y los puntos observados. Por tanto, resulta adecuado para explicar de forma gráfica el papel de EI en la selección de nuevas evaluaciones.

![Evolución del proceso de *infill* en Forrester sin ruido. Cada panel muestra el estado del GP tras incorporar un número creciente de puntos. La línea azul representa la función real, la línea naranja la media predictiva del GP, las bandas muestran la incertidumbre y los puntos indican las observaciones disponibles.](/assets/articles/surrogate_models/app_forrester_gp_evolution_no_noise.png)

La Figura 6.1 permite observar cómo el modelo sustituto cambia a medida que se incorporan nuevas observaciones. En las primeras iteraciones, la incertidumbre domina grandes zonas del dominio. Conforme el proceso avanza, el modelo ajusta mejor la región cercana al mínimo y reduce la incertidumbre alrededor de los puntos evaluados. Esta visualización ayuda a interpretar por qué el proceso no debe evaluarse solo mediante MAE: una predicción global imperfecta puede seguir siendo útil si orienta la búsqueda hacia regiones prometedoras.

### Decisión puntual mediante Expected Improvement

La Figura 6.2 muestra una decisión concreta de EI en Forrester. El GP se entrena con cuatro puntos iniciales de Sobol y varias evaluaciones de *infill* previas. En el paso representado, EI propone el siguiente punto en $x_{\mathrm{next}}=0.763$, muy cercano al óptimo conocido del benchmark.

![Síntesis visual de una decisión de Expected Improvement en Forrester. El panel superior muestra la media posterior, la incertidumbre, las observaciones previas, el *incumbent* y el punto propuesto. El panel inferior izquierdo representa la función de adquisición EI, y el panel inferior derecho cuantifica el cambio en el gap al óptimo tras evaluar el nuevo punto.](/assets/articles/surrogate_models/app_forrester_ei_decision_synthesis_step5.png)

| **Elemento** | **Valor** |
|:---|:---|
| Benchmark | Forrester |
| Modelo | GP_Matern52 |
| Diseño inicial | Sobol, $n_{\mathrm{train}}=4$, sin ruido |
| Paso de *infill* | 5 |
| Punto propuesto | $x_{\mathrm{next}}=0.763$ |
| EI en el punto propuesto | 1.519 |
| Mejor valor observado antes | -4.364 |
| Valor observado en $x_{\mathrm{next}}$ | -6.004 |
| Óptimo conocido | $f(x^\star)=-6.021$ |
| Gap cerrado tras la evaluación | 99.0 % |

*Resumen numérico de la decisión EI representada en la Figura 6.2.*

Este ejemplo muestra el papel práctico de la incertidumbre en un GP. EI no selecciona un nuevo punto solo por la media predicha, sino por la combinación entre mejora esperada y dispersión predictiva. En este caso, el punto elegido queda muy cerca del óptimo real y reduce casi por completo el gap que existía antes de la evaluación.

### Contrastes estadísticos complementarios

Además de las tablas descriptivas, los registros generados por el experimento permiten realizar contrastes no paramétricos sobre los resultados de los benchmarks. Estos contrastes no se plantean como una demostración formal de superioridad universal, sino como una comprobación complementaria de dos efectos observados en el análisis principal: la mejora del *incumbent* durante el proceso de *infill* y la existencia de diferencias entre familias de kernels GP.

Para evaluar la mejora del *incumbent*, se comparó el mejor valor observado al inicio y al final de cada trayectoria activa. Cada trayectoria queda definida por benchmark, muestreador, tamaño inicial, configuración de ruido, modo de validación y modelo. Dado que la función objetivo de los benchmarks se formula como minimización, una reducción del *incumbent* indica mejora. El Cuadro 6.3 resume un contraste de rangos con signo de Wilcoxon unilateral, con alternativa de mejora tras el proceso de *infill*.

| **Modelo** | **Trayectorias** | **Mejoran** | **Mediana mejora** | **$p$-valor** |
|:---|---:|---:|---:|---:|
| Todos los GP | 372 | 316 | 2.812 | $7.35\times10^{-54}$ |
| GP_Linear | 66 | 30 | 0.000 | $8.67\times10^{-7}$ |
| GP_Matern32 | 66 | 62 | 2.970 | $3.79\times10^{-12}$ |
| GP_Matern52 | 66 | 60 | 2.519 | $8.15\times10^{-12}$ |
| GP_Matern52_ARD | 54 | 54 | 8.105 | $8.13\times10^{-11}$ |
| GP_RBF | 66 | 59 | 2.574 | $1.20\times10^{-11}$ |
| GP_RBF_ARD | 54 | 51 | 7.602 | $2.57\times10^{-10}$ |

*Contraste de mejora del incumbent entre el inicio y el final de las trayectorias de infill.*

Los resultados muestran que el proceso de *infill* reduce el *incumbent* de forma sistemática en las trayectorias consideradas. El efecto es especialmente claro en los kernels con ARD y en los kernels Matérn/RBF, mientras que el modelo lineal presenta una mejora mediana nula, aunque sin empeorar en muchas configuraciones y con mejoras puntuales suficientes para que el contraste agregado sea significativo.

Para comparar los kernels GP en términos predictivos, se utilizó el MAE de los bloques completos compartidos por todos los kernels. Cada bloque corresponde a una combinación de benchmark, muestreador, tamaño inicial, configuración de ruido y modo de validación. Sobre los 54 bloques completos, el test de Friedman detecta diferencias entre modelos $(p=1.88\times10^{-6})$. El Cuadro 6.4 muestra los rangos medios y los contrastes pareados de Wilcoxon frente al kernel con mejor rango medio.

| **Modelo** | **Rango medio** | **$p$ vs. GP_Matern52_ARD** | **Lectura** |
|:---|---:|---:|:---|
| GP_Matern52_ARD | 2.741 | -- | Mejor rango medio. |
| GP_RBF_ARD | 2.815 | 0.677 | Rendimiento prácticamente indistinguible del mejor. |
| GP_Matern52 | 3.370 | $4.51\times10^{-4}$ | Peor que la versión ARD en el contraste pareado. |
| GP_RBF | 3.593 | 0.138 | Diferencia no concluyente frente al mejor. |
| GP_Matern32 | 4.056 | $2.04\times10^{-5}$ | Peor que el mejor kernel en los bloques compartidos. |
| GP_Linear | 4.426 | $5.28\times10^{-8}$ | Menor rendimiento predictivo agregado. |

*Comparación no paramétrica entre kernels GP usando el MAE en bloques completos. Rangos menores indican mejor comportamiento.*

Estos contrastes apoyan dos conclusiones utilizadas en la memoria: primero, el ciclo de *infill* aporta mejora efectiva sobre el mejor valor observado; segundo, los kernels con ARD tienden a ocupar las mejores posiciones en MAE. Sin embargo, no se observa una separación estadística clara entre GP_Matern52_ARD y GP_RBF_ARD, por lo que la elección de un kernel concreto debe interpretarse junto con la estabilidad, el coste computacional y el comportamiento en cada problema.

## Material complementario del caso real

Este anexo recoge material de apoyo del caso real de *Hermetia illucens*. El objetivo es documentar la trazabilidad del dataset modelado y aportar figuras complementarias que ayudan a interpretar los resultados del Capítulo 3, sin sustituir las figuras principales de generalización LODO, comparación de kernels, incertidumbre y pre-*infill*.

### Inventario de datasets generados

El flujo de preparación genera varios datasets a partir de los archivos experimentales originales. Sin embargo, el modelado del caso real se limita al dataset de productividad de *Hermetia*, por las razones de alcance y homogeneidad descritas en el Capítulo 3. El Cuadro 7.1 resume los datasets generados y su papel en el trabajo.

| **Dataset** | **Filas** | **Dietas** | **Especie** | **Uso** |
|:---|:---|:---|:---|:---|
| `productivity_hermetia_lote.csv` | 33 | 11 | *Hermetia* | Dataset modelado en el flujo de preparación del caso real. |
| `productivity_tenebrio_lote.csv` | 57 | 19 | *Tenebrio* | Dataset generado disponible, no modelado en los resultados actuales. |
| `productivity_all_lote.csv` | 90 | 19 | Ambas | Dataset combinado disponible para análisis posteriores. |
| `quality_hermetia_dieta.csv` | 11 | 11 | *Hermetia* | Dataset de calidad de dieta generado como soporte. |
| `quality_tenebrio_dieta.csv` | 19 | 19 | *Tenebrio* | Dataset de calidad de dieta generado como soporte. |
| `quality_all_dieta.csv` | 30 | 19 | Ambas | Dataset combinado de calidad disponible. |

*Inventario de datasets generados en el caso Entomotive.*

En el dataset modelado, las 33 observaciones corresponden a 11 dietas con tres réplicas por dieta. Los objetivos utilizados son `FCR`, `QUITINA (%)` y `PROTEINA (%)`. La disponibilidad es completa para quitina y proteína, con 33 valores válidos, y ligeramente menor para FCR, con 31 valores válidos.

### Valores observados por dieta

La Figura 7.1 resume los valores medios observados por dieta para los tres objetivos del caso real. Esta figura no se utiliza como resultado principal del modelo, sino como comprobación descriptiva previa: permite ver que los mejores valores observados no pertenecen siempre a la misma dieta y que el problema tiene una naturaleza multiobjetivo.

![Valores medios observados por dieta para FCR, quitina y proteína en el dataset de *Hermetia*. En FCR valores menores son mejores; en quitina y proteína valores mayores son mejores.](/assets/articles/surrogate_models/app_real_dataset_targets_by_diet.png)

| **Objetivo** | **Criterio** | **Dieta** | **Subproducto** | **Valor medio observado** |
|:---|:---|:---|:---|:---|
| FCR | Minimizar | Quinoa30 | quinoa | 1.543 |
| Quitina | Maximizar | Orujo70 | orujo | 15.318 |
| Proteína | Maximizar | Orujo50 | orujo | 32.147 |

*Mejores dietas observadas por objetivo individual.*

### Paridad del mejor GP por objetivo

La Figura 7.2 complementa las métricas agregadas del Capítulo 3. En lugar de resumir el error en un único valor, representa observados frente a predichos en validación LODO para el mejor GP de cada objetivo. La línea diagonal indica predicción perfecta.

![Predicciones LODO del mejor GP por objetivo frente a los valores observados. Cada punto corresponde a una réplica experimental y el color indica el tipo de subproducto.](/assets/articles/surrogate_models/app_real_best_gp_parity_by_target.png)

Esta figura permite detectar patrones que quedan ocultos en el MAE macro. En FCR, las predicciones tienden a concentrarse en un intervalo estrecho, lo que indica dificultad para extrapolar entre dietas. En quitina y proteína se observa una señal más clara, aunque con errores relevantes en algunos subproductos. Por tanto, el modelo aporta información útil, pero no debe interpretarse como sustituto del ensayo experimental.

### Contrastes LODO frente al modelo base Dummy

Los resultados LODO del caso real también permiten una lectura estadística complementaria, aunque necesariamente prudente por el tamaño muestral. En este análisis, cada dieta retenida en validación LODO se utiliza como unidad emparejada, de modo que los modelos se comparan sobre las mismas particiones de entrenamiento y prueba. La métrica utilizada es el MAE por dieta, ya que es la métrica principal del análisis de generalización.

Se aplicó primero un test de Friedman por objetivo para comparar todos los modelos disponibles. Después, como contraste dirigido frente al modelo base, se aplicó un test de rangos con signo de Wilcoxon unilateral entre el Dummy y el GP con mejor rango medio en cada objetivo. Los $p$-valores deben interpretarse como evidencia exploratoria, no como una prueba concluyente, porque cada objetivo dispone solo de 11 dietas.

| **Objetivo** | **GP de referencia** | **Rango** | **$p_F$** | **Mejora** | **$\Delta$MAE med.** | **Lectura** |
|:---|:---|---:|---:|---:|---:|:---|
| FCR | GP_Matern32_NoARD | 3.455 | 0.521 | 8/11 | 0.017 | Tendencia favorable al GP, pero sin evidencia concluyente frente al Dummy $(p=0.062)$. |
| Proteína | GP_Matern32_ARD | 3.273 | 0.209 | 9/11 | 0.827 | Evidencia favorable al GP frente al Dummy $(p=0.034)$; otros kernels GP también muestran mejoras significativas exploratorias. |
| Quitina | GP_Linear | 2.636 | 0.194 | 8/11 | 0.380 | Mejora mediana positiva, pero contraste no concluyente $(p=0.160)$. |

*Contrastes complementarios sobre el MAE LODO por dieta frente al modelo base Dummy. \DeltaMAE representa MAE\_{\mathrm{Dummy}}-MAE\_{\mathrm{GP}}, por lo que valores positivos favorecen al GP.*

La lectura principal de estos contrastes es coherente con las figuras de paridad y con las métricas agregadas. En proteína, el modelo GP aporta una mejora más consistente respecto al modelo base. En FCR se observa una señal débil, compatible con la dificultad de extrapolar entre dietas y con la concentración de las predicciones en un rango estrecho. En quitina, aunque algunos folds favorecen al GP, la evidencia no es suficiente para afirmar una mejora estadísticamente robusta con los datos disponibles.

### Intervalos predictivos por dieta

La Figura 7.3 muestra, para cada dieta retenida en validación LODO, la media predictiva del GP junto con intervalos aproximados del $95\,\%$. Esta visualización complementa el análisis de calibración presentado en la Sección 3.3.6.3, ya que permite observar de forma desagregada en qué dietas los intervalos contienen razonablemente los valores observados y en cuáles aparecen fallos concretos de cobertura.

![Intervalos predictivos por dieta bajo validación LODO. Los puntos negros representan la media predictiva del GP y las barras muestran intervalos aproximados del $95\,\%$ construidos como $\mu \pm 1.96\sigma$. Los puntos coloreados corresponden a los valores observados por tipo de subproducto.](/assets/articles/surrogate_models/fig_06_prediction_intervals_by_diet.png)

### Paisaje prospectivo de EI

La Figura 7.4 muestra el paisaje prospectivo de EI sobre la rejilla factible de formulaciones considerada en el caso real. Esta visualización complementa el cribado presentado en la Sección 3.3.6.4, donde se comparan los incumbents observados con los candidatos no ensayados priorizados por EI.

El objetivo de esta figura no es demostrar que los candidatos propuestos sean óptimos, sino mostrar cómo el criterio de adquisición distribuye el valor prospectivo sobre el espacio de búsqueda. Las zonas con mayor EI corresponden a combinaciones en las que el modelo identifica una posible mejora respecto al estado observado o una incertidumbre suficientemente relevante como para justificar una evaluación futura.

![Paisaje prospectivo de EI sobre la rejilla factible de formulaciones. Los máximos de EI identifican puntos candidatos para una posible evaluación futura, combinando predicción media e incertidumbre del GP.](/assets/articles/surrogate_models/fig_08_ei_candidate_landscape.png)

Por tanto, esta figura debe interpretarse como una herramienta de apoyo al diseño experimental posterior. En el estado actual del trabajo, los puntos destacados por EI constituyen hipótesis de ensayo y no resultados experimentales validados.

## Configuración computacional y reproducibilidad

Este anexo documenta la configuración computacional empleada para generar los resultados del trabajo. Su objetivo no es introducir nueva metodología, sino dejar trazable cómo se conectan el protocolo experimental, el código y los ficheros de salida utilizados en la memoria. A diferencia de los anexos A--C, centrados en resultados y figuras complementarias, aquí solo se recoge la configuración necesaria para reproducirlos.

### Configuración del experimento de benchmarks

El experimento de benchmarks reproduce un escenario de optimización con evaluaciones limitadas. Para cada función se genera un diseño inicial, se ajusta un GP y se añaden nuevas observaciones mediante EI. La configuración relevante procede de `src/configs/benchmark_tuning_specs.py` y de la lógica de aprendizaje activo implementada en `src/analysis/active_learning.py`. El Cuadro 8.1 resume los valores utilizados.

| **Elemento** | **Configuración** |
|:---|:---|
| Benchmarks retenidos en la memoria | Forrester, Branin, Hartmann6 y Borehole. |
| Muestreo inicial | Sobol y muestreo aleatorio uniforme. |
| Tamaños iniciales | $n_{\mathrm{train}}\in\{1,d,4d\}$, donde $d$ es la dimensión del benchmark. |
| Conjunto de test | 200 puntos generados dentro del dominio de cada benchmark. |
| Ruido sintético | Sin ruido, ruido gaussiano con $\sigma=0.5$ y ruido gaussiano con $\sigma=1.0$. |
| Familias de modelo | Dummy, GP lineal, GP RBF, GP Matérn $3/2$, GP Matérn $5/2$; para dimensión mayor que uno se incluyen variantes ARD en RBF y Matérn $5/2$. |
| Presupuesto de *infill* | $5d$ nuevas evaluaciones, con un máximo de 50 observaciones totales durante la trayectoria. |
| Función de adquisición | Expected Improvement con parámetro de exploración $\xi=0.01$. |
| Optimización de EI | Differential Evolution sobre el dominio continuo del benchmark, con refinamiento local posterior. En la implementación se minimiza $-EI(\bm{x})$, lo que equivale a maximizar $EI(\bm{x})$. |
| Presupuesto del optimizador de EI | $\max(2000,500d)$ evaluaciones candidatas equivalentes para dimensionar la búsqueda de la adquisición. |
| Semilla base | 42\. En cada paso de *infill*, la semilla de EI se desplaza como $42 + 1009\cdot \mathrm{step}$. |

*Configuración computacional del experimento de benchmarks.*

Los niveles de ruido usados son absolutos y por tanto no representan la misma severidad relativa en todos los benchmarks. Por ello, el análisis del ruido debe entenderse como una prueba de robustez del pipeline bajo perturbaciones fijas, no como una comparación normalizada del efecto del ruido entre funciones.

Los hiperparámetros internos del GP, como longitudes características, amplitud de señal y nivel de ruido blanco, se reestiman durante el ajuste del `GaussianProcessRegressor` mediante maximización de la log-verosimilitud marginal. Por tanto, en los benchmarks no debe interpretarse cada familia de kernel como un conjunto fijo de parámetros numéricos, sino como una hipótesis estructural de covarianza que se ajusta a los datos disponibles en cada iteración.

La implementación contiene además rejillas específicas por benchmark en `benchmark_grids.py`. Estas rejillas sirven como soporte para auditorías o modos de ajuste, pero la lectura principal de la memoria se centra en la comparación agregada de familias GP, el efecto del diseño inicial, el ruido, el método de muestreo y la evolución del *incumbent* durante el proceso de *infill*.

### Configuración del caso real

El caso real se ejecuta a partir del dataset `productivity_hermetia_lote.csv`. Para cada objetivo se construye una matriz de entrada, un vector de respuesta y un vector de grupos por dieta. El grupo de validación no es la réplica, sino la dieta completa, de forma que las tres réplicas de una formulación permanecen juntas en entrenamiento o en test.

Se evalúan tres objetivos activos: FCR, quitina y proteína. FCR se interpreta como una variable a minimizar, mientras que quitina y proteína se interpretan como variables a maximizar en la fase prospectiva. La disponibilidad final es de 31 observaciones para FCR y 33 observaciones para quitina y proteína.

#### Representaciones de entrada

Se comparan dos representaciones de entrada. Ambas incluyen variables numéricas de formulación y composición de la dieta, junto con indicadores binarios del tipo de subproducto. El nombre de la dieta se utiliza para definir los grupos LODO, pero no se utiliza como variable predictora directa.

| **Modo** | **Dimensión** | **Variables incluidas** |
|:---|:---|:---|
| Reducido | 9 variables | Cinco variables numéricas: porcentaje de inclusión, proteína media de la dieta, fibra media, grasa media y TPC medio de la dieta. A ellas se añaden cuatro indicadores binarios de subproducto: control, hoja, orujo y quinoa. |
| Completo | 14 variables | Las variables del modo reducido, más cenizas medias, carbohidratos medios y tres ratios derivados: proteína/carbohidratos, proteína/fibra y fibra/grasa. También incluye los cuatro indicadores de subproducto. |

*Representaciones de entrada utilizadas en el caso real.*

Las variables de entrada seleccionadas no presentan valores ausentes en el dataset final utilizado. La imputación por mediana existe en el flujo de preparación como mecanismo de seguridad, pero no modifica la representación de entrada en los resultados principales. El escalado de variables se realiza dentro del `Pipeline` del modelo GP, por lo que los parámetros de estandarización se ajustan únicamente con el entrenamiento de cada fold.

#### Modelos y rejillas del caso real

El ajuste del caso real utiliza validación LODO anidada. En cada fold externo se retiene una dieta completa como test. Sobre las dietas restantes se ejecuta una validación LODO interna para seleccionar hiperparámetros. Después, el modelo se reentrena con todas las dietas del entrenamiento externo y se evalúa sobre la dieta retenida. La métrica de selección interna es MAE.

| **Modelo** | **Descripción** |
|:---|:---|
| Dummy | Línea base constante. La rejilla permite seleccionar entre media y mediana del conjunto de entrenamiento interno. |
| GP lineal | Kernel `DotProduct` más ruido blanco. Representa una tendencia global lineal. |
| GP RBF | Kernel RBF más ruido blanco. Se evalúa en versión isotrópica y, cuando procede, con ARD. |
| GP Matérn $3/2$ | Kernel Matérn con $\nu=3/2$ más ruido blanco. Se evalúa en versión isotrópica y, cuando procede, con ARD. |
| GP Matérn $5/2$ | Kernel Matérn con $\nu=5/2$ más ruido blanco. Se evalúa en versión isotrópica y, cuando procede, con ARD. |
| GP compuesto | Suma de componente lineal, componente Matérn $5/2$ y ruido blanco. En el protocolo final se evalúa sin ARD para evitar una exploración excesiva de variantes en un dataset pequeño. |

*Familias de modelos evaluadas en el caso real.*

La rejilla común de los GP se resume en el Cuadro 8.4. Cada familia de kernel se evalúa con la estructura indicada en el Cuadro 8.3. Dentro de cada fold, `GaussianProcessRegressor` optimiza los parámetros continuos del kernel mediante la log-verosimilitud marginal.

| **Parámetro** | **Valores considerados** |
|:---|:---|
| `alpha` | $\{10^{-10},10^{-5},10^{-2},1.0\}$. Este término se añade a la diagonal de la matriz kernel durante el ajuste y ayuda a controlar estabilidad numérica y regularización efectiva. |
| `kernel` | Una estructura de kernel por familia de modelo: lineal, RBF, Matérn $3/2$, Matérn $5/2$ o compuesto. |
| `normalize_y` | `True`. La variable objetivo se normaliza internamente durante el ajuste del GP. |
| `n_restarts_optimizer` | 15 reinicios del optimizador de la log-verosimilitud marginal. |
| Ruido blanco | `WhiteKernel` con nivel inicial $10^{-4}$ y cotas $[10^{-7},0.8]$. |
| Longitudes características | En kernels isotrópicos se usa una escala inicial común. En ARD se usa un vector inicial de unos, con una escala por variable de entrada. Las cotas son $[10^{-2},10^{5}]$. |

*Rejilla de hiperparámetros usada en el caso real.*

Para limitar el número de modelos en un conjunto de datos pequeño, ARD no se aplica a todas las familias de forma indiscriminada. Primero se comparan las familias base y después se evalúa una variante ARD seleccionada por combinación de objetivo y representación de entrada. El Cuadro 8.5 resume esas variantes.

| **Representación** | **Objetivo** | **Variante ARD evaluada** |
|:-------------------|:-------------|:--------------------------|
| Reducida           | FCR          | GP RBF ARD                |
| Reducida           | Quitina      | GP RBF ARD                |
| Reducida           | Proteína     | GP Matérn $3/2$ ARD       |
| Completa           | FCR          | GP Matérn $5/2$ ARD       |
| Completa           | Quitina      | GP Matérn $5/2$ ARD       |
| Completa           | Proteína     | GP Matérn $3/2$ ARD       |

*Variantes ARD evaluadas por combinación de objetivo y representación.*

### Prevención de fuga de información

La prevención de fuga de información es especialmente relevante en el caso real, porque las observaciones son réplicas de dietas y no muestras independientes sin estructura. El protocolo evita la fuga mediante cuatro decisiones:

1.  **Agrupación por dieta.** Todas las réplicas de una misma formulación comparten el mismo grupo. En cada fold externo se deja fuera una dieta completa.

2.  **Selección anidada.** La dieta retenida en el fold externo no participa en la selección de hiperparámetros. La selección se realiza con LODO interno sobre las dietas de entrenamiento.

3.  **Escalado dentro del modelo.** La estandarización de $X$ se ajusta dentro del `Pipeline` usando únicamente los datos de entrenamiento del fold correspondiente.

4.  **Uso estructural del tipo de subproducto.** El modelo recibe indicadores de subproducto y variables de composición, pero no usa el nombre de la dieta como identificador predictivo directo. El nombre de dieta se reserva para construir los grupos LODO.

La codificación binaria del tipo de subproducto se construye a partir de una variable experimental conocida de antemano. Esta codificación no utiliza la respuesta del fold retenido. En consecuencia, el riesgo principal de fuga que se evita no es conocer qué subproductos existen, sino permitir que réplicas de una dieta aparezcan simultáneamente en entrenamiento y test.

### Reproducción de resultados del caso real

El caso real puede reproducirse con el script `reproduce_tfg_outputs.py`. Por defecto, este script reutiliza los registros de ajuste ya calculados, ejecuta la auditoría, regenera las figuras finales y copia los recursos gráficos necesarios a la carpeta de recursos del TFG. Si se desea recalcular completamente el ajuste LODO, puede ejecutarse con la opción `--rerun-tuning`.

    python reproduce_tfg_outputs.py
    python reproduce_tfg_outputs.py --rerun-tuning

El script comprueba que existan los resultados esperados para las dos representaciones de entrada y los tres objetivos activos. En la ejecución registrada, el manifiesto de reproducibilidad recoge 6 resúmenes de ajuste y 42 ficheros de folds, correspondientes a las combinaciones de representación, objetivo y modelos activos. El fichero de manifiesto se guarda en:

    outputs/reproducibility_real_case_manifest.json

La reproducibilidad documentada en este anexo tiene dos niveles. En primer lugar, el caso real puede regenerarse desde los registros de ajuste existentes, lo que permite reconstruir las tablas, auditorías, figuras y recursos gráficos usados por la memoria. En segundo lugar, si se ejecuta el ajuste completo, se recalculan los modelos LODO anidados desde el dataset de *Hermetia*. Esta segunda opción es computacionalmente más costosa porque repite la selección interna de hiperparámetros para cada fold externo, objetivo, representación y familia de modelo.

En los benchmarks, los resultados dependen de las semillas, del método de muestreo, del tamaño inicial, del ruido y del presupuesto de *infill*. Por ello, la memoria no interpreta trayectorias individuales como conclusiones definitivas, sino patrones agregados sobre las configuraciones evaluadas. Esta distinción es importante: el objetivo del experimento benchmark es estudiar el comportamiento del ciclo GP + EI bajo condiciones controladas, mientras que el objetivo del caso real es comprobar si el mismo marco puede aportar señal útil antes de plantear una nueva campaña experimental.

## Acrónimos

|  |  |
|:---|:---|
| GP | Gaussian Process (Proceso Gaussiano) |
| BO | Bayesian Optimization (Optimización Bayesiana) |
| SBO | Surrogate-Based Optimization (Optimización basada en modelos sustitutos) |
| DoE | Design of Experiments (Diseño de experimentos) |
| BB | Black-Box (Caja negra) |
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| NLL | Negative Log-Likelihood |
| LHS | Latin Hypercube Sampling (Muestreo Hipercubo Latino) |
| CV | Cross-Validation (Validación cruzada) |
| ARD | Automatic Relevance Determination (Determinación automática de relevancia) |
| RBF | Radial Basis Function (Función de base radial) |
| SE | Squared Exponential (Exponencial cuadrática; también RBF) |
| LML | Log Marginal Likelihood (Log-verosimilitud marginal / evidencia) |
| EI | Expected Improvement |
| DE | Differential Evolution |

## Referencias

Bishop, Christopher M. 2006. *Pattern Recognition and Machine Learning*. Information Science and Statistics. Springer.

Dixon, L. C. W., and G. P. Szegö. 1978. "The Global Optimization Problem: An Introduction." In *Towards Global Optimisation 2*, edited by L. C. W. Dixon and G. P. Szegö. North-Holland.

Forrester, Alexander I. J., András Sóbester, and Andy J. Keane. 2008. *Engineering Design via Surrogate Modelling: A Practical Guide*. Wiley. <https://doi.org/10.1002/9780470770801>.

Gneiting, Tilmann, and Adrian E. Raftery. 2007. "Strictly Proper Scoring Rules, Prediction, and Estimation." *Journal of the American Statistical Association* 102 (477): 359--78. <https://doi.org/10.1198/016214506000001437>.

Hastie, Trevor, Robert Tibshirani, and Jerome Friedman. 2009. *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. 2nd ed. Springer Series in Statistics. Springer. <https://doi.org/10.1007/978-0-387-84858-7>.

Jiang, Ping, Qi Zhou, and Xinyu Shao. 2020. *Surrogate Model-Based Engineering Design and Optimization*. Springer Tracts in Mechanical Engineering. Springer Singapore. <https://doi.org/10.1007/978-981-15-0731-1>.

Jones, Donald R., Matthias Schonlau, and William J. Welch. 1998. "Efficient Global Optimization of Expensive Black-Box Functions." *Journal of Global Optimization* 13 (4): 455--92. <https://doi.org/10.1023/A:1008306431147>.

Kuleshov, Volodymyr, Nathan Fenner, and Stefano Ermon. 2018. "Accurate Uncertainties for Deep Learning Using Calibrated Regression." *Proceedings of the 35th International Conference on Machine Learning*, Proceedings of machine learning research, vol. 80: 2796--804. <https://proceedings.mlr.press/v80/kuleshov18a.html>.

Morris, Max D., Toby J. Mitchell, and Donald Ylvisaker. 1993. "Bayesian Design and Analysis of Computer Experiments: Use of Derivatives in Surface Prediction." *Technometrics* 35 (3): 243--55. <https://doi.org/10.1080/00401706.1993.10485320>.

NIST Digital Library of Mathematical Functions. 2010. "Chapter 10: Bessel Functions." Edited by Frank W. J. Olver, Daniel W. Lozier, Ronald F. Boisvert, and Charles W. Clark. <https://dlmf.nist.gov/10>.

Rasmussen, Carl Edward, and Christopher K. I. Williams. 2006. *Gaussian Processes for Machine Learning*. Adaptive Computation and Machine Learning. MIT Press. <http://www.GaussianProcess.org/gpml>.

Schölkopf, Bernhard, Ralf Herbrich, and Alexander J. Smola. 2001. "A Generalized Representer Theorem." *Computational Learning Theory*, Lecture notes in computer science, vol. 2111: 416--26. <https://doi.org/10.1007/3-540-44581-1_27>.

Settles, Burr. 2009. *Active Learning Literature Survey*. Computer Sciences Technical Report No. 1648. University of Wisconsin--Madison.

Sobol', Ilya M. 1967. "On the Distribution of Points in a Cube and the Approximate Evaluation of Integrals." *USSR Computational Mathematics and Mathematical Physics* 7 (4): 86--112. <https://doi.org/10.1016/0041-5553(67)90144-9>.

Surjanovic, Sonja, and Derek Bingham. 2013. "Virtual Library of Simulation Experiments: Test Functions and Datasets." <https://www.sfu.ca/~ssurjano/optimization.html>.
