---
title: "Generación de Imágenes mediante Ecuaciones Diferenciales Estocásticas y Procesos de Difusión"
date: 2025-05-27
excerpt: "Un análisis exhaustivo de los modelos de difusión y su aplicación en la generación de imágenes."
# tags: ["python", "sklearn", "mlstudio", "gemini"]
draft: false
lang: "es"
kind: "article"
# readingTime: 7
---
# Del ruido a la imagen: una lectura matemática de los modelos de difusión

Los modelos de difusión se pueden entender a partir de una idea aparentemente simple: si sabemos destruir una imagen poco a poco hasta convertirla en ruido, quizá podamos aprender el camino inverso. La parte interesante es que ese camino inverso no consiste en memorizar imágenes ni en aplicar un filtro de limpieza clásico. Consiste en aprender la geometría local de una distribución de probabilidad: hacia dónde debe moverse una muestra ruidosa para parecerse cada vez más a una imagen real.

En este artículo presento una implementación de modelos de difusión formulados mediante Ecuaciones Diferenciales Estocásticas (*Stochastic Differential Equations*, SDEs), siguiendo el marco de Song et al. [^song2021scorebased_sde]. La idea no es únicamente mostrar imágenes generadas, sino explicar por qué el método funciona, qué papel juega el *score* $\nabla_x \log p_t(x)$ y cómo aparecen de forma natural la SDE inversa, el entrenamiento por *denoising score matching*, los *samplers* y la *probability flow ODE*.

El código asociado está disponible en GitHub:

<https://github.com/trentisiete/image_generation_difussion_project>

---

## 1. El problema: aprender una distribución en un espacio enorme

Una imagen no es solo una matriz de píxeles. Desde el punto de vista probabilístico, una imagen es un punto en un espacio de dimensión muy alta. Una imagen RGB de $32 \times 32$, como las de CIFAR-10, vive en un espacio de dimensión $32 \cdot 32 \cdot 3 = 3072$. Sin embargo, no todos los puntos de ese espacio son imágenes naturales. La mayoría son ruido sin estructura. Las imágenes reales ocupan una región muy especial: contienen bordes, texturas, simetrías, objetos, fondos, colores compatibles y patrones que no aparecen al azar.

El objetivo de un modelo generativo es aproximar la distribución de los datos reales, que denotamos por $p_{\text{data}}(x)$, para poder generar nuevas muestras:

$$
x \sim p_{\text{model}}(x)
$$

donde $p_{\text{model}}$ intenta parecerse a $p_{\text{data}}$.

El reto es que $p_{\text{data}}(x)$ no se conoce de forma explícita. Tenemos ejemplos, pero no una fórmula cerrada de la distribución. Los modelos de difusión atacan este problema de una manera elegante: en vez de intentar aprender directamente cómo generar una imagen limpia desde cero, construyen primero un proceso que destruye los datos de manera controlada y después aprenden a invertirlo.

---

## 2. La intuición central: destruir es fácil, reconstruir es aprender

El proceso directo empieza con una imagen real $x(0) \sim p_{\text{data}}$ y va añadiendo ruido gradualmente hasta obtener una variable $x(T)$ que se aproxima a una distribución simple, normalmente una Gaussiana. Conceptualmente:

$$
x(0) \longrightarrow x(t) \longrightarrow x(T) \approx \mathcal{N}(0,I)
$$

Este proceso es fácil porque lo definimos nosotros. Podemos elegir cuánto ruido añadir y cómo evoluciona ese ruido con el tiempo.

El proceso inverso es el verdaderamente generativo:

$$
x(T) \longrightarrow x(t) \longrightarrow x(0)
$$

Partimos de ruido puro y vamos retirando ruido hasta obtener una imagen plausible. Pero aquí aparece la pregunta clave: ¿cómo sabe el modelo hacia dónde debe moverse una muestra ruidosa?

La respuesta es el *score*:

$$
\nabla_x \log p_t(x)
$$

Este vector apunta en la dirección en la que aumenta más rápidamente la densidad logarítmica de los datos perturbados en el tiempo $t$. Intuitivamente, si $x(t)$ es una imagen ruidosa, el score indica hacia qué zona del espacio debería desplazarse para ser más probable bajo la distribución de imágenes con ese nivel de ruido.

**Primer hallazgo importante:** el modelo no necesita aprender directamente la distribución completa $p_t(x)$. Le basta con aprender su campo de gradientes. En vez de preguntarse “¿cuál es la probabilidad exacta de esta imagen?”, aprende “¿hacia dónde debería moverme para estar en una región más probable?”.

---

## 3. El proceso directo como SDE

En tiempo continuo, el proceso de difusión directo se expresa mediante una Ecuación Diferencial Estocástica:

$$
dx = f(x,t)\,dt + g(t)\,dW(t)
$$

donde:

- $x(t)$ representa la muestra en el instante $t$.
- $f(x,t)$ es el término de deriva (*drift*), que introduce una evolución determinista.
- $g(t)$ controla la intensidad del ruido.
- $W(t)$ es un proceso de Wiener estándar, es decir, movimiento Browniano.

La SDE combina dos efectos. El término $f(x,t)\,dt$ empuja la muestra de forma determinista, mientras que $g(t)\,dW(t)$ introduce aleatoriedad. En el contexto de difusión, esa aleatoriedad es precisamente el mecanismo que transforma datos estructurados en ruido.

### 3.1. Variance Exploding SDE

La VE-SDE se define como:

$$
dx = \sqrt{\frac{d[\sigma^2(t)]}{dt}}\,dW(t)
$$

Aquí no hay deriva determinista, porque $f(x,t)=0$. El dato conserva su centro, pero su varianza aumenta con el tiempo. Por eso se llama *Variance Exploding*: la varianza crece progresivamente hasta que la señal original queda dominada por el ruido.

La transición del proceso directo tiene forma gaussiana:

$$
p_{0t}(x(t)\mid x(0)) =
\mathcal{N}\left(x(t);\,x(0),\, [\sigma^2(t)-\sigma^2(0)]I\right)
$$

En esta familia, el dato se perturba añadiendo ruido cada vez más intenso alrededor de la imagen original.

### 3.2. Variance Preserving SDE

La VP-SDE tiene la forma:

$$
dx = -\frac{1}{2}\beta(t)x(t)\,dt + \sqrt{\beta(t)}\,dW(t)
$$

A diferencia de la VE-SDE, aquí sí hay deriva. El término

$$
-\frac{1}{2}\beta(t)x(t)
$$

contrae progresivamente la señal, mientras que el término estocástico añade ruido. La combinación está diseñada para preservar la varianza global cuando los datos están normalizados.

Su transición también es gaussiana:

$$
p_{0t}(x(t)\mid x(0)) =
\mathcal{N}\left(
x(t);
x(0)e^{-\frac{1}{2}\int_0^t \beta(s)\,ds},
\left[1-e^{-\int_0^t \beta(s)\,ds}\right]I
\right)
$$

Esta ecuación es muy reveladora: la media de la imagen se va apagando con el factor exponencial, mientras que la varianza del ruido aumenta de manera complementaria. La señal desaparece, pero no de golpe; se disuelve siguiendo una trayectoria controlada.

### 3.3. Sub-VP SDE

La sub-VP SDE modifica el término de difusión:

$$
dx =
-\frac{1}{2}\beta(t)x(t)\,dt
+
\sqrt{\beta(t)\left(1-e^{-2\int_0^t \beta(s)\,ds}\right)}\,dW(t)
$$

Esta variante mantiene una estructura similar a la VP-SDE, pero ajusta la cantidad de ruido introducida en cada instante. En la práctica puede producir mejores resultados en términos de verosimilitud, aunque no necesariamente coincide siempre con la mejor calidad perceptual.

---

## 4. El giro matemático: invertir una difusión

Hasta aquí solo hemos definido cómo destruir una imagen. La parte generativa aparece cuando usamos el resultado de Anderson sobre procesos de difusión en tiempo inverso [^anderson1982reverse]. Si el proceso directo es:

$$
dx = f(x,t)\,dt + g(t)\,dW(t)
$$

entonces existe un proceso inverso que permite recorrer las mismas distribuciones marginales en sentido contrario:

$$
dx =
\left[
f(x,t) - g(t)^2 \nabla_x \log p_t(x)
\right]dt
+
g(t)\,d\bar{W}(t)
$$

donde $d\bar{W}(t)$ representa el movimiento Browniano en tiempo inverso.

Esta ecuación contiene el núcleo de los modelos de difusión basados en score. Todo lo conocido del proceso directo aparece en $f$ y $g$. Lo desconocido queda concentrado en un único término:

$$
\nabla_x \log p_t(x)
$$

Es decir, para invertir la difusión no necesitamos aprender toda la dinámica desde cero. Necesitamos estimar el score de la distribución perturbada en cada instante.

**Segundo hallazgo importante:** la generación se reduce a aprender un campo vectorial. Si el modelo sabe estimar en cada nivel de ruido hacia dónde aumenta la densidad de las imágenes, puede guiar una muestra desde el ruido hasta una región donde viven imágenes realistas.

En la práctica se entrena una red neuronal $s_\theta(x,t)$ para aproximar:

$$
s_\theta(x,t) \approx \nabla_x \log p_t(x)
$$

Al sustituir el score real por el score aprendido, la SDE inversa queda:

$$
dx =
\left[
f(x,t) - g(t)^2 s_\theta(x,t)
\right]dt
+
g(t)\,d\bar{W}(t)
$$

Esta es la ecuación que se discretiza durante el muestreo.

---

## 5. Cómo se entrena el score sin conocer la densidad

A primera vista parece que tenemos un problema circular. Queremos entrenar una red para aproximar $\nabla_x \log p_t(x)$, pero no conocemos $p_t(x)$. Si no conocemos la distribución, ¿cómo obtenemos la etiqueta de entrenamiento?

La clave está en no usar directamente el score marginal desconocido, sino el score condicional del proceso de perturbación, que sí conocemos porque el proceso directo lo hemos definido nosotros.

Para muchas SDEs utilizadas en difusión, la transición directa se puede escribir como una Gaussiana:

$$
p_{0t}(x(t)\mid x(0)) =
\mathcal{N}\left(x(t);\mu_t(x(0)),\sigma_t^2 I\right)
$$

donde $\mu_t(x(0))$ es la media en el tiempo $t$ y $\sigma_t^2$ la varianza del ruido añadido.

Ahora calculamos el score de esta Gaussiana respecto a $x(t)$. La densidad logarítmica, ignorando constantes que no dependen de $x(t)$, es:

$$
\log p_{0t}(x(t)\mid x(0))
=
-\frac{1}{2\sigma_t^2}
\left\|x(t)-\mu_t(x(0))\right\|_2^2
+
C
$$

Derivando respecto a $x(t)$:

$$
\nabla_{x(t)} \log p_{0t}(x(t)\mid x(0))
=
-\frac{x(t)-\mu_t(x(0))}{\sigma_t^2}
$$

Esta ecuación es uno de los puntos más elegantes del método. Si generamos una muestra ruidosa $x(t)$ a partir de una imagen limpia $x(0)$, conocemos exactamente cuál debería ser el score condicional que apunta de vuelta hacia la media de la perturbación.

**Tercer hallazgo importante:** el propio proceso de añadir ruido genera las etiquetas de entrenamiento. No necesitamos anotar imágenes, ni conocer $p_{\text{data}}(x)$, ni calcular una densidad intratable. Basta con tomar una imagen, perturbarla y enseñar a la red a predecir el vector que deshace esa perturbación.

La función de pérdida de *denoising score matching* en tiempo continuo queda:

$$
\theta^*
=
\arg\min_\theta
\mathbb{E}_{t \sim \mathcal{U}(0,T)}
\left[
\lambda(t)
\mathbb{E}_{x(0)\sim p_{\text{data}}}
\mathbb{E}_{x(t)\mid x(0)}
\left[
\left\|
s_\theta(x(t),t)
-
\nabla_{x(t)}
\log p_{0t}(x(t)\mid x(0))
\right\|_2^2
\right]
\right]
$$

Sustituyendo el score condicional gaussiano:

$$
\left\|
s_\theta(x(t),t)
+
\frac{x(t)-\mu_t(x(0))}{\sigma_t^2}
\right\|_2^2
$$

La red aprende, para cada nivel de ruido, cómo debe corregirse una muestra perturbada para volver hacia una zona de alta probabilidad.

---

## 6. Del score al muestreo: cómo aparece una imagen

Una vez entrenado $s_\theta(x,t)$, la generación consiste en resolver numéricamente la SDE inversa desde $t=T$ hasta $t=0$. Se inicializa:

$$
x(T) \sim p_T(x)
$$

normalmente una Gaussiana, y se aplica un integrador en tiempo inverso.

### 6.1. Euler-Maruyama

Euler-Maruyama es el integrador básico para SDEs. Si discretizamos el tiempo como $t_N=T,\dots,t_0=0$, un paso inverso puede interpretarse como:

$$
x_{i-1}
=
x_i
+
\left[
f(x_i,t_i)
-
g(t_i)^2 s_\theta(x_i,t_i)
\right]\Delta t
+
g(t_i)\sqrt{|\Delta t|}\,z_i
$$

con:

$$
z_i \sim \mathcal{N}(0,I)
$$

Este método combina una corrección determinista, guiada por el score, con un término aleatorio que mantiene la naturaleza estocástica del proceso.

### 6.2. Predictor-Corrector

Los *samplers* Predictor-Corrector añaden una segunda idea. El predictor avanza en el tiempo inverso, por ejemplo con Euler-Maruyama. El corrector, en cambio, refina la muestra en el mismo nivel de ruido mediante pasos inspirados en dinámica de Langevin.

El corrector puede verse como una forma de decir: “antes de pasar al siguiente nivel de ruido, ajustemos un poco más la muestra para que sea más probable bajo $p_t$”. Esto suele mejorar la calidad visual, porque no solo se avanza hacia menos ruido, sino que se corrige la posición dentro de cada distribución intermedia.

### 6.3. Probability Flow ODE

El marco SDE tiene una propiedad especialmente interesante: existe una ODE determinista cuyas trayectorias comparten las mismas distribuciones marginales que la SDE. Esta *probability flow ODE* se escribe como:

$$
dx =
\left[
f(x,t)
-
\frac{1}{2}g(t)^2\nabla_x \log p_t(x)
\right]dt
$$

Sustituyendo el score real por el aprendido:

$$
dx =
\left[
f(x,t)
-
\frac{1}{2}g(t)^2s_\theta(x,t)
\right]dt
$$

**Cuarto hallazgo importante:** el mismo modelo de score permite dos formas de generar. Una estocástica, mediante la SDE inversa, y otra determinista, mediante la ODE de flujo de probabilidad. La primera conserva ruido durante el muestreo; la segunda transforma una condición inicial en una trayectoria única.

La ODE también permite estimar verosimilitudes mediante la fórmula instantánea de cambio de variables [^chen2018neuralode]. Si $p_t(x(t))$ evoluciona siguiendo una ODE $dx/dt = v_t(x)$, entonces:

$$
\frac{d \log p_t(x(t))}{dt}
=
-\nabla_x \cdot v_t(x(t))
$$

En difusión basada en SDEs, esto permite calcular métricas como la NLL o los *Bits Per Dimension* (BPD), algo que no todos los modelos generativos ofrecen de forma natural.

---

## 7. Generación condicional: dirigir el proceso hacia una clase

La generación incondicional produce imágenes sin especificar una categoría. Para condicionar el proceso a una clase $y$, se puede utilizar un clasificador auxiliar dependiente del tiempo $p_\phi(y\mid x(t),t)$.

La derivación parte de Bayes:

$$
p_t(x\mid y)
=
\frac{p_t(y\mid x)p_t(x)}{p_t(y)}
$$

Tomamos logaritmos:

$$
\log p_t(x\mid y)
=
\log p_t(y\mid x)
+
\log p_t(x)
-
\log p_t(y)
$$

Derivamos respecto a $x$:

$$
\nabla_x \log p_t(x\mid y)
=
\nabla_x \log p_t(x)
+
\nabla_x \log p_t(y\mid x)
$$

porque $p_t(y)$ no depende de $x$.

Esta ecuación tiene una interpretación muy clara: el score condicional es el score incondicional más una fuerza adicional que empuja la imagen hacia la clase deseada.

En la implementación se aproxima como:

$$
\nabla_x \log p_t(x\mid y)
\approx
s_\theta(x,t)
+
w \nabla_x \log p_\phi(y\mid x,t)
$$

donde $w$ controla la intensidad de la guía. Así, la SDE inversa condicional queda:

$$
dx =
\left[
f(x,t)
-
g(t)^2
\left(
s_\theta(x,t)
+
w\nabla_x \log p_\phi(y\mid x,t)
\right)
\right]dt
+
g(t)d\bar{W}(t)
$$

En términos prácticos, el modelo de score aporta conocimiento general sobre cómo son las imágenes, mientras que el clasificador introduce una preferencia semántica: “quiero que esta imagen acabe pareciéndose a un avión, un perro, un barco o una rana”.

---

## 8. Imputación: generar solo lo que falta

La misma idea también puede utilizarse para imputar regiones ocultas de una imagen. Sea $m$ una máscara binaria, donde $m=1$ indica píxeles conocidos y $m=0$ indica zonas a reconstruir.

Durante el muestreo inverso, el modelo propone una imagen completa. Sin embargo, en cada paso se reemplaza la parte conocida por una versión perturbada de la imagen original al mismo nivel de ruido. La actualización puede expresarse como:

$$
x'_{t_{i-1}}
=
m \odot \tilde{x}_{t_{i-1}}
+
(1-m)\odot x_{t_{i-1}}
$$

donde:

- $\tilde{x}_{t_{i-1}}$ es la imagen original perturbada mediante el proceso directo hasta el instante $t_{i-1}$.
- $x_{t_{i-1}}$ es la muestra generada por el modelo.
- $\odot$ denota producto elemento a elemento.

La interpretación es sencilla: el modelo tiene libertad para inventar las regiones desconocidas, pero debe respetar la información observada. Esta estrategia convierte un modelo generativo incondicional en una herramienta de reconstrucción condicionada por contexto.

---

## 9. Implementación: teoría traducida a componentes

La implementación se organizó para que cada concepto matemático tuviera una representación directa en código. Las SDEs se implementaron como objetos capaces de proporcionar sus coeficientes de deriva y difusión, sus distribuciones de transición y las operaciones necesarias para entrenar el score.

Los elementos principales fueron:

- Familias de SDEs: VE-SDE, VP-SDE y Sub-VP SDE.
- Schedules de ruido lineales y cosenoidales.
- Modelos de score basados en U-Net.
- Samplers: Euler-Maruyama, Predictor-Corrector, Probability Flow ODE e integradores exponenciales.
- Métricas: FID, Inception Score, NLL y BPD.
- Extensiones para generación condicional e imputación.

La arquitectura U-Net resulta especialmente adecuada porque combina información local y global. En imágenes, las estructuras pequeñas —bordes, texturas, contornos— deben convivir con relaciones de mayor escala, como forma, objeto y fondo. Las conexiones *skip* permiten recuperar detalle espacial mientras las capas profundas capturan contexto.

---

## 10. Resultados: qué se observa al generar imágenes

La experimentación se realizó principalmente con CIFAR-10 y MNIST. Para ilustrar el comportamiento del sistema, se muestran resultados representativos con una VP-SDE lineal y distintos métodos de muestreo.

<figure id="fig:evolucion_muestras_vp_lineal">
<img src="/assets/articles/difussion_models/evolucion_muestras_vp_lineal.png" />
<figcaption>Evolución de la generación con VP-SDE lineal y sampler Predictor-Corrector sobre CIFAR-10. De izquierda a derecha, la muestra evoluciona desde ruido inicial hasta una imagen final.</figcaption>
</figure>

La secuencia muestra visualmente la idea central del método: la generación no aparece de golpe. Primero se forman masas de color, después texturas, más tarde estructuras reconocibles y finalmente detalles más definidos. Este comportamiento encaja con la interpretación probabilística: al principio se resuelven distribuciones muy ruidosas y suaves; al final se trabaja en niveles de ruido bajos, donde importan los detalles finos.

<figure id="fig:comparativa_samplers_vp_lineal">
<img src="/assets/articles/difussion_models/comparativa_samplers_vp_lineal.png" />
<figcaption>Comparación de imágenes finales generadas con distintos samplers: Predictor-Corrector, Exponential Integrator, Probability Flow ODE y Euler-Maruyama.</figcaption>
</figure>

En las comparativas, Predictor-Corrector tiende a producir resultados visualmente más estables, porque combina avance temporal y refinamiento local. Euler-Maruyama es más simple, pero suele requerir más pasos. La Probability Flow ODE ofrece una trayectoria determinista y resulta especialmente relevante cuando interesa estimar verosimilitudes.

---

## 11. Métricas: FID, IS y BPD no cuentan la misma historia

La evaluación cuantitativa se realizó con métricas habituales en modelos generativos: Fréchet Inception Distance (FID), Inception Score (IS) y Bits Per Dimension (BPD).

<figure id="fig:metricas_fid_bpd">
<div class="minipage">
<img src="/assets/articles/difussion_models/fid.jpeg" />
</div>
<div class="minipage">
<img src="/assets/articles/difussion_models/bpd.jpeg" />
</div>
<figcaption>Distribución de resultados para FID y BPD entre diferentes configuraciones SDE.</figcaption>
</figure>

<figure id="fig:metricas_is">
<div class="minipage">
<img src="/assets/articles/difussion_models/is_media.jpeg" />
</div>
<div class="minipage">
<img src="/assets/articles/difussion_models/is_std.jpeg" />
</div>
<figcaption>Distribución del Inception Score y su desviación estándar interna.</figcaption>
</figure>

Los resultados muestran un punto importante: no todas las métricas premian lo mismo. En las ejecuciones realizadas, SubVP-SDE lineal obtuvo los mejores resultados perceptuales relativos, con menor FID y mayor IS. En cambio, las variantes con schedule cosenoidal destacaron en BPD, especialmente SubVP-SDE cosenoidal.

Esta diferencia es relevante porque BPD mide ajuste probabilístico, mientras que FID e IS se acercan más a calidad perceptual y diversidad semántica. Un modelo puede asignar buena probabilidad a los datos y, aun así, no producir las muestras visualmente más convincentes. Este desacoplamiento entre verosimilitud y calidad perceptual es una de las lecciones más importantes al evaluar modelos generativos.

**Quinto hallazgo importante:** “mejor” modelo depende de qué significa “mejor”. Si buscamos imágenes visualmente convincentes, FID e IS pueden ser más informativos. Si buscamos modelado probabilístico y verosimilitud, BPD aporta una lectura diferente. La evaluación debe mirar varias métricas, no una sola.

---

## 12. Generación condicional

Para la generación condicional se utilizó un clasificador dependiente del tiempo, basado en una arquitectura tipo Wide ResNet, entrenado para reconocer clases incluso cuando la imagen está perturbada por ruido.

<figure id="fig:classifier_arch">
<img src="/assets/articles/difussion_models/classifier_architecture_image.png" style="width:70.0%" />
<figcaption>Arquitectura esquemática del clasificador dependiente del tiempo utilizado para guiar la generación condicional.</figcaption>
</figure>

El clasificador aporta el término $\nabla_x \log p_\phi(y\mid x,t)$, que modifica el score efectivo durante el muestreo. Así, la generación deja de ser completamente libre y se orienta hacia una clase concreta.

<figure id="fig:conditional_comparison">
<img src="/assets/articles/difussion_models/final_conditional_comparison.png" />
<figcaption>Comparación de resultados en generación condicional para distintas configuraciones.</figcaption>
</figure>

La generación condicional mostró que el marco SDE no solo sirve para producir imágenes aleatorias, sino también para controlar parcialmente el resultado. El grado de control depende del clasificador, de la escala de guiado y de la calidad del modelo de score. Una escala demasiado baja puede no orientar suficientemente la muestra; una escala demasiado alta puede forzar artefactos o reducir diversidad.

---

## 13. Imputación de imágenes

En la tarea de imputación se aplicaron máscaras derivadas de dígitos MNIST sobre imágenes de CIFAR-10. El modelo debía reconstruir las regiones ocultas manteniendo coherencia con el contexto visible.

<figure id="fig:imputacion_cifar_mnist_3etapas">
<img src="/assets/articles/difussion_models/imputacion_cifar_mnist_3etapas.png" />
<figcaption>Imputación de regiones en CIFAR-10 utilizando máscaras de MNIST. Arriba: imágenes enmascaradas. Centro: resultado imputado. Abajo: imagen original de referencia.</figcaption>
</figure>

La imputación es interesante porque demuestra que el modelo no solo genera desde ruido puro. También puede actuar como una distribución previa sobre imágenes naturales: cuando falta información, propone contenido compatible con lo observado. En otras palabras, el modelo no “recupera” la región exacta perdida, sino que genera una reconstrucción plausible bajo la distribución aprendida.

---

## 14. Convergencia durante el entrenamiento

También se comparó la evolución de la pérdida durante las primeras 50 épocas para diferentes configuraciones SDE.

<figure id="fig:loss_curves_violin_sdes_50epochs">
<div class="minipage">
<img src="/assets/articles/difussion_models/loss_curves_all_sdes.png" />
</div>
<div class="minipage">
<img src="/assets/articles/difussion_models/loss_violin_plot_all_sdes.png" />
</div>
<figcaption>Evolución y distribución del loss durante las primeras 50 épocas para distintas configuraciones SDE.</figcaption>
</figure>

<figure id="fig:loss_barplots_sdes_50epochs">
<div class="minipage">
<img src="/assets/articles/difussion_models/loss_final_barplot_all_sdes.png" />
</div>
<div class="minipage">
<img src="/assets/articles/difussion_models/loss_average_barplot_all_sdes.png" />
</div>
<figcaption>Comparación del loss final y del loss promedio durante el entrenamiento inicial.</figcaption>
</figure>

Las curvas sugieren que SubVP-SDE lineal y VE-SDE alcanzan valores de pérdida bajos de forma más temprana. Sin embargo, esta lectura debe interpretarse con cautela: minimizar mejor la pérdida de entrenamiento no garantiza automáticamente mejores imágenes. La pérdida mide la precisión en la estimación del score bajo el esquema de perturbación elegido, mientras que la calidad generativa depende también del sampler, del schedule de ruido, de la arquitectura y de cómo se propagan los errores durante la integración inversa.

---

## 15. Limitaciones

El proyecto tiene limitaciones que conviene dejar explícitas. El entrenamiento de los modelos de score y de los clasificadores dependientes del tiempo es computacionalmente costoso, lo que restringió el número de épocas y la exhaustividad de algunas comparativas. La evaluación de métricas como FID se realizó sobre subconjuntos de datos, por lo que los valores absolutos podrían variar. A esto se suma la volatilidad inherente a las métricas de evaluación de modelos generativos: ninguna métrica por sí sola captura todos los aspectos deseables de la generación.

---

## 16. Conclusión: lo que deja claro el marco SDE

La formulación mediante SDEs ofrece una forma especialmente clara de entender los modelos de difusión. El proceso directo convierte datos en ruido mediante una dinámica conocida. El proceso inverso permite generar datos, pero requiere conocer el score de las distribuciones intermedias. Como ese score no está disponible, se aproxima con una red neuronal entrenada mediante *denoising score matching*.

Lo más potente del enfoque es que muchas piezas encajan de manera natural:

- La SDE directa define cómo se corrompen los datos.
- El score indica cómo volver hacia regiones de alta probabilidad.
- La SDE inversa transforma ruido en imágenes.
- La Probability Flow ODE ofrece una alternativa determinista y permite estimar verosimilitudes.
- La guía por clasificador permite generación condicional.
- La imputación aparece al combinar muestreo inverso con restricciones sobre píxeles conocidos.

Desde el punto de vista experimental, los resultados muestran que no existe una única configuración universalmente superior. SubVP-SDE lineal se comportó mejor en métricas perceptuales como FID e IS, mientras que variantes cosenoidales destacaron en BPD. Esta diferencia refuerza una idea importante: evaluar modelos generativos exige mirar simultáneamente calidad visual, diversidad, estabilidad, coste computacional y ajuste probabilístico.

En conjunto, el trabajo permitió conectar teoría, implementación y experimentación en un mismo marco: desde la derivación matemática del score hasta la generación de imágenes, la generación condicional y la imputación.

---

## Referencias

[^song2021scorebased_sde]: Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-Based Generative Modeling through Stochastic Differential Equations. In International Conference on Learning Representations (ICLR).

[^songermon2019ncsn]: Song, Y., & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution. In Advances in Neural Information Processing Systems (NeurIPS).

[^sohl2015deep]: Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics. In International Conference on Machine Learning (ICML).

[^ho2020denoising]: Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. In Advances in Neural Information Processing Systems (NeurIPS).

[^krizhevsky2009cifar]: Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images. Technical report, University of Toronto.

[^lecun1998gradient]: LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.

[^goodfellow2014generative]: Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (NeurIPS).

[^kingma2013auto]: Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.

[^dinh2016density]: Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2016). Density estimation using Real NVP. arXiv preprint arXiv:1605.08803.

[^kingma2018glow]: Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative flow with invertible 1x1 convolutions. In Advances in neural information processing systems (NeurIPS).

[^anderson1982reverse]: Anderson, B. D. (1982). Reverse-time diffusion equation models. Stochastic Processes and their Applications, 13(3), 313-326.

[^vincent2011connection]: Vincent, P. (2011). A connection between score matching and denoising autoencoders. Neural computation, 23(7), 1661-1674.

[^chen2018neuralode]: Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018). Neural ordinary differential equations. In Advances in neural information processing systems (NeurIPS).

[^hyvarinen2005scorematching]: Hyvärinen, A. (2005). Estimation of non-normalized statistical models by score matching. Journal of Machine Learning Research, 6(Apr), 695-709.

[^nicholdhariwal2021improved]: Nichol, A. Q., & Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. In International Conference on Machine Learning (ICML).

[^ronneberger2015unet]: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (MICCAI).

[^dhariwal2021diffusion]: Dhariwal, P., & Nichol, A. (2021). Diffusion models beat gans on image synthesis. In Advances in Neural Information Processing Systems (NeurIPS).

[^zagoruyko2016wide]: Zagoruyko, S., & Komodakis, N. (2016). Wide residual networks. arXiv preprint arXiv:1605.07146.
