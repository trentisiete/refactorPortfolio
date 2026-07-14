---
title: "BGG Review Intelligence: De Reseñas a Conocimiento de Opinión Estructurado"
date: 2025-11-15
excerpt: "BGG Review Intelligence explora cómo las reseñas de BoardGameGeek pueden transformarse en predicciones de sentimiento reproducibles y conocimiento de opinión estructurado."
kind: "article"
draft: false
readingTime: 10
lang: "es"
translated: true
sourceHash: "2a1f03b5cc521ca4"
---

## Un estudio en varias etapas de preprocesamiento lingüístico, clasificación de sentimiento, aprendizaje contextual y análisis basado en aspectos

Las reseñas de juegos de mesa contienen mucho más que una simple afirmación sobre si a un jugador le gustó o no un juego. Describen mecánicas, duración, complejidad, ilustraciones, equilibrio, accesibilidad, interacción entre jugadores, rejugabilidad y las condiciones sociales bajo las cuales un juego triunfa o fracasa. Una sola reseña puede elogiar uno de estos aspectos y rechazar otro. Puede contener ironía, comparaciones, recomendaciones condicionales o un cambio de opinión tras varias partidas. Esta combinación de juicio global y experiencia detallada convierte a las reseñas de BoardGameGeek en un dominio particularmente rico para el Procesamiento del Lenguaje Natural.

Este proyecto estudia cómo ese lenguaje no estructurado puede transformarse en conocimiento de opinión reproducible e interpretable. Su desarrollo siguió una secuencia acumulativa. La primera etapa estableció una base lingüística y de procesamiento de corpus fiable. La segunda introdujo representaciones léxicas dispersas, características explícitas de opinión y modelos clásicos de aprendizaje automático. La tercera exploró embeddings de palabras estáticos y modelos neuronales de secuencia. La cuarta aplicó aprendizaje por transferencia contextual con BERT. La etapa final fue más allá de las etiquetas globales de sentimiento y examinó si un modelo de lenguaje de gran tamaño podía extraer aspectos estructurados, polaridades y evidencia de respaldo.

El resultado no es simplemente una colección de ejercicios aislados. Es un cuerpo conectado de conocimiento técnico y empírico. Cada etapa se construyó sobre las estructuras de datos, decisiones de preprocesamiento y artefactos de evaluación producidos en las etapas anteriores. El proyecto documenta, por tanto, tanto el rendimiento de los modelos como el camino metodológico necesario para obtener, interpretar y comparar ese rendimiento.

![Progresión de la investigación desde la construcción del corpus hasta la explicación estructurada de la opinión](/assets/articles/bgg_review_intelligence/research-progression.svg)

*La progresión de la investigación es acumulativa: los modelos posteriores dependen del corpus, las decisiones de preprocesamiento y los artefactos de evaluación establecidos anteriormente.*

## El problema detrás de la tarea de clasificación

A nivel de documento, el objetivo inicial es clasificar una reseña como negativa, neutral o positiva. Las etiquetas objetivo se derivan de la valoración numérica asociada a la reseña. Las valoraciones de 0 a 3 se tratan como negativas, las de 4 a 7 como neutrales, y las superiores a 7 como positivas.

```text
negative: rating <= 3
neutral:  4 <= rating <= 7
positive: rating > 7
```

Esta política crea una tarea práctica de aprendizaje supervisado y produce tres clases de igual tamaño. Aun así, es importante entender qué representa la etiqueta. El modelo aprende a predecir una categoría de polaridad derivada de la valoración a partir del lenguaje de la reseña. No aprende a partir de una anotación humana separada que evalúe directamente el sentimiento del texto.

Esa distinción importa porque las valoraciones y el lenguaje no siempre coinciden perfectamente. Un usuario puede asignar una valoración moderada mientras escribe una reseña emocionalmente negativa. Otro usuario puede describir problemas serios pero aun así recomendar el juego a un público concreto. Una reseña también puede contener aspectos positivos y negativos a la vez. El amplio intervalo neutral, que incluye las valoraciones 4, 5, 6 y 7, contiene por tanto varias situaciones lingüísticas distintas: sentimiento débil, sentimiento mixto, aprobación condicional, incertidumbre o desacuerdo entre el texto y la valoración.

El proyecto aborda en consecuencia un problema más interesante que la simple detección de polaridad. Estudia la relación entre la evaluación textual y una señal conductual externa, al tiempo que expone la ambigüedad que se introduce cuando un juicio continuo u ordinal se convierte en tres clases discretas.

## Construyendo la base lingüística y de datos

Antes de comparar clasificadores, el proyecto estableció una representación coherente de cada reseña. El texto sin procesar generado por usuarios es ruidoso. Puede contener marcado, espaciado inconsistente, Unicode mal formado, caracteres repetidos, emojis, variación ortográfica, vocabulario específico del dominio y fragmentos en distintos idiomas. Si estos elementos se procesan de forma inconsistente, las diferencias posteriores entre modelos podrían reflejar diferencias en la preparación de los datos en lugar de diferencias en la capacidad de aprendizaje.

La arquitectura de preprocesamiento se diseñó como una secuencia de transformaciones independientes que comparten un contrato común. Cada transformación recibe un documento, modifica solo los campos relacionados con su responsabilidad, y devuelve el documento enriquecido. La estructura puede resumirse así:

```python
class PreprocStep(Protocol):
    name: str
    version: str
    params: dict

    def transform(self, doc: dict) -> dict:
        ...
```

La canalización normaliza la codificación, elimina o convierte el marcado, limpia espacios en blanco y caracteres repetidos, gestiona emojis, detecta el idioma, segmenta oraciones, tokeniza palabras, filtra stopwords y registra las anotaciones resultantes. La versión más reciente utiliza spaCy como procesador lingüístico unificado para la tokenización, la lematización, las etiquetas de categoría gramatical y las relaciones de dependencia. Esta unificación es metodológicamente importante porque las dependencias sintácticas, los lemas y las posiciones de los tokens deben referirse a la misma tokenización.

El procesamiento de stopwords también se adaptó a la tarea de sentimiento. Las palabras gramaticales comunes pueden eliminarse de las representaciones léxicas, pero los términos de negación como "not", "no" y "never" deben permanecer disponibles porque pueden invertir la interpretación de una expresión evaluativa. La puntuación se conserva cuando es necesaria para el análisis lingüístico y se excluye de la representación léxica final cuando solo añadiría ruido.

Los documentos procesados se almacenan en un corpus JSONL estructurado, acompañado de metadatos, índices, estadísticas e información de esquema. Los identificadores estables de reseña permiten recuperar registros individuales, mientras que el lector del corpus admite tanto el acceso en flujo como el acceso dirigido. Cada documento puede conservar el texto sin procesar, el texto limpio, las anotaciones lingüísticas, las características derivadas, las entradas vectoriales e información sobre la configuración de la canalización que lo generó.

Esta capa de corpus es una de las contribuciones centrales del proyecto. Separa el acceso a los datos de la lógica del modelo y permite aplicar diferentes representaciones a los mismos documentos subyacentes. También crea un camino trazable desde un resultado experimental de vuelta al texto, la configuración de preprocesamiento y la partición de la que se obtuvo ese resultado.

## Composición del corpus y particiones experimentales

Tras filtrar las reseñas en inglés con texto utilizable y valoraciones numéricas válidas, el corpus P2 contiene 526.869 reseñas. El procedimiento de balanceo de clases produce exactamente 175.623 reseñas en cada categoría de polaridad.

El corpus se divide mediante un procedimiento estratificado con una semilla aleatoria fija de 42. La partición de entrenamiento contiene 368.808 reseñas, la partición de validación contiene 52.687 reseñas, y la partición de prueba reservada contiene 105.374 reseñas. Las proporciones de clase permanecen prácticamente idénticas en cada partición.

| Partición | Reseñas totales | Negativa | Neutral | Positiva |
|---|---:|---:|---:|---:|
| Entrenamiento | 368.808 | 122.936 | 122.936 | 122.936 |
| Validación | 52.687 | 17.563 | 17.562 | 17.562 |
| Prueba | 105.374 | 35.124 | 35.125 | 35.125 |
| **Corpus completo** | **526.869** | **175.623** | **175.623** | **175.623** |

La partición de validación fija se utiliza para la selección de modelos e hiperparámetros. La partición de prueba permanece separada hasta que se ha seleccionado una configuración final. Esta distinción evita que el conjunto de prueba forme parte del proceso de optimización y otorga a los valores finales una interpretación clara como rendimiento en datos reservados.

Dado que el corpus está perfectamente equilibrado, la exactitud (accuracy) resulta más fácil de interpretar de lo que lo sería en un conjunto de datos muy desequilibrado. Aun así, el F1 macro sigue siendo la medida de evaluación principal porque asigna la misma importancia al rendimiento en las clases negativa, neutral y positiva. Se conservan la precisión, la exhaustividad, el F1 por clase y las matrices de confusión para que el rendimiento agregado no oculte errores sistemáticos.

## Representando la opinión con información léxica y lingüística

La etapa de modelado clásico crea dos entradas complementarias a partir de cada reseña procesada. La primera, `tfidf_input`, contiene una versión lematizada del texto sin stopwords. Se utiliza para construir vectores TF-IDF con n-gramas de palabras. La segunda, `opinion_input`, es un diccionario disperso de información lingüística explícita relacionada con la polaridad.

La representación de opinión captura fenómenos como elementos léxicos positivos y negativos, negación, intensificación, atenuación y patrones evaluativos relacionados con el dominio. Estas características intentan representar conocimiento que una bolsa de palabras convencional no codifica de forma explícita. Una frase como "not enjoyable", por ejemplo, contiene una interacción entre una negación y un término evaluativo positivo. Una representación lingüística puede registrar esa interacción directamente, mientras que una representación léxica debe inferirla a partir del n-grama observado.

El `CombinedVectorizer` permite que la misma canalización experimental funcione en tres modos. El modo TF-IDF utiliza solo n-gramas léxicos, el modo de opinión utiliza solo el diccionario de características explícitas, y el modo combinado concatena ambas matrices dispersas.

```python
vectorizer = CombinedVectorizer(
    mode="combined",
    tfidf_params={
        "ngram_range": (1, 2),
        "max_features": 20_000,
        "min_df": 10,
        "sublinear_tf": True,
    },
)

X_train = vectorizer.fit_transform(train_documents)
X_validation = vectorizer.transform(validation_documents)
X_test = vectorizer.transform(test_documents)
```

El vectorizador se ajusta únicamente sobre los documentos de entrenamiento. Los documentos de validación y prueba se transforman utilizando el vocabulario y los valores de frecuencia inversa de documento aprendidos a partir de los datos de entrenamiento. Esto evita que la información léxica de las particiones reservadas influya en la representación.

TF-IDF sigue siendo una representación potente para este dominio. Los unigramas capturan términos evaluativos individuales, mientras que los bigramas y n-gramas más largos capturan patrones contextuales breves como la negación, recomendaciones o expresiones habituales de insatisfacción. La frecuencia de término sublineal reduce la influencia de las palabras repetidas, y `min_df` elimina características extremadamente raras que pueden representar ruido en lugar de evidencia estable.

La representación explícita de opinión cumple un propósito distinto. Sus dimensiones son menos numerosas y más interpretables, pero inevitablemente simplifica el lenguaje mediante categorías y recursos léxicos diseñados manualmente. La comparación entre TF-IDF, las características de opinión y su combinación plantea, por tanto, si el conocimiento lingüístico estructurado aporta información más allá de un espacio léxico amplio.

## Experimentos de aprendizaje automático clásico

Se evaluaron cuatro familias de modelos en los tres modos de representación. Naive Bayes proporciona una base probabilística eficiente para texto disperso. Una Máquina de Vectores de Soporte (SVM) lineal aprende fronteras de decisión de margen máximo en el espacio léxico de alta dimensionalidad. Random Forest introduce interacciones no lineales mediante un conjunto de árboles de decisión. XGBoost utiliza árboles regularizados construidos secuencialmente para modelar relaciones de características más complejas.

Las configuraciones candidatas se entrenan sobre la partición de entrenamiento y se clasifican según el F1 macro de validación. Una vez identificada la configuración más sólida dentro de una familia de modelos, se reentrena utilizando los datos combinados de entrenamiento y validación, y se evalúa una sola vez sobre el conjunto de prueba. Las predicciones, los informes por clase, las matrices de confusión, los vectorizadores, los modelos y los artefactos de resumen se guardan junto con el experimento.

Los valores exactos de prueba que se muestran a continuación fueron verificados frente a los informes de evaluación almacenados.

| Modelo | Representación ganadora | Exactitud | F1 macro | F1 negativo | F1 neutral | F1 positivo |
|---|---|---:|---:|---:|---:|---:|
| SVM lineal | TF-IDF | **0,7333** | **0,7309** | **0,8040** | **0,6291** | **0,7595** |
| Naive Bayes multinomial | TF-IDF | 0,7277 | 0,7254 | 0,7974 | 0,6237 | 0,7552 |
| XGBoost | Combinada | 0,6904 | 0,6871 | 0,7537 | 0,5894 | 0,7182 |
| Random Forest | TF-IDF | 0,6439 | 0,6346 | 0,7169 | 0,5085 | 0,6785 |

La configuración clásica más sólida es una SVM lineal operando sobre vectores TF-IDF compuestos de unigramas y bigramas. Su vocabulario está limitado a 20.000 características, los términos deben aparecer en al menos diez documentos, la frecuencia de término sublineal está activada, y el parámetro de regularización de la SVM es `C=1.0`.

```python
tfidf_params = {
    "ngram_range": (1, 2),
    "max_features": 20_000,
    "min_df": 10,
    "sublinear_tf": True,
}

model = LinearSVC(C=1.0, class_weight=None)
```

El resultado demuestra la fortaleza del modelado lineal disperso para colecciones de texto de gran tamaño. La SVM se sitúa solo ligeramente por delante de Naive Bayes multinomial, con una diferencia de aproximadamente 0,0055 de F1 macro. Ambos modelos lineales superan a las familias basadas en árboles. En un espacio disperso de alta dimensionalidad, una frontera lineal puede aprovechar de forma eficiente la evidencia acumulada de muchas dimensiones léxicas. Los conjuntos de árboles deben particionar ese espacio mediante decisiones sucesivas y pueden tener dificultades para representar la misma geometría sin una complejidad considerablemente mayor.

XGBoost obtiene su resultado verificado más sólido con la representación combinada, lo que indica que las características explícitas de opinión aportan información de alto nivel útil para los árboles potenciados (boosted trees). Sin embargo, su F1 macro final permanece por debajo de los dos modelos lineales. Random Forest produce el resultado más débil y muestra un descenso particularmente acusado en la clase neutral.

Las puntuaciones específicas por clase revelan un patrón que la exactitud agregada por sí sola no puede expresar. Las reseñas negativas son sistemáticamente las más fáciles de identificar. Las reseñas positivas también logran resultados relativamente sólidos. Las reseñas neutrales siguen siendo sustancialmente más difíciles para todos los modelos. Incluso la SVM ganadora obtiene un F1 neutral de 0,6291, frente a 0,8040 para las reseñas negativas y 0,7595 para las positivas.

Esto no es simplemente una consecuencia del desequilibrio de clases, porque el conjunto de datos está equilibrado. La dificultad pertenece a la estructura semántica de las etiquetas y del lenguaje. Las reseñas cercanas al centro de la escala de valoración pueden incluir opiniones débiles, aspectos contradictorios, recomendaciones condicionales o un lenguaje que se asemeja a una de las clases extremas incluso cuando la valoración global es moderada.

## Embeddings estáticos y arquitecturas neuronales

La siguiente etapa examina si las representaciones densas preentrenadas de palabras pueden capturar relaciones semánticas que las características léxicas dispersas tratan como dimensiones independientes. Se compararon Word2Vec GoogleNews, embeddings de FastText crawl y embeddings GloVe Dolma, usando vectores de 300 dimensiones.

Se utilizaron dos arquitecturas para exponer las consecuencias de la representación de documentos. La red feedforward recibe un vector por reseña. Ese vector se obtiene promediando los embeddings de todas las palabras del documento que están en el vocabulario.

```text
document_vector = (1 / n) * sum(word_vector_i)
```

El mean pooling es eficiente y produce una representación semántica de tamaño fijo independientemente de la longitud del documento. Su limitación es igualmente clara: elimina el orden de las palabras y asigna la misma importancia estructural a cada token incluido. “El juego es bueno pero demasiado largo” y una reordenación de las mismas palabras pueden producir promedios casi idénticos, aunque el orden discursivo y el contraste afecten la interpretación.

La LSTM recibe la secuencia de embeddings de palabras en lugar de su promedio. Las reseñas se rellenan o truncan a una longitud máxima fija, y la red recurrente procesa los embeddings en orden. Esto permite que el modelo conserve relaciones locales, cambios de tono, negación y otra información secuencial que desaparece durante el mean pooling.

FastText proporcionó la representación base más sólida en las comparaciones reportadas. Su uso de información de subpalabras a nivel de caracteres le da una ventaja con palabras raras, variantes morfológicas, diferencias ortográficas y vocabulario especializado de juegos de mesa. Los modelos finales de FastText usaron una red feedforward con una dimensión oculta de 512 y una LSTM con una dimensión oculta de 128.

| Modelo | Representación | Accuracy | Macro-F1 | F1 Negativo | F1 Neutral | F1 Positivo |
|---|---|---:|---:|---:|---:|---:|
| FastText + FNN | Embedding de documento con mean-pooling | 0.697 | 0.695 | 0.770 | 0.590 | 0.726 |
| FastText + LSTM | Secuencia ordenada de embeddings de palabras | **0.752** | **0.752** | **0.823** | **0.657** | **0.776** |

La LSTM mejora el macro-F1 en aproximadamente 0.057 y produce ganancias en las tres clases. El F1 neutral sube de 0.590 a 0.657. Esta comparación aporta evidencia directa de que el orden de las palabras y la composición secuencial contienen información que se pierde cuando los vectores estáticos de palabras se reducen a un promedio.

El resultado no vuelve obsoletos a los modelos dispersos anteriores. El SVM lineal sigue siendo computacionalmente eficiente, reproducible y muy competitivo. En cambio, la comparación neuronal aclara qué información adicional puede recuperar una representación secuencial y cómo afecta esa información a la clase que resulta más difícil para los modelos más simples.

## Transferencia de aprendizaje contextual con BERT

Los embeddings estáticos asignan un vector fijo a una palabra sin importar su uso. BERT reemplaza esa suposición con representaciones contextuales. El vector asociado a un token depende de la secuencia que lo rodea, lo que permite al modelo distinguir significados y funciones que un embedding fijo fusiona en uno solo.

El proyecto ajusta (fine-tunes) `bert-base-uncased` con una cabeza de clasificación de secuencias de tres clases. El texto crudo de las reseñas se procesa con tokenización WordPiece. Se añaden tokens especiales, las secuencias se rellenan o truncan a 128 tokens, y una máscara de atención distingue los tokens reales del relleno. Tanto la capa de clasificación como el transformer preentrenado se actualizan durante el ajuste.

```python
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5

model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
)
```

El experimento con BERT usó un corpus balanceado separado de aproximadamente 60.000 reseñas debido a las limitaciones computacionales del entorno de entrenamiento de Colab. También usó texto crudo en lugar de la entrada lematizada y filtrada de stopwords que usaron varios modelos anteriores. Estas diferencias son esenciales al interpretar las puntuaciones.

Después de la primera época, el modelo alcanzó una accuracy de validación de 0.8327 y un macro-F1 de validación de 0.8304. Después de la segunda época, la accuracy de validación alcanzó 0.8464 y el macro-F1 alcanzó 0.8468. Los resultados reportados sobre el conjunto de retención (held-out) fueron accuracy de 0.8431, macro-F1 de 0.8432, F1 ponderado de 0.8432, y pérdida de evaluación de 0.4420.

| Medida de evaluación de BERT | Valor |
|---|---:|
| Accuracy de test | 0.8431 |
| Macro-F1 de test | 0.8432 |
| F1 ponderado de test | 0.8432 |
| Pérdida de evaluación de test | 0.4420 |

Estos resultados aportan evidencia sólida de que la transferencia de aprendizaje contextual es muy adecuada para el dominio. BERT puede relacionar cada token con su contexto circundante y preservar palabras funcionales, puntuación, formas completas de las palabras y señales discursivas que se reducen o eliminan en el pipeline clásico.

La puntuación de BERT no se presenta como una victoria numérica directa sobre el SVM del corpus completo o la LSTM, porque los modelos no se evaluaron sobre el mismo benchmark. El resultado establece el valor del enfoque contextual dentro de su propio entorno documentado. También muestra cómo el conocimiento lingüístico preentrenado puede lograr un alto rendimiento tras solo dos épocas de ajuste sobre un corpus mucho más pequeño que la colección P2 completa.

![Resultados de macro-F1 reportados separados por entorno experimental comparable](/assets/articles/bgg_review_intelligence/model-performance-overview.svg)

*Los modelos clásicos comparten un conjunto de test controlado. Los resultados neuronales y contextuales conservan los entornos reportados en el registro P3, por lo que los paneles se separan intencionalmente en lugar de presentarse como una única tabla comparativa.*

## De la polaridad global a la comprensión a nivel de aspecto

La clasificación de documentos comprime una reseña entera en una sola etiqueta. Esa respuesta es útil para la medición a gran escala, pero no puede explicar qué partes del juego produjeron el juicio. Por eso, la última etapa del proyecto cambia la estructura de la salida en lugar de simplemente sustituir un clasificador por otro.

El experimento basado en aspectos usa un modelo de lenguaje grande para analizar veinte reseñas seleccionadas de RoboRally. Las reseñas se cargan mediante el lector de corpus existente y se insertan en un prompt estructurado. En lugar de solicitar un resumen sin restricciones, el prompt define una estructura JSON que contiene el nombre del juego, una conclusión agregada, una lista de aspectos, una polaridad para cada aspecto y evidencia literal de apoyo.

```json
{
  "juego": "Game name",
  "conclusion": "Aggregate sentiment",
  "aspectos": [
    {
      "aspecto": "Duration / Mechanics / Art / ...",
      "polaridad": "Positive / Negative / Mixed",
      "evidencia_i": "Exact supporting review text"
    }
  ]
}
```

El análisis generado caracterizó a las reseñas seleccionadas como abrumadoramente negativas y organizó sus críticas en categorías coherentes. Los aspectos extraídos incluyeron duración, aleatoriedad y caos, eliminación de jugadores, accesibilidad, interacción, balance, agencia del jugador y experiencia general. Se asociaron cadenas de evidencia a las conclusiones, de modo que la salida estructurada se mantuvo conectada al lenguaje de origen.

Este experimento demuestra un tipo de capacidad distinto al de los clasificadores de sentimiento. BERT, LSTM y SVM estiman una clase global con un rendimiento predictivo medible. El modelo de lenguaje guiado por prompt descompone las opiniones en unidades interpretables y explica por qué los reseñadores seleccionados reaccionaron negativamente.

El resultado de aspectos es cualitativo, no una puntuación de benchmark. Las veinte reseñas pertenecen a una secuencia negativa deliberadamente seleccionada, y no se usó ningún conjunto de datos de aspectos anotado de forma independiente. El experimento demuestra, por tanto, viabilidad y valor expresivo, no una precisión medida de extracción de aspectos. Su importancia dentro del proyecto es conceptual: conecta la infraestructura escalable del corpus con una representación de la opinión que refleja más de cerca la estructura interna de una reseña.

## Qué muestra la evidencia combinada

La secuencia completa de experimentos revela varios hallazgos consistentes. El primero es que un diseño cuidadoso de datos y de lenguaje sigue siendo importante incluso cuando se dispone de modelos sofisticados. Identificadores estables, particiones deterministas, preprocesamiento trazable, predicciones guardadas y evaluación por clase son lo que hace interpretables los resultados de un modelo. Sin este fundamento, una puntuación más alta no puede separarse de diferencias en la preparación de los datos o en las condiciones de evaluación.

El segundo hallazgo es la fortaleza continua de las líneas base léxicas. TF-IDF con un SVM lineal alcanza un macro-F1 de 0.7309 sobre más de cien mil reseñas de retención. Multinomial Naive Bayes rinde solo ligeramente por debajo. Estos resultados muestran que una gran proporción de la información de sentimiento en el corpus se expresa mediante patrones léxicos recurrentes que pueden capturarse sin una arquitectura profunda.

El tercer hallazgo es que el orden y el contexto añaden información medible. La LSTM mejora sustancialmente respecto a un modelo feedforward con mean-pooling usando la misma base de FastText. El experimento con BERT alcanza una puntuación aún más alta dentro de su entorno separado, mostrando el valor de las representaciones contextualizadas y de la transferencia de aprendizaje.

El cuarto hallazgo, y el más persistente, es la dificultad de la neutralidad. Este patrón aparece en los modelos clásicos y en las redes neuronales. Las clases negativa y positiva tienen fronteras léxicas más claras, mientras que la clase neutral absorbe evaluaciones mixtas, sentimiento débil, valoraciones límite y desacuerdo entre texto y calificación. La neutralidad en este corpus no es, por tanto, simplemente un estado emocional intermedio. Es una región heterogénea creada por el lenguaje, el comportamiento del usuario y la construcción de etiquetas elegida.

El quinto hallazgo es que las características lingüísticas explícitas de opinión tienen un valor condicional más que universal. Añaden información interpretable sobre negación, intensificación y expresiones del dominio, pero los modelos léxicos de alta dimensionalidad ya codifican muchos de los mismos patrones a través de n-gramas. Su efecto depende de la familia de modelos y de la geometría de la representación. XGBoost obtiene su configuración verificada más sólida a partir de la entrada combinada, mientras que los modelos más sólidos de SVM y Naive Bayes usan solo TF-IDF.

Por último, el proyecto establece que la predicción y la explicación deben tratarse como salidas diferentes. Un clasificador puede estimar la polaridad global de forma eficiente sobre cientos de miles de documentos. Un sistema basado en aspectos puede exponer los temas y la evidencia que subyacen a esa polaridad. Juntos, estos niveles proporcionan una representación de la opinión más completa que cualquiera de los dos por separado.

## Lo que aprendimos, lo que sigue siendo incierto y hacia dónde lleva el trabajo

La lección más importante de este proceso es que la calidad de un resultado de NLP no puede separarse de la estructura del corpus que lo produjo. Al principio, era tentador pensar en el preprocesamiento como una tarea preliminar que había que completar antes de que pudiera comenzar el modelado "real". Los experimentos posteriores cambiaron esa visión. Las decisiones sobre filtrado de idioma, tokenización, lematización, stopwords, puntuación, umbrales de etiquetas y divisiones de documentos definen qué pueden observar los modelos y qué significan sus puntuaciones. El corpus no es una entrada pasiva para la investigación; es parte del propio objeto de investigación.

También aprendimos que la complejidad no invalida la simplicidad. El SVM lineal y el Multinomial Naive Bayes siguen siendo sólidos incluso después de que el proyecto introduce arquitecturas recurrentes y contextuales. Su rendimiento muestra que los patrones léxicos repetidos contienen una gran cantidad de información sobre la polaridad derivada de las valoraciones. Esto proporciona un punto de referencia importante para interpretar cada modelo más complejo. Una arquitectura más profunda es científicamente interesante cuando captura información que la línea base no puede representar, no simplemente porque sea más nueva.

La comparación entre la red feedforward y la LSTM hizo visible esta distinción. Ambos modelos partieron de embeddings estáticos de FastText, pero solo la LSTM preservó el orden de las palabras. Su mejora, especialmente en la clase neutral, sugiere que parte de la información faltante es composicional. La negación, el contraste, la matización y los cambios de tono no pueden reducirse completamente a un promedio de significados de palabras. BERT desarrolla la misma idea aún más al permitir que la representación de cada token dependa de su contexto completo.

Al mismo tiempo, el resultado de BERT creó una de las principales dudas que persisten en el proyecto. Su macro-F1 reportado es el más alto, pero se obtuvo en un corpus más pequeño, preparado por separado, y con texto sin procesar en lugar de la entrada léxica procesada que usaron varios modelos anteriores. Podemos decir que el aprendizaje por transferencia contextual funcionó muy bien en ese entorno. Todavía no podemos decir exactamente cuánto de la diferencia proviene de la arquitectura, del subconjunto del corpus, del texto de entrada o de las condiciones de entrenamiento. Esta incertidumbre no es una debilidad que ocultar; es parte de lo que el registro experimental nos ha enseñado sobre la comparación justa de modelos.

El camino natural de investigación que se deriva de esta observación comienza con la comparabilidad. Los experimentos existentes ya han identificado los modelos que vale la pena comparar. Lo que ahora falta es un único benchmark fijo en el que el modelo disperso más fuerte, el modelo recurrente y el modelo contextual observen las mismas reseñas y se evalúen mediante el mismo procedimiento. Tal comparación transformaría la progresión actual, de un conjunto de resultados sólidos específicos de cada etapa, en un relato controlado de lo que aporta cada representación.

La clase neutral plantea una duda más profunda. Su dificultad persiste en algoritmos que hacen suposiciones muy diferentes. Esto sugiere que el problema no pertenece únicamente a los clasificadores. También puede pertenecer a la propia definición de neutralidad. Las valoraciones de 4 a 7 cubren un intervalo amplio, y las reseñas asociadas pueden expresar decepción, aprobación moderada, aspectos mixtos, incertidumbre o recomendación condicional. Llamar a todos estos casos "neutrales" crea una categoría conveniente, pero quizás no un único fenómeno lingüístico.

Esto lleva a una pregunta que las etiquetas automáticas actuales no pueden responder: cuando un modelo discrepa de la clase derivada de la valoración, ¿está el modelo equivocado, o ha identificado en el texto un sentimiento que el umbral de valoración no representa? La respuesta requeriría leer y anotar de forma independiente un grupo cuidadosamente seleccionado de reseñas, en particular las cercanas a los límites de valoración y aquellas en las que los modelos discrepan. El desacuerdo humano también sería informativo. Si los lectores no asignan de forma consistente un sentimiento a estas reseñas, la incertidumbre debería tratarse como parte de los datos y no como un error que se espera que un clasificador elimine.

El experimento basado en aspectos hace esta pregunta aún más significativa. Una vez que una reseña se descompone en duración, interacción, accesibilidad, balance o agencia del jugador, el sentimiento mixto se vuelve más fácil de entender. Una reseña no necesita ser globalmente neutral en el sentido ordinario; puede ser fuertemente positiva en un aspecto y fuertemente negativa en otro. La valoración global representa entonces una agregación hecha por el reseñador, mientras que un modelo de aspectos intenta recuperar los componentes de esa agregación.

Esto cambió nuestra comprensión del objetivo final. Al principio, la extracción de aspectos parecía ser una capa explicativa colocada después de la clasificación. El experimento sugiere que también puede ofrecer una mejor representación del propio problema. El sentimiento global y el sentimiento por aspectos no son salidas en competencia. La etiqueta global proporciona escala y comparabilidad, mientras que los aspectos preservan la estructura interna de la opinión. La relación entre ambos puede ser más informativa que cualquiera de los dos resultados considerados de forma aislada.

El experimento con el LLM también deja una duda metodológica importante. Su salida es coherente y está respaldada por frases reconocibles de las reseñas seleccionadas, pero la coherencia no es lo mismo que la fidelidad medida. Un modelo generativo puede organizar el lenguaje de manera convincente incluso cuando un aspecto está incompleto, una frase de evidencia está alterada, o una conclusión es más fuerte de lo que la fuente permite. El experimento nos enseñó por tanto a separar la legibilidad de la validez. El JSON estructurado y la evidencia citada hacen que el resultado sea inspeccionable, pero una evaluación real depende de la comparación con aspectos anotados por humanos y fragmentos de fuente exactos.

La hoja de ruta que surge de estas lecciones no es una secuencia de adiciones de software. Es una secuencia de preguntas. Un benchmark común aclara la contribución de cada representación. La lectura humana de reseñas difíciles aclara qué significa la neutralidad. La anotación de aspectos aclara cómo se componen los juicios globales. La evaluación basada en evidencia aclara si las explicaciones generativas son fieles. Solo después de comprender esas relaciones cobra sentido estudiar cuán bien se transfieren las conclusiones a otros juegos, períodos de tiempo, idiomas o dominios de reseñas.

También existe incertidumbre sobre qué aprenden los modelos de la identidad del dominio. Una división aleatoria a nivel de reseña puede permitir que vocabulario similar de los mismos juegos o los mismos tipos de usuarios aparezca tanto en los datos de entrenamiento como en los de prueba. Un rendimiento sólido puede por tanto reflejar una comprensión genuina del sentimiento, familiaridad con vocabularios de juegos particulares, o una combinación de ambos. Los resultados actuales describen la generalización a reseñas no vistas del mismo corpus amplio. Todavía no describen la generalización a juegos completamente desconocidos, comunidades nuevas o períodos posteriores.

Estas dudas no disminuyen el proyecto. Son evidencia de que el trabajo ha alcanzado una etapa en la que las preguntas interesantes ya no se limitan a elegir un algoritmo. Los experimentos han expuesto relaciones entre valoraciones, lenguaje, contexto, aspectos y evidencia. También han mostrado dónde el rendimiento numérico es suficiente y dónde deja el fenómeno insuficientemente explorado.

Los límites de la evidencia siguen siendo claros. Los modelos clásicos comparten un corpus grande y balanceado y un conjunto de prueba común, por lo que su comparación es controlada. La FNN y la LSTM comparten el entorno de embeddings estáticos, lo que hace que su comparación arquitectónica sea significativa. BERT pertenece a un conjunto de datos y un entorno de entrada diferentes. El análisis de aspectos es una demostración cualitativa basada en un grupo pequeño y deliberadamente negativo de reseñas. Cada resultado es valioso dentro de sus propias condiciones, y mantener esas condiciones visibles es lo que permite que el proyecto combinado sirva como una base de conocimiento fiable.

## Conclusión

Este proyecto construye un camino continuo desde las reseñas de usuarios en bruto hasta una inteligencia de opinión estructurada. Comienza tratando el diseño del corpus y la consistencia lingüística como parte del problema científico. Luego pone a prueba si el sentimiento se representa mejor a través de estadísticas léxicas, conocimiento lingüístico explícito, vectores semánticos densos, procesamiento secuencial o aprendizaje por transferencia contextual. Finalmente, extiende la tarea de la clasificación global a la interpretación a nivel de aspectos fundamentada en la evidencia de las reseñas.

Los experimentos clásicos establecen una línea base sólida y eficiente: un SVM lineal con TF-IDF alcanza un macro-F1 de 0,7309 en un conjunto de prueba balanceado de 105.374 reseñas. Los experimentos con embeddings estáticos demuestran el valor del orden de las palabras, con FastText y una LSTM alcanzando un macro-F1 de 0,752 en el entorno de corpus grande reportado. El experimento con BERT muestra el potencial del aprendizaje por transferencia contextual, alcanzando un macro-F1 de 0,8432 en su corpus balanceado independiente. El experimento con el LLM demuestra cómo la misma base de datos puede sustentar explicaciones estructuradas sobre duración, aleatoriedad, accesibilidad, interacción, balance y agencia del jugador.

Más importante que cualquier puntuación individual es el conocimiento creado por la propia progresión. Los modelos léxicos dispersos muestran cuánto se puede aprender del lenguaje recurrente. Las características lingüísticas revelan qué estructuras evaluativas pueden hacerse explícitas. Los modelos recurrentes demuestran el papel de la secuencia. BERT captura el contexto mediante atención preentrenada. La extracción de aspectos restaura la naturaleza multidimensional de la opinión que una etiqueta global comprime necesariamente.

En conjunto, estas etapas conforman una base sustancial para comprender la opinión en texto de reseñas específico de un dominio. El proyecto no trata el NLP como la selección aislada de un modelo. Trata la procedencia de los datos, la estructura lingüística, la representación, la evaluación, la interpretación y la evidencia como partes conectadas del mismo objeto de investigación.
