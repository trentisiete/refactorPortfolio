---
title: "Diverger y Los LLMs"
date: 2026-03-11
excerpt: "Un análisis exhaustivo de los modelos de lenguaje y su razonamiento."
# tags: ["python", "sklearn", "mlstudio", "gemini"]
draft: true
lang: "es"
kind: "blog"
# readingTime: 7
---


# Del Software 1.0 a los agentes autónomos

## Prólogo

2026: 200K lineas de código intensas (En un mes). -> Inditex

Enero 2026: Y si repetimos este proyecto?
Algo que duró 14 semanas, lo hizo en una semana.

2 RITMOS
1. Ritmo en el que se transforma la tecnología (YA HA PASADO)
2. Ritmo en el que se adopta la tecnología

Diverger: tODOS usan herramientas de IA Generativa al máximo.

## LLMs

En todos los modelos de lenguaje subyace una tecnología que es un Modelo de Lenguaje (L-Large (Gran Escala) LM).

Yogurth: Categoría -> LLM
Oikos: La marca -> GPT-x
Pack de yogurths -> ChatGPT
_______________ -> OTRA CAPA MAS (API o Suscripción por ventanas)

Los LLMs buscan coherencia, no veracidad.
Esto ya existía antes de los modelos de lenguaje (Transformers - Attention is all you need).


### El contexto
Antes:
Juego de "envía la siguiente palabra de whatsapp" -> Inchorente
Ahora:
Cuantas mas palabras de contexto tenga, mas coherencia para calcular la siguiente palabra.

"En un lugar de la mancha cuyo nombre no quiero _________" - Acordarme.

¿Que pasa cuando coges mucho contexto?
No todas las palabras tienen la misma relevancia. Unas son mas importantes que otras.

La arquitectura de transformers procura solucionar esto. Es capaz de predecir cual es el contexto mas relevante!

Los modelos van consiguiendo cada vez contextos mas grandes.

Los tokens de salida son mas caros que los de entrada. Por que? Porque son autoregresivos.

Cada vez contextos mas amplios:
- GPT -3.5 (2023) -> 4k tokens ~ 8 páginas de contexto - Se llenaba muy rápido.
- Ahora: Gemini -> 1M tokens

Los LLMS hacen Stimming o Lemmatización -> Con esto se crean los tokens (Tokenizadores)
No se usan caracteres, los caracteres no tienen semántica, la agrupaciones de caracteres si que tienen semántica.

Esto nos lleva a la maldición de la dimensionalidad -> Se usan tokens porque las palabras nos lleva a estos idiomas. (Limitación).
Aún asi los tokens tienen sus propias limitaciones. (Idiomas p.ej)

### 2020

OpenAI da un salto muy grande en el volumen, tanto de la red como los datos.

Los modelos frontera usan todos los datos disponibles, lo sacan de todos los lados posibles.

Antes de aumentarlos, estos modelos eran capaces de extraer sintaxis. Esto no es una sorpresa. Las técnicas basadas en IA todas tienen en cuenta que los algoritmos extraerán patrones (Patrones Evidentes).

Pero lo que ocurre cuando se aumenta el volumen es que aumenta la semántica. (La habitación china : *Una máquina no puede extraer semántica*)

Los LLMs van recolocando palabras, pero de tanto cambiar palabras, va recolocando mas conceptos que palabras. Ese espacio relaciona conceptos.
Hombre y mujer tiene misma relación que rey y reina -> Información semántica de Género.

Aquí entran los sesgos. No puede haber un sistema de decisiones que sea insesgado-natural.

### Que había antes de los Modelos de lenguaje - Embeddings.

Modelos conceptuales del mundo muy limitados, solo entrenados con texto.
Los modelos naturales tienen vision, tacto, etc y ya luego texto.

Todo esto no implica consciencia. Los LLMs no son conscientes.

- Embeddings (Normales): 400-600 dimensiones.
- Embeddings de Gemini: 3000 dimensiones.

----

Prompt : Problema nuevo, inventado que no existe en su entrenamiento.
Esto implica que el modelo entiende lo que le he pedido.

**Loros Estocásticos**
Los modelos no entienden nada y solamente predicen palabra por palabra.
Pero, y si modificamos el significado de enteder, razonar, etc.
Pero como eso solamente lo hemos asociado a personas, entonces si tu entiendes, implica que comprendes y luego implica que razonas, etc.
Pero si separamos estos significados, el LLM si que entra dentro de esta descripcion.

Que un modelo razone no significa que lo haga como nosotros, significa que su salida resulta de un razonamiento.

No hay pensamiento, escribe siguiente palabra, siguiente palabra, siguiente palabra,...

**Entender y comprender**
- **Entender**: Una parte.
- **Comprender**: Has entendido todo.

### Competencia sin comprensión

Ejemplo: Pajaro cuco, pone huevos en otros nidos, tira los huevos del nido y así el tiene mas comida. Esto lo hace pero no tiene comprensión de esta acción aun teniendo la competencia.

----

Aunque un LLM tiene mas conocimiento que cualquier persona, es conocimiento sin comprensión.

### 3 cosas que no saben hacer bien los LLMs
- No es una base de Datos: Cuando le preguntas algo, no lo busca en ningun sitio.
- No hace calculos: El 1000 lo separa en 2 tokens: 100 y 0. Son palabras, no números
    - Haciendolo paso a paso, puede ser que lo haga bien. El espectro de posibilidades se reduce.
    - Lo que si hace bien es generar la herramienta de programación para esa operación.
    - Alucinan: Generan información falsa. La preocupación es la coherencia, no la veracidad.

        El LLM no coge la palabra mas probable. Coge una de las mas probables para que no sea determinista (Misma entrada, misma salida).
        "Michael Jordan ganó 2 veces a Magic Johnson" -> FALSO. Una vez que pone "2" ya está condenado a poner 2 fechas. "Michael Jordan ganó 2 veces a Magic Johnson en *1991 y 1992*"

        ¿Como se mitiga?
        - Añadir informacion en el contexto. Le pesa mas el contexto que todo el entrenamiento. Mala política rebatirle a un LLM ya usado.  Ya ha usado su contexto y está empecinado. Es mejor hacerlo en un nuevo chat.
            En este contexto, el mismo LLM te puede ayudar a rebatirle en otro chat.

            Los agentes usan esto. La probabilidad de que 1 modelo falle es 0.001, la de que 2 fallen es 0.002: Por eso, la prob de que las 2 fallen es 0.001*0.002

            ### Pensar rápido, pensar despacio

            El auge de los modelos razonadores, que ahora no son instantaneos, ahora razonan. OpenAI o1 model.
            Dentro tiene un prompt que le dice que piense alternativas que no sean obvias.

## Agente

### Que es un agente?
Problema de nomenclatura.
**Agente de Software**: Software que actúa para un usuario.

### 2024 - Agentic AI (Agentico no existe, agentivo si pero los llamaremos Sistemas multiagente)
Google y Microsoft les llama agentes, pero porque se pusieron de moda. No es un chatbot, hay proceso de entrar.

Se basan en los modelos de lenguaje: Cogen sus capacidad, limitaciones, etc.
- Sistema multiagentes: Compuesto con varios agentes.

### Capacidades
Cada agente tiene diferentes capacidades:

**Directas**
- Razonamiento y Planificacion (No humano): Basado en la capacidad que hereda cada modelo de lenguaje.
- Reflexión: Parte de las alucinaciones.
**Indirectas**
- Memoria (En psicología memoria de trabajo): A menudo compartida.
    ¿Por qué?
    Un modelo no tiene memoria, cada llamada a un modelo de lenguaje es independiente, se le añade memoria. Los modelos de lenguaje son estáticos
- Herramientas externas: Sobretodo les da autonomía.
- Colaboración: Entre sí, varios agentes, cada uno con su responsabilidad.
- Otro nivel: Sistema Multiagente que colabora con otros sistemas multiagente. A2A: Protocolo de Google


*ChatDev architecture: Image adapted from "Communicative Agents for Software Development" - Qian et al. (2023)*
### Patrones de Diseño de Agentes autónomos
- En cascada: El que mejor funciona para los agentes actuales
### Roles
Hay distintos roles, cada uno con el suyo.

Si esto funciona tan bien, por que existen las empresas de software aún? **GOMOKU**.
Esa palabra se carga de contexto, el LLM tiene la información de todo. Pero en la vida real no funciona así.

Poner objetivos es algo muy humano y poco automatizable.
El principal **beneficio** que tienen los agentes es la **adaptabilidad a la variabilidad del problema**.

UN sistema de agentes es intencional, un sistema de software tradicional requiere ser escrito.
**PERO**
Un sistema multiagente puede perder el **control**.

#### ¿Como se toma el control del sistema de un agente?
Meter mas agente, control.

Ej: DeepResearch:
    Le pido un informe y lo que ocurre es un proceso de hacer informes. Hay múltiples agentes, cada uno con una tarea. Proceso agentico compuesto de agentes, pero este sistema está hecho especificamente para hacer informes. Es muy útil pero no se puede usar para otras cosas.
    Para programar, hacer pruebas, etc habría que hacer uno diferente aunque sean similares.


## Hardware y Software
1. Primeros ordenadores, la intención de la máquina residía en el Hardware. Había que conectar los cables para la tarea.
2. Protosoftwares
3. Por fin, el código: Aparece COBOL (1960s)

### 3 tipos de software
1. Software 1.0:
    Programador->ordenador->programa
    (Datos, programa)->ordenador->ejecución

2. Software 2.0: Machine Learning
    Entrenamiento -> Inferencia (ejecución)
    El software 2.0 no sustituye al software 1.0. Lo usa.

### Software: Productos o servicios cuyo coste marginal tiende a cero.
*the zero marginal cost society*
    1975: Bill Gates fue el primero que se dió cuenta
        - Windows: Le costaba lo mismo vender 1 licencia que 1M de licencias. Habrá un número de licencias que me sirven para costear la creación de Windows. Pero el resto es Beneficio.

    1998-2002s: Internet: Hace que el coste marginal tienda a cero todavia mas.

    AHORA:
    3º Etapa: Activo fisico digitalizada - Trabajo Humano.

### Claude Code
El claude Code tiene un proceso agéntico para crear código. Tal y como hacia el *"Investigación en Profundidad"* de OpenAI
Tiene mucha personalización - Te permite poner Skills, MCP, Tools que puede usar. Luego una parte menos estructural (Hooks). Cuando se termina un proceso, que se empiece otro. O que se despliegue.

#### El problema del compilador de C
Un trabajador de Anthropic creo un compilador de C con 25000 dolares en 14 días.
Stallman tardó 18 meses en conseguirlo.

#### Que hacen bien
Llevan una gestión de contexto excelente de forma autónoma. Funcionan mucho mejor y durante mucho mas tiempo.
Han dejado de lado los embeddings, *Que gurda un embedding?PREGUNTA*

Si bien programar, nos deja en estado de flow. A dia de hoy, una tarea de Claude Code puede tardar 3~4 horas pero es el humano el que prepara la tarea.
Para los clientes no se puede hacer vibe coding, por que vibe coding parece que se refiere a que no te interesa el código, en la vida real si que te interesa -> Exponential Programming.

#### Si ya no vas a tocar el código?
- Cuanto tiempo se tarda en adoptar?
- Cual es el ritmo me adopción?
    Si lo escribe una máquina y lo lee una máquina, ya no hace falta que sea legible no? FALSO
    Es necesario porque el código debe ser legible, detrás tiene un modelo de lenguaje.

### Métodos
- **Domain driven design**
- **Spec driven Development**

### Métodos de desarrollo
- **Especificación**: Que quieres que haga - Lo va a hacer una persona
    No solo funcionales, estructural, arquitectónico, de seguridad...
- **Desarrollo**: La hace el agente
- **Pruebas y Validación**: Debería ser automatizable

### Sin experiencia vs con experiencia
1 persona junior puede rendir como una que tuviera 5 o 6 años de experiencia. Ahora una persona que coordina 6-7 personas, lo puede hacer todo solo. Entonces, cómo se introducen nuevas personas a la operación.
- **Como se combina el aprendizaje con la utilización de las herramientas?**: La calculadora y las multiplicaciones.

### Trabajo asíncrono
Como cambia el trabajo día a día.
- Informe: *AI Doesn't reduce work - It intensifies it*
Además de que cognitivamente es mas cargado.

### Paradoja de Jevon's
Cada vez que se ha eficientado la forma de hacer software, se ha generado ma software y se han contratado mas programadores.

### Why demand for code is infinite; How AI creates more developer jobs -Stack Overflow
La paradoja de Jevon's afecta al software.

### La ley de Brooks
Si tienes un proyecto retrasado y metes ams personas, el proyecto se retrasará mas.
Pero, ahora equipos pequeños hacen mas trabajo porque resuelven el problema de la comunicación.

¿Como se estima un proyecto?
A dia de hoy no es facil saber como estimar un proyecto

Si el equipo hace cosas mas rapido, lo tienes que alimentar mas de cosas que hacer - Mas alimento

## SaaS o Producto de Software
1. Anclado a algo fisico: Esto te puede prteger
2. El software que tienes necesita efecto red. Esto le proteje a Facebook por ejemplo
3. La capacidad de distrbucion: No es tanto hacer el producto como que me compren, ue lo puedan usar.

Si no tienes ninguno de esto, lo tienes muy mal.

Probablemnete las empresas no van a crear un CRM, ahora lo van a comprar, comprarán las empresas pequeñas/medianas, es por eso que es complicado que hayan muchos unicornios.

## Diverger
Motivo de Diverger: Programar usando todas las herramientas, desde 2023.
### Equipo transversal:
Dedicado a estar en la última de lo que hay. Esto es porque los que desarrollan tienen dificil ponerse con todo esto tambien.
### Herramientas
Cada uno puede usar la herramienta que quiera de 20$. Pero cada uno puede justificar la herramienta de 200.
### Formación
Inteligencia artificial para desarrolladores o para cientificos o para...
### Comunidad
La gente cuenta lo que hace y lo que usa: Estándar para los demás proyectos.


----
#### Es esto legal?
La mayoria de las legislaciones si que lo permiten.
- Que hacia anthropic? Compraba los libros y los quemaba una vez los usaba.

#### Si ya no puedes escalar por fuerza bruta

1. Encontrar texto de otro sitio (Marginal) -> Transcripción del audio de YT.
    Aquí no hay mucho camino.
2. Cambio de arquitectura -> Mixture of Experts, RLHF (Reinforcement Learning for Human Feedback)
3. Mejora de rendimiento no por entrenamiento, por inferencia.

----
