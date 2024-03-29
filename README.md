# information-retrieval-system

## Autores
- Daniel Abad Fundora C411
- Anabel Benítez González C411
- Enzo Rojas D'Toste C411

## Introducción

La recuperación de información consiste en encontrar material
(generalmente documentos) de una naturaleza no estructurada (generalmente texto) que satisface una necesidad de
información dentro de grandes colecciones (generalmente almacenadas en computadoras).
En este proyecto implementamos un Sistema de Recuperación de Información utilizando el Modelo Booleano Extendido, a
continuación se describen
todas las etapas del proceso de recuperación de información, desde
el procesamiento de la consulta hecha por un usuario, a la representación de
los documentos y la consulta, el funcionamiento del motor de búsqueda y la
obtención de los resultados.

## ¿Qué es el modelo booleano extendido?

El Modelo Booleano Extendido fue presentado en un artículo de "Communications of the ACM" en 1983 por Gerard Salton,
Edward A. Fox y Harry Wu. Su propósito es superar las limitaciones del Modelo Booleano utilizado en la recuperación de
información.

Este modelo combina características del Modelo Vectorial y el modelo Booleano. Esto permite considerar los pesos de los
términos en las consultas y superar las restricciones del Modelo Booleano estándar. Para esto, cada documento se
representa como un vector, similar al Modelo Vectorial. Más adelante, lo veremos con más detalle.

Este modelo puede considerarse como una generalización de los modelos Booleano y Vectorial. Ha demostrado mejoras en la
efectividad en el procesamiento de consultas en comparación con el Modelo Booleano estándar.

En resumen, el Modelo Booleano Extendido permite una recuperación de información más flexible y precisa que el booleano
al considera tanto la correspondencia parcial como los pesos de los términos en las consultas y documentos.

## Consideraciones tomadas a la hora de desarrollar la solución

- **Eficiencia en el procesamiento de consultas y documentos:**
  Se implementó un sistema eficiente para procesar consultas y documentos, utilizando técnicas de preprocesamiento y
  representación de datos, guardando los índices en archivos para no tener que construirlos cada vez que se ejecute el
  programa.

- **Representación de queries en lenguaje natural:**
  Cuando el usuario hace una consulta en lenguaje natural, es necesario transformarla para poder llevarlo a forma normal
  disyuntiva. Para esto, introducimos el operador `and` entre cada palabra de la consulta.

## Explicación de la solución desarrollada

El proyecto fue desarrollado utilizando el lenguaje de programación Python. A continuación, se describen las etapas
principales del proceso de recuperación de información y cómo se implementaron en nuestro sistema.

### Preprocesamiento de los documentos:

Este paso es esencial para representarlos de manera lógica y procesarlos eficientemente.
A continuación, se describen las etapas clave de este proceso:

- Eliminación de signos de puntuación y caracteres especiales:
  Se eliminan los signos de puntuación (,.;:) y otros caracteres especiales (#<>*+).
  Esto mejora la eficacia en la recuperación de información al evitar ruido innecesario.
- Normalización a minúsculas:
  Las mayúsculas/minúsculas no aportan información relevante.
  Por lo tanto, todo el texto se representa en minúsculas.
- Tokenización:
  El texto se divide en una secuencia de “tokens” (palabras o fragmentos significativos).
  Cada token tiene un significado coherente en el idioma utilizado.
- Eliminación de stopwords:
  Se descartan las palabras comunes (stopwords) que no contribuyen significativamente a la clasificación del texto.
  Para esto se utiliza la biblioteca nltk de Python.
- Stemming:
  Se aplica un algoritmo de stemming (como el algoritmo de Snowball) para reducir las palabras a su raíz.
  Esto ayuda en la recuperación de información al agrupar variantes de una misma palabra.

### Representación de los documentos:

Después del preprocesamiento textual, el siguiente paso es la construcción de índices. Un término indexado es una
palabra cuya semántica ayuda a definir el tema principal de un documento, brindando un resumen de su contenido.
Generalmente, se utilizan las palabras contenidas en el propio documento como términos indexados.

La recuperación de información basada en términos indexados se fundamenta en la idea de que la semántica de los
documentos y las necesidades informativas de los usuarios pueden expresarse mediante un conjunto de términos clave.
Estos términos permiten identificar y relacionar documentos relevantes.

En nuestro sistema, utilizamos la librería gensim y su clase Dictionary para construir estos índices. El Dictionary
mapea palabras normalizadas a sus identificadores enteros únicos. Además, almacena información relevante, como las
frecuencias de colección y las frecuencias de documentos para cada término.

Toda lo descrito anteriormente se encuentra en el módulo `corpus` que se encarga de procesar los documentos y
construir los índices. Para no construir los índices cada vez que se ejecute el programa, se guardan en el directorio
llamado `indexed_corpus` utilizando la biblioteca pickle, que permite serializar y deserializar objetos de Python.

### Procesamiento de la consulta:

El procesamiento de la consulta es una etapa crucial en el proceso de recuperación de información. La consulta es la
expresión de las necesidades informativas del usuario. Por lo tanto, es esencial procesarla de manera eficiente para
obtener resultados precisos.

En nuestro sistema, en el módulo `query` podemos encontrar una clase QueryProcessor que se encarga de procesar consultas
en lenguaje natural y las representa como un vector de términos indexados, utilizando la función doc2bow de la clase
Dictionary de gensim.

Este es el procesador de la consulta que usamos en nuestra implementación del modelo vectorial, pero las consultas en
lenguaje natural no son adecuadas para el modelo booleano, por lo que necesitamos un procesador
especial para este modelo. En el módulo `boolean_query_processor` podemos encontrar una clase BooleanQueryProcessor
que se encarga de procesar consultas y llevarlas a forma normal disyuntiva. El preprocesamiento de la consulta se
realiza
similar al anteriormente mencionado, teniendo cuidado de no eliminar información relevante como los paréntesis y los
operadores lógicos. Primeramente se eliminan los signos de puntuación y caracteres especiales, se normaliza a
minúsculas, y se reemplazan las palabras `and`, `or` y `not` por los operadores lógicos `&`, `|` y `~` respectivamente.
Luego, se tokeniza la consulta y se realiza stemming. Finalmente, se lleva la consulta a forma normal disyuntiva
utilizando la función
`to_dnf` de la clase sympy.

### Modelo de recuperación de información:

Un modelo de recuperación de información se define como una tupla (D, Q, F, R(qj , dj )), donde:

- D es el conjunto de documentos.
- Q es el conjunto de consultas.
- F es un framework para modelar las representaciones de los documentos, consultas y sus relaciones.
- R es una función que asocia un número real a cada par (qj , dj) de consulta y documento. La evaluación de esta función
  establece un cierto orden entre los documentos de acuerdo a la consulta.

En nuestro sistema, como mencionamos anteriormente utilizamos el Modelo Booleano Extendido para la recuperación de
información. Este modelo combina características del Modelo Vectorial y el modelo Booleano. Esto permite considerar los
pesos de los términos en las consultas y superar las restricciones del Modelo Booleano estándar. Para esto, cada
documento se representa como un vector, similar al Modelo Vectorial. Cada componente del vector corresponde a un término
asociado al documento. El peso de cada componente se calcula utilizando la fórmula:

$$
w = tf(t, d) \cdot idf(t, D) / max_idf
$$

Donde:

$$
tf(t, d) = f(t, d) / max(f(t_i,d)) \forall t_i \in d
$$

$$
idf(t, D) = log(N / n(t, D))
$$

El tf es la frecuencia del término t en el documento d, el idf es el inverso de la frecuencia del término t en la
colección de documentos D, N es el número de documentos en la colección y n(t, D) es el número de documentos que
contienen el término t. Fijémonos en que se normaliza el tf para evitar que documentos más largos tengan más peso.
Además se normaliza el idf, para que el peso sea un número entre 0 y 1.

Posteriormente, para calcular la similitud entre un documento y una consulta se utiliza la fórmula:

$$
sim(d, q_{and}) = 1 - \sqrt{\frac{(1 - w_1)^2 \ldots (1 - w_t)^2}{t}}
$$

$$
sim(d, q_{or}) = \sqrt{\frac{w_1^2 + w_2^2 \ldots + w_t^2}{t}}
$$

donde \(w_i\) es el peso del término i en los documentos y t es el número de términos en la consulta.

Luego de obtener este ranking, se ordenan los documentos de acuerdo a su medida de similitud de mayor a menor, ya que
los documentos más similares tienen una mayor similitud. Para la recuperación de los documentos, puede establecerse un
umbral de similitud y recuperar los documentos cuyo grado de similitud sea mayor que este umbral.

Cabe destacar que usando las fórmulas anteriores es muy sencillo calcular el peso de una forma normal disyuntiva, ya
que simplemente se calcula el peso de cada componente conjuntiva y se aplica la fórmula del or.

### Agrupamiento de documentos:

Los algoritmos de agrupamiento de documentos dividen un conjunto de documentos en grupos (clusters) de manera que los
documentos similares estén en el mismo grupo, mientras que los documentos disímiles se encuentren en grupos distintos.

Para determinar la cantidad óptima de grupos, se utiliza una métrica basada en la suma de las diferencias al cuadrado
entre cada elemento y el centroide del grupo asignado mediante el método de agrupamiento. Intuitivamente, el valor ideal
es aquel en el eje x (cantidad de grupos) donde se produce el mayor cambio en la pendiente de la función. Demasiados
grupos pueden generar diferencias significativas, mientras que muy pocos no generalizan lo suficiente. El método del
codo se utiliza comúnmente para estimar la cantidad de grupos en cada corpus.

En cuanto a la representación de los documentos, existen varias opciones, como vectores binarios, vectores reales (tf ×
idf) o Word Embeddings. En nuestro algoritmo, utilizamos la representación tf-idf de los documentos.

La biblioteca scikit-learn proporciona una implementación del algoritmo KMeans, que es un método de agrupamiento
basado en centroides. Este algoritmo divide los datos en k grupos, donde cada grupo es representado por el centroide
más cercano. El algoritmo minimiza la suma de las distancias al cuadrado entre cada punto y el centroide de su grupo
asignado.

En nuestro sistema utilizamos la clase KMeans para agrupar los documentos. Mediante esta
clase, clusterizamos los documentos y determinamos en qué cluster se encuentran los documentos preferidos por el
usuario.

### Recomendación de documentos:

La recomendación de documentos es una etapa crucial en el proceso de recuperación de información. La recomendación
consiste en presentar al usuario documentos que puedan ser de su interés, basándose en sus preferencias y en la
similitud con otros documentos que le gustaron.

En nuestro sistema, utilizamos el clustering anteriormente descrito para recomendar documentos.

Cuando el usuario hace una consulta, se le pregunta cuáles fueron los documentos que le gustaron. Estos documentos se
guardan con un ranking (en nuestro caso siempre usamos 1). Luego, cuando el usuario pide recomendaciones, se le
presentan 5 documentos que pertenecen al mismo cluster que los documentos que le gustaron. Estos documentos se
presentan ordenados por similitud con los documentos que le gustaron.

También tenemos una pseudo-retroalimentación que ocurre cuando el usuario no quiere participar en el proceso anterior,
pero igualmente queremos conocer sus intereses para poder recomendarle.
En este caso, se guardan los 5 documentos más relevantes para la consulta en la lista rankings.

Aspectos a destacar:

- Definición de Similitud (S(d_i, d_j)):
  Se calcula la similitud entre los ítems i y j. Esto puede realizarse utilizando diferentes medidas, como la similitud
  de Jaccard o la similitud de coseno. En nuestro caso, utilizamos la similitud de Jaccard definida como:
  sim(A, B) = |rA ∪ rB| / |rA ∩ rB|
- Selección de Vecinos Cercanos (N(i, x)):
  Se seleccionan los k ítems más cercanos a i que han sido evaluados por el usuario x. Estos ítems son los más similares
  a i según la medida de similitud.
- Estimación del Rating (ˆrxi):
  Se estima el rating del usuario x para el ítem i utilizando la siguiente fórmula:
  ˆrxi = bxi + Σj∈N(i,x) Sij (rxj + bxj) / Σj∈N(i;x) Sij
  Donde:
  µ es la media de los ratings de todos los ítems.
  bx es la desviación del rating del usuario x respecto a la media global.
  bi es la desviación del rating del ítem i respecto a la media global.
- Predictor Baseline (bxi): bxi = µ + bx + bi
- Selección de Vecinos usando Clusters:
  Para simplificar, se utilizan los clusters calculados con el algoritmo K-means para seleccionar los vecinos más
  cercanos
  N(i; x). Se eligen todos los documentos que están en el mismo cluster que el documento i y que han sido evaluados por
  x.

Nota:
Es importante señalar que algunas componentes de la fórmula asumen la existencia de múltiples usuarios, lo cual no es
consistente con nuestro sistema.

## Evaluación del sistema

La evaluación de un sistema de recuperación de información es esencial para determinar su eficacia y precisión. La
evaluación se realiza mediante la comparación de los resultados obtenidos con los resultados esperados. A continuación
se describen las métricas utilizadas para evaluar nuestro sistema.

### Métricas de evaluación:

- Precisión:
  La precisión es la proporción de documentos relevantes recuperados con respecto al total de documentos recuperados.
  Se calcula mediante la fórmula:

  Precisión = (Documentos relevantes recuperados) / (Documentos recuperados)

- Recall:
  El recall es la proporción de documentos relevantes recuperados con respecto al total de documentos relevantes.
  Se calcula mediante la fórmula:

  Recall = (Documentos relevantes recuperados) / (Documentos relevantes)

- F1-Score:
  El F1-Score es la media armónica de la precisión y el recall. Esta métrica proporciona una medida única que
  combina ambas métricas.
  Se calcula mediante la fórmula:
  
  F1-Score = 2 * (Precisión * Recall) / (Precisión + Recall)

- Fallout:
  El fallout es la proporción de documentos no relevantes recuperados con respecto al total de documentos no
  relevantes.
  Se calcula mediante la fórmula:

  Fallout = (Documentos no relevantes recuperados) / (Documentos no relevantes)

- r-Precision:
  La R-Precision es la precisión calculada sobre los primeros r documentos recuperados y los primeros r relevantes.

- r-Recall:
  La R-Recall es el recall calculado sobre los primeros r documentos recuperados y los primeros r relevantes.

Para evaluar nuestro sistema, utilizamos el dataset `cranfield` de prueba que contiene consultas y documentos
relevantes,
aunque se pudiera utilizar cualquier dataset simplemente implementando una clase en el módulo `corpus` que herede de la
clase abstracta `Corpus` e implemente el método abstracto `parse_documents` que define cómo se parsea el conjunto de
documentos y los añade a los documentos del corpus.

### Resultados de la evaluación:

Evaluando 25 queries en el corpus mencionado obtenemos los siguientes resultados:

    Model    Precision  Recall     F1        R-Precision  R-Recall  Fallout  
    Boolean   0.000000  0.000000  0.000000     0.000000    0.000000  0.000063  
    Vector    0.008577  0.219358  0.015507     0.000000    0.000000  0.261966  
    Extended  0.006646  0.649103  0.013020     0.043478    0.019876  0.610591 

Nota: se tomó r = 5

Analizando los resultados vemos que nuestra implementación del modelo booleano extendido es mejor que nuestra
implementación del modelo booleano y del vectorial, aunque aún está lejos de lo ideal para un sistema de recuperación de
información. 

## Problemas y posibles mejoras

1. Palabras reservadas en sympy: La biblioteca de sympy da error al introducir ciertas palabras reservadas en la función
   `sympyfy`. Una posible solución podría ser guardar estas palabras y mostrar una advertencia al
   usuario cuando intente introducirlas ya que actualmente no se hace le dice al usuario qué palabras son reservadas.
   También pudiera buscarse otra biblioteca o parsear a mano las expresiones lógicas para evitar este comportamiento de
   la biblioteca.

2. Sistema de filtrado: Podría añadirse un sistema de filtrado (por ejemplo, por autores de los documentos) para
   mejorar la precisión de las recomendaciones. Para esto se debería implementar documentos específicos por cada corpus,
   para aprovechar completamente todo lo que nos brinda.

3. Precálculo de datos del tf-idf: Podríamos precalcular los datos del tf-idf de los documentos y archivarlos para
   mejorar la eficiencia del sistema.

4. Mejorar la interfaz gráfica: La interfaz gráfica actual es muy básica y podría mejorarse para hacerla más amigable y mostrar todas las funcionalidades.

5. Añadir un sistema de retroalimentación más avanzado: Actualmente, el sistema de retroalimentación es muy básico,
   podría hacerse más efectivo, guardando consultas anteriores y recomendando documentos basados en estas consultas.

6. Añadir sinónimos: Podrían buscarse sinónimos para mejorar la precisión de las consultas.
