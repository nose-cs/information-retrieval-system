# information-retrieval-system



## Introducción
En este proyecto implementamos un Sistema de Recuperación de Información utilizando el Modelo Booleano Extendido. Además, creamos un sistema de recomendación utilizando clusters.

## ¿Qué es el modelo booleano extendido?
El Modelo Booleano Extendido fue presentado en un artículo de "Communications of the ACM" en 1983 por Gerard Salton, Edward A. Fox y Harry Wu. Su propósito es superar las limitaciones del Modelo Booleano utilizado en la recuperación de información.

Este modelo combina características del Modelo Vectorial y el modelo Booleano. Esto permite considerar los pesos de los términos en las consultas y superar las restricciones del Modelo Booleano estándar. Para esto, cada documento se representa como un vector, similar al Modelo Vectorial. Cada componente del vector corresponde a un término asociado al documento. El peso de cada término se mide por su frecuencia de término normalizada.

Para hallar la similitud entre una consulta y un documento se calcula utilizando fórmulas que consideran los pesos de los términos. Por ejemplo, podemos usar la distancia Euclidiana o la Norma-P para medir la similitud.

Este modelo puede considerarse como una generalización de los modelos Booleano y Vectorial. Ha demostrado mejoras en la efectividad en el procesamiento de consultas en comparación con el Modelo Booleano estándar.

En resumen, el Modelo Booleano Extendido permite una recuperación de información más flexible y precisa al considerar tanto la correspondencia parcial como los pesos de los términos en las consultas y documentos.

# Detalles de la implementación

El proyecto fue desarrollado utilizando el lenguaje de programación Python. Para la implementación del modelo Booleano, en primer lugar procesamos la consulta utilizando sympify, llevándola a forma normal disyuntiva.

Para hallar el peso de nuestra consulta, calculamos el peso de cada término de la consulta utilizando tf-idf. Luego, encontramos el peso de cada componente conjuntiva mediante la fórmula:

\[1 - \sqrt{\frac{(1 - w_1)^2 \ldots (1 - w_t)^2}{t}}\]

A continuación, a partir de los pesos de cada componente conjuntiva, hallamos los pesos de la consulta mediante la fórmula:

\[\sqrt{\frac{w_1^2 + w_2^2 \ldots + w_t^2}{t}}\]

Los resultados hallados se ordenan por importancia y se devuelven. Si un resultado tiene peso 0, no se muestra.

Para el sistema de recomendación, utilizamos la clase KMeans de la biblioteca scikit-learn. Mediante esta clase, clusterizamos los documentos y determinamos en qué cluster se encuentran los documentos preferidos por el usuario. Luego, devolvemos los documentos que pertenecen al mismo cluster.

La interfaz gráfica se encuentra en la carpeta "gui" y está desarrollada utilizando la biblioteca Streamlit de Python.

---

## Problemas y posibles mejoras

1. Palabras reservadas en sympy: La biblioteca de sympy da error al introducir ciertas palabras reservadas en el método para llevar a forma normal. Una posible solución podría ser guardar estas palabras y mostrar una advertencia al usuario cuando intente introducirlas.

2. Sistema de filtrado: Podría añadirse un sistema de filtrado (por ejemplo, por autores de los documentos).

3. Precálculo de datos del tf-idf: Podríamos precalcular los datos del tf-idf de los documentos y archivarlos para no tener que calcularlos cada vez que se ejecute el programa.