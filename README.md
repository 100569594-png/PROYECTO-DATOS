 # Análisis de la polarización ideológica en redes sociales a partir de la propagación de contenidos desinformativos

## 1.Descripción del problema 
Uno de los principales problemas que enfrentan las redes sociales y los medios digitales es la desinformación.  La verificación manual de la información se vuelve más difícil y la polarización ideológica se ve favorecida por la rápida propagación de contenidos falsos.  Este proyecto tiene como finalidad detectar automáticamente noticias falsas (fake news) mediante el análisis de su contenido textual, analizando distintas estrategias de representación vectorial y aplicando un clasificador.

La base de datos empleada proviene del WELFake Dataset, que se compone de miles de noticias categorizadas como verdaderas (label = 0) o falsas (label = 1).  Cada entrada incluye el contenido textual de la noticia, que ha sido purificado y limpiado. Después de suprimir documentos sin texto, así como los que eran demasiado breves o excesivamente extensos (menos de 200 palabras o más de 2000), se filtró el dataset de datos para asegurar su coherencia y calidad, quedando listo para ser analizado.

### Asimismo, se llevó a cabo un análisis de exploración que comprende:

 - Distribución de las clases.

 - Longitud de los textos y sus histogramas correspondientes.

 - Términos más comunes en cada clase.

 - Nubes de palabras para identificar patrones léxicos que diferencien.

El propósito final es identificar la metodología que ofrece el rendimiento más alto en la detección automatizada de desinformación.



## 2. Análisis del conjunto de datos.

El dataset está compuesto por un gran número de artículos clasificados en dos categorías:
0 — Real
1 — Fake
Las variables principales incluyen el texto completo de la noticia, su título (si se dispone de él) y la etiqueta asociada. Los datos se encuentran en formato  textual y numérico. Aquellos valores que correspondian a entrada incompletas o nulas, se purificaron, garantizando el contenido real para cada noticia, evitando fallos posteriores en el análisis lingüístico. Concluyendo que la estructura del dataset es adecuada para tareas de clasificación propuestas.


## 2.2. Estadísticas básicas y distribución de clases. 
Se observó una distribución equilibrada entre noticias reales y falsas, lo que resulta beneficioso para el entrenamiento de modelos sin necesidad de aplicar técnicas de rebalanceo. Esta simetría entre las clases disminuye  el riesgo de sesgos hacia una categoría, permitiendo una evaluación más fiable.

En cuanto a la longitud de los textos, se calculó el número de palabras por noticia. La mayoría de los artículos se  sitúan entre 100 y 600 palabras, con pocos casos que superan este rango. Este comportamiento es frecuente para textos periodísticos o derivados de redes sociales, donde predominan  breves piezas informativas. 


## 2.3. Comparación de longitudes entre noticias reales y fake

El análisis de la longitud por clase hizo notorias diferencias estructurales relevantes:

- Las noticias reales suelen mostrar mayor variabilidad en la extensión de sus textos, contando con una distribución más amplia longitudes.
- Las noticias fake presentan una tendencia a ser más cortas y menos diversas en su tamaño.

Este patrón coincide con hallazgos en literatura previa sobre desinformación, donde los contenidos falsos suelen estar diseñados para ser rápidos de consumir y fáciles de  difundir.

## 2.4. Análisis preliminar del contenido textual
Con el objetivo de comprender mejor las características lingüísticas, se analizaron las palabras más frecuentes en cada clase. Donde se determino lo siguiente:

- Las noticias reales poseen un  vocabulario más descriptivo, informativo y diverso.
- Las noticias fake presentan una mayor repetición de términos cargados emocionalmente  o directamente vinculados a discursos sensacionalistas.

Estas diferencias semánticas pueden resultar útiles para la representación vectorial y el modelado.


## 3.Explicación de las metodologías utilizadas.

## 3.1. Preprocesamiento del texto

## 3.2. Representación vectorial del texto

- TF-TDF

- Word2Vec
- BERT (embeddings contextuales)


## 2.3. Modelos de clasificación

## 3.Resultados experimentales.



## Discusión

## 4. Conclusiones


bibliografia
