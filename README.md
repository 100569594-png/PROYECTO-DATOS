 # Análisis de la polarización ideológica en redes sociales a partir de la propagación de contenidos desinformativos

## 1.Descripción del problema 
Hoy en día, uno de los principales problemas a los que nos enfrentamos es la desinformación. Las redes sociales y los medios digitales facilitan la difusión rápida y masiva de contenidos, permitiendo que los usuarios compartan información sin que esta haya sido previamente verificada, lo que favorece la propagación de bulos y noticias falsas, y como consecuencia, se contribuye al aumento de la polarización ideológica.

En este contexto, surge la necesidad de garantizar la veracidad de los textos, con el objetivo de mitigar los efectos de la desinformación. 

Para ello, en este proyecto se realiza un preprocesamiento de los datos y se comparan distintas estrategias de representación vectorial del texto. En concreto, se implementan tres técnicas de vectorización, junto con varios modelos de clasificación supervisada de la librería scikit-learn y una red neuronal artificial.

Finalmente, se emplea un modelo Transformer preentrenado mediante la librería Hugging Face, ajustado para la tarea de detección de noticias falsas. Y se compara con los modelos de clasificación implementados para determinar el más eficaz en la detección de contenido manipulado.


## 2. Análisis del conjunto de datos.

La base de datos empleada proviene del WELFake Dataset, este conjunto de datos esta compuesto por un gran número de artículos clasificados en dos categorías:

- 0 — Real
- 1 — Fake

Las variables principales incluyen el texto completo de la noticia, su título (si se dispone de él) y la etiqueta asociada. Los datos se encuentran en formato  textual y numérico, lo que permite aplicar técnicas de procesado de lenguaje natural y análisis contextual.

Antes de su utilización, se llevo a cabo un proceso de depuración pudiendo asi garantizar la calidad del contenido. Se eliminaron aquellos valores que correspondian a entradas incompletas o inconsistentes con el objetivo de  garantizar un contenido robusto para  cada noticia. Todo ello, con el fin de evitar errores en las fases posteriores de tratamiento lingüístico y modelado.


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

Además, se procedió a la visualización de las palabras más repetidas  mediante nubes de palabras para cada categoría. Esta representación afirma las tendencias de cada clase anteriormente comentadas, donde las noticias reales mostraron mayor diversidad léxica, mientras que las fake exhibieron concentraciones de palabras llamativas o polarizantes. 
Estas representaciones visualen hacen posible  identificar de manera de forma sencilla  patrones de contenido que pueden relacionarse con estrategias de manipulación narrativa.


## 2.5. Ejemplos representativos por clase
Se examinaron muestras individuales de textos reales y falsos, lo que permitió identificar diferencias en el estilo de redacción:
Las noticias reales suelen presentar estructuras más coherentes, referencias informativas y mayor profundidad descriptiva.
Las noticias falsas tienden a utilizar frases más directas, simplificadas o diseñadas para provocar una respuesta emocional inmediata.
Esta observación cualitativa complementa los hallazgos cuantitativos y apoya la hipótesis de que la desinformación sigue patrones estilísticos distintivos.

## 2.6. Hipótesis iniciales

A partir del análisis  de la base de datos pueden formularse varias hipótesis para guiar las etapas posteriores del proyecto:

- La longitud del texto puede ser un posible  indicador útil para distinguir entre noticias reales y fake.
- Las diferencias léxicas aportan para que las  representaciones vectoriales capten  patrones de discriminación relevantes.
- Los términos emocionalmente cargados en noticias falsas apuntan a  una relación entre desinformación y
polarización ideológica.

## 3.Explicación de las metodologías utilizadas.

En esta sección se describe el procesamiento de los datos, la división del conjunto de datos,  la representación vectorial del texto y los modelos de clasificación utilizados.   

## 3.1. Preprocesamiento del texto

Para el preprocesamiento del texto se han aplicado  técnicas estándar de NLP(Natural Language Processing). Estas técnicas Antes de aplicar cualquier técnica de representación vectorial, se llevó a cabo un preprocesamiento del texto, que consiste en preparar y transformar el texto en un formato adecuado para su análisis.

El preprocesamiento que se ha seguido es el siguiente: en primer lugar, se ha convertido todo el texto a minúsculas y se eliminaron los caracteres especiales y  signos de puntuación.

Luego se procedió a la tokenización del texto y a la eliminación de las stopwords en inglés. Por último, se aplicó la lematización de las palabras con el fin de reducirlas a su forma canónica.

Como resultado de este proceso, se generó una versión limpia del texto que sirvió como entrada para los distintos métodos de vectorización empleados en el proyecto.


## 3.2. Representación vectorial del texto
Una vez se ha completado el preprocesamiento del texto, se pasa a la división del conjunto de datos  en diferentes particiones  con el fin de garantizar un análisis objetivo de los diferentes modelos de clasificación. Esta división se trata de un proceso clave, mediante el cual puede  evitarse  el sobreajuste, permitiendo medir de forma totalmente realista la  capacidad de generalización del sistema ideado.

Para el correcto desarrollo del proyecto  se empleó una estructura de tres subconjuntos:

#### - Conjunto de entrenamiento (train): (60%)
#### -Conjunto de validación (validation): (20%)
#### -Conjunto de prueba (test): (20%)



## 3.3. Representación vectorial del texto
Se han implementado tres técnicas de representación vectorial: TF-IDF, Word2Vec y embeddings contextuales. 

### TF-IDF (Term Frequency – Inverse Document Frequency) 
La vectorización se realizó utilizando la clase TfidfVectorizer de la librería scikit-learn, configurada con los siguientes parámetros:
- max_features: Se limitó el vocabulario a un máximo de 20 000 características, seleccionando las palabras más frecuentes.
  
- ngram_range=(1,2): Se consideraron unigramas y bigramas con el objetivo de capturar tanto palabras individuales como combinaciones de dos términos.
  
- min_df: Se estableció una frecuencia mínima de 5 noticias
La vectorización se ajustó sobre el conjunto de entrenamiento y se aplicó a los conjuntos de validación y test, quedando cada noticia representada mediante un vector de dimensión fija.


### Word2Vec (Procesamiento de lenguaje)
La segunda técnica de representación vectorial es Word2Vec, que permite capturar relaciones semánticas entre términos a partir de su contexto de aparición.

Para su implementación se utilizó la librería gensim y los hiperparámetros seleccionados fueron:

- vector_size: Cada palabra se representa como un vector de 300 componentes.

- window=2: Considerando las relaciones entre las dos palabras anteriores y semánticas. Se seleccionó un valor de 5 al principio, pero finalmente se ajustó a 2.

- min_count: Se consideraron palabras que aparecieran al menos 20 veces en en el conjunto de datos.

- sample: Se aplicó un umbral de submuestreo de 6×10^-5 para reducir la influencia de términos muy frecuentes.

- alpha, min_alpha: se probaron distintos valores de tasa de aprendizaje, fijándose finalmente 0.03 y 0.0007, respectivamente.

- negative: Se emplea un muestreo negativo de 20 muestras

- sg=1: arquitectura Skip-gram

Cada noticia se representa como el promedio de los vectores de las palabras que lo componen.

###  Embeddings contextuales (BERT)
Como tercera técnica de representación vectorial se emplean los embeddings contextuales obtenidos mediante el Transformer preentrenado DistilBERT. Este modelo fue seleccionado por estar optimizado para textos en inglés y por su capacidad de sensibilidad al contexto.
En este caso, el modelo no se entrena, sino que se utiliza para extraer las características, manteniendo los pesos congelados. Como hiperparámetros se seleccionó: 

- max_length: La longitud máxima de la secuencia de entrada es 256.
- batch_size = 16.
  
Como resultado, cada noticia quedó representada mediante un vector correspondiente a la dimensión oculta del modelo.



## 4. Modelos de clasificación





## 5.Resultados experimentales.


A lo largo de esta sección se presentan los resultados obtenidos tras el entrenamiento y evaluación de los  distintos modelos desarrollados. El propósito principal es seleccionar el modelo más adecuado para un sistema de clasificación verídica  de noticias.

Para la evaluación, se ha priorizado la Precisión de la clase 0 (Noticias Reales). Esta métrica es clave para valorar el rendimiento del modelo,  ya que mide la proporción de aciertos sobre el total de noticias etiquetadas como verdaderas.
El objetivo es asegurar una alta confianza  en las noticias validadas, reduciendo drásticamente la posibilidad de catalogar erróneamente una noticia falsa como real.


Por otro lado, también se tendrá en cuenta como métrica evaluable el accuracy, proporcionando qué porcentaje total de predicciones son correctas sin distinguir entre clases. 



<img width="642" height="112" alt="image" src="https://github.com/user-attachments/assets/1daa2ebc-d407-4fee-9f1f-e31c52eb616d" />

<img width="643" height="112" alt="image" src="https://github.com/user-attachments/assets/7a448159-56f7-4abd-96b2-41eca2f53adb" />

<img width="644" height="114" alt="image" src="https://github.com/user-attachments/assets/a60c6ddd-a293-4661-b55f-6650469711e7" />



## Discusión

## 5. Conclusiones


bibliografia
