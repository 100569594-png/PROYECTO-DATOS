 # Análisis de la polarización ideológica en redes sociales a partir de la propagación de contenidos desinformativos

## 1.Descripción del problema 
Hoy en día, uno de los principales problemas a los que nos enfrentamos es la desinformación. Las redes sociales y los medios digitales facilitan la difusión rápida y masiva de contenidos, permitiendo que los usuarios compartan información sin que esta haya sido previamente verificada, lo que favorece la propagación de bulos y noticias falsas, y como consecuencia, se contribuye al aumento de la polarización ideológica.
En este contexto, surge la necesidad de garantizar la veracidad de los textos, con el objetivo de mitigar los efectos de la desinformación. 
Para ello, en este proyecto se realiza un preprocesamiento de los datos y se comparan distintas estrategias de representación vectorial del texto. En concreto, se implementan tres técnicas de vectorización, junto con varios modelos de clasificación supervisada de la librería scikit-learn y una red neuronal artificial.
Finalmente, se emplea un modelo Transformer preentrenado mediante la librería Hugging Face, ajustado para la tarea de detección de noticias falsas. Y se compara con los modelos de clasificación implementados para determinar el más eficaz en la detección de contenido manipulado.


## 2. Análisis del conjunto de datos.

La base de datos empleada proviene del WELFake Dataset, este conjunto de datos esta compuesto por un gran número de artículos clasificados en dos categorías:

0 — Real
1 — Fake

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

El  proyecto se desglosa en cuatro partes principales:

- Preprocesamiento.
- Vectorización.
- Modelado.
- Evaluación.

## 3.1. Preprocesamiento del texto

Para el preprocesamiento del texto se han aplicado  técnicas estándar de NLP(Natural Language Processing). Estas técnicas se tratan de procedimientos fundamentales que permiten transformar texto en bruto en una representación totalmente estructurada, limpia y procesable por un modelo algorítmico.

Estas técnicas sirven son utililes para reducir el ruido textual, normalizadar el lenguaje, extraer el significado de las palabras, preparar los datos para su posterior vectorización y mejorar el rendimiento de los modelos de clasificación. Si estas técnicas no se aplican previamente, el modelo trabajaría con un texto totalmente desordenado.

A continuación, se encuentrán técnicas aplicadas al conjunto de datos, con el objetivo de obtener los mejores resultados podibles.

- #### Conversión a minúsculas: evita que trate la misma palabra como palabras diferentes, reduciendo la dimensionalidad del vocabulario.
  
- #### Eliminación de símbolos y caracteres no alfanuméricos:  se suprimen elementos que no aportan ningun tipo de significado útil para la clasificación.
  
- #### Tokenización (NLTK): convierte el texto en una lista de palabras, permitiendo un análisis individual por palabra.
  
- #### Eliminación de stopwords: suprime palabras frecuentes con poco valor semántico, no aportando valor en la clasificación.
  
- #### Lematización (WordNet): transformación de las palabras a su forma base.

Una vez el preprocesado del texto ha finalizado, el resultado se almacena en las columnas tokens y text_limpio, necesarias para las técnicas de vectorización.

## 3.2. Representación vectorial del texto

- TF-TDF

- Word2Vec:
- 
- BERT (embeddings contextuales):
- 


## 3.3. Modelos de clasificación

## 4.Resultados experimentales.



## Discusión

## 5. Conclusiones


bibliografia
