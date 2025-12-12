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
<img width="716" height="401" alt="Captura de pantalla 2025-12-12 a las 20 22 47" src="https://github.com/user-attachments/assets/ae5881ff-7b5d-4eec-92ec-65f8ae425e8a" />

## 2.4. Análisis preliminar del contenido textual
Con el objetivo de comprender mejor las características lingüísticas, se analizaron las palabras más frecuentes en cada clase. Donde se determino lo siguiente:

- Las noticias reales poseen un  vocabulario más descriptivo, informativo y diverso.
- Las noticias fake presentan una mayor repetición de términos cargados emocionalmente  o directamente vinculados a discursos sensacionalistas.

Estas diferencias semánticas pueden resultar útiles para la representación vectorial y el modelado.

Además, se procedió a la visualización de las palabras más repetidas  mediante nubes de palabras para cada categoría. Esta representación afirma las tendencias de cada clase anteriormente comentadas, donde las noticias reales mostraron mayor diversidad léxica, mientras que las fake exhibieron concentraciones de palabras llamativas o polarizantes. 
Estas representaciones visualen hacen posible  identificar de manera de forma sencilla  patrones de contenido que pueden relacionarse con estrategias de manipulación narrativa.

Nubes de datos para **TEXTOS REALES**
<img width="944" height="501" alt="image" src="https://github.com/user-attachments/assets/80e9c094-09fa-4c0f-9a9b-5641eb1879ab" />
<img width="931" height="493" alt="image" src="https://github.com/user-attachments/assets/a6c33a95-7c28-4390-82ab-b6dcfc7be6b8" />


Nubes de datos para **TEXTOS FAKE**
<img width="922" height="498" alt="image" src="https://github.com/user-attachments/assets/ce64a435-2a25-465c-a183-c662be45a21a" />
<img width="934" height="499" alt="image" src="https://github.com/user-attachments/assets/2c809d38-efba-4232-a28d-9e35dd997d81" />




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


## 3.2. División de los datos
Una vez se ha completado el preprocesamiento del texto, se pasa a la división del conjunto de datos  en diferentes particiones  con el fin de garantizar un análisis objetivo de los diferentes modelos de clasificación. Esta división se trata de un proceso clave, mediante el cual puede  evitarse  el sobreajuste, permitiendo medir de forma totalmente realista la  capacidad de generalización del sistema ideado.

Para el correcto desarrollo del proyecto  se empleó una estructura de tres subconjuntos:

#### - Conjunto de entrenamiento (train): (60%)
#### -Conjunto de validación (validation): (20%)
#### -Conjunto de prueba (test): (20%)

## 3.2. Representación vectorial del texto
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

Para el entrenamiento y evaluación de los modelos se han empleado tres algoritmos de clasificación de la librería Scikit-learn, una red neuronal en PyTorch y un modelo Transformer preentrenado  ajustado mediante fine-tuning.

- **Regresión logística:** la implementación se realizó mediante la clase LogisticRegression, y como hiperparámetros configurado a resaltar tenemos “max_iter”, que se fijó a 300.  Dicho valor que se seleccionó tras probar configuraciones con 100 y 200 iteraciones
  
- **SVM:** implementada mediante la clase LinearSVC con sus parámetros por defecto.
  
- **KNN:** se realizó un ajuste mediante Grid Search, probando distintos valores del número de vecinos [3, 5, 7, 9, 11] sobre el conjunto de entrenamiento, y seleccionando la configuración óptima en función del accuracy.
  
- **Red neuronal:** para el entrenamiento se utilizó un tamaño de batch de 64 muestras. En cuanto a la arquitectura, la red neuronal está compuesta por tres capas totalmente conectadas, con normalización por lotes y función de activación ReLU en las capas intermedias. 

El entrenamiento se realizó mediante el optimizador Adam y se fijó un máximo de 15 épocas utilizando la estrategia early stopping, para que si en 3 épocas no mejora el AUC en validación para el entrenamiento. 

Durante la evaluación se aplica una función sigmoide para obtener probabilidades asociadas a la clase positiva. La función de pérdida utilizada fue BCEWithLogitsLoss.

- **Modelo Transformer con fine-tuning:** se empleó el modelo preentrenado BERT-base uncased, adaptándolo a nuestro proyecto mediante un proceso de fine-tuning orientado a la clasificación de noticias reales y falsas. Para ello, se ajustaron los parámetros del modelo utilizando nuestro conjunto de entrenamiento. Una vez entrenado, se realizaron predicciones sobre el conjunto de test y se obtuvieron las métricas correspondientes, permitiendo evaluar su rendimiento y compararlo con el resto de modelos generados.

## 5.Resultados experimentales y discusión.


A lo largo de esta sección se presentan los resultados obtenidos para evaluar el rendimiento de los distintos modelos. El objetivo principal ha sido identificar el modelo más adecuado para su posible integración en un sistema de validación de la veracidad de noticias, asegurando la confianza  en las noticias validadas.

Para la evaluación se emplearon distintas métricas de rendimiento, priorizando la precisión de la clase 0 (noticias reales), ya que para este contexto es especialmente relevante, y también el accuracy, proporcionando una medida sobre el rendimiento global del proyecto.

A continuación se encuentran los resultados obtenidos, donde se compara cada uno de los modelos de clasificación para cada técnica de representación vectorial del texto.

### TF-IDF
<img width="642" height="112" alt="image" src="https://github.com/user-attachments/assets/1daa2ebc-d407-4fee-9f1f-e31c52eb616d" />

El modelo SVM alcanza la mayor precisión en la clase 0, es decir, es el más preciso clasificando las noticias verdaderas. No obstante, la red neuronal también ofrece un rendimiento muy competitivo, con resultados cercanos a los obtenidos por SVM.
En el caso de la regresión logística, se obtienen también buenos resultados, aunque inferiores a los de los modelos anteriores. Por su parte, el clasificador KNN es el que presenta el peor comportamiento, especialmente en la detección de noticias verdaderas, lo que limita su utilidad en este contexto.

En general, se observa un buen rendimiento global de los modelos evaluados, con la excepción de KNN, cuyo desempeño es notablemente inferior al del resto. Atendiendo a la precisión en la clase 0, los modelos pueden ordenarse de la siguiente forma: SVM>RN>LG>KNN

##### Mejor modelo clasificador para TF-IDF: SVM


### Word2Vec
<img width="643" height="112" alt="image" src="https://github.com/user-attachments/assets/7a448159-56f7-4abd-96b2-41eca2f53adb" />

Los resultados muestran que la red neuronal es el modelo que obtiene el mejor rendimiento global, alcanzando una precisión de 0,92 en la clase 0 y la mayor accuracy (0,9316) entre los modelos evaluados. Esto indica una buena capacidad del modelo para aprovechar representaciones densas y capturar relaciones semánticas relevantes.

El modelo SVM y la regresión logística presenta un rendimiento competitivo, situándose como alternativas de la red neuronal.

Por último, el clasificador KNN vuelve a ser el modelo con peor comportamiento, especialmente en términos de precisión para detectar noticias verdaderas, lo que indica una mayor tasa de errores.

Ordenando los modelos según su capacidad para clasificar correctamente noticias reales se nos quedaría la siguiente distribución: RD>SVM>LG>KNN 

##### Mejor modelo clasificador para Word2Vec: Red Neuronal

### BERT
<img width="644" height="114" alt="image" src="https://github.com/user-attachments/assets/a60c6ddd-a293-4661-b55f-6650469711e7" />

Los resultados obtenidos muestran que la red neuronal es el modelo con mejor rendimiento al combinarse con embeddings BERT. Los modelos SVM y regresión logística presentan rendimientos similares, con precisiones de 0,89 y 0,88 respectivamente en la clase 0. Aunque estos resultados son aceptables, se sitúan por debajo de los obtenidos por la red neuronal.

Por su parte, KNN continúa mostrando un comportamiento inferior al resto de modelos, con una precisión de 0,81 en la clase 0.

Atendiendo a la precisión en la clase 0, los modelos pueden ordenarse de la siguiente manera:RD>SVM>LG>KNN.

##### Mejor modelo clasificador para BERT: Red Neuronal

 
#### Modelo Transformer con fine-tuning

<img width="683" height="185" alt="image" src="https://github.com/user-attachments/assets/253e8cbf-307c-4e23-97ef-9e58013706b5" />
Este modelo obtiene el mejor rendimiento entre todas las técnicas evaluadas, alcanzando una accuracy del 99,31%. La precisión de la clase 0 (noticia real) es 0,9918, mientras que para la clase 1 (noticia falsa) asciende a 0,9945, lo que muestra que el modelo comete muy pocos falsos positivos y falsos negativos. Estos resultados hacen evidente la ventaja de los modelos transformer, capaces de capturar  dependencias semánticas y contextuales complejas.

## 7. Conclusiones

BERT ofrece la representación más avanzada, aprendiendo contextualmente mediante relaciones sintácticas y desambiguación semántica.

al utilizar En este caso, además, se aplicó fine-tuning, permitiendo que el modelo ajuste sus pesos.
Entre los clasificadores evaluados, la Red neuronal destaca de forma clara, superando al resto de  clasificadores. Por un lado,  SVM y Logistic Regression obtienen rendimientos aceptables, aunque limitado por su linealidad. En contraste, KNN vuelve a mostrar el rendimiento más bajo, ya que es especialmente sensible a vectores de alta dimensión.

BERT  es capaz de captar matices lingüísticos que TF-IDF y Word2Vec, lo que resulta clave en textos ambiguos o donde la manipulación informativa depende del tono, el contexto o la selección de palabras.

## 8. Bibliografía 

YouTube. [Cómo hacer fine-tuning de un transformer] [Video]. YouTube. https://www.youtube.com/watch?v=YwfRtgcUddw

DataCamp. (s.f.). PyTorch tutorial: Building a simple neural network from scratch. https://www.datacamp.com/es/tutorial/pytorch-tutorial-building-a-simple-neural-network-from-scratch
