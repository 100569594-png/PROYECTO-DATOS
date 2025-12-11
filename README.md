 # Análisis de la polarización ideológica en redes sociales a partir de la propagación de contenidos desinformativos

## 1.Descripción del problema y conjunto de datos.

Uno de los principales problemas que enfrentan las redes sociales y los medios digitales es la desinformación.  La verificación manual de la información se vuelve más difícil y la polarización ideológica se ve favorecida por la rápida propagación de contenidos falsos.  Este proyecto tiene como finalidad detectar automáticamente noticias falsas (fake news) mediante el análisis de su contenido textual, analizando distintas estrategias de representación vectorial y clasificadores supervisados.

 La base de datos empleada proviene del WELFake Dataset, que se compone de miles de noticias categorizadas como verdaderas (label = 0) o falsas (label = 1).  Cada entrada incluye el contenido textual de la noticia, que ha sido purificado y limpiado. Después de suprimir documentos sin texto, así como los que eran demasiado breves o excesivamente extensos (menos de 200 palabras o más de 2000), se filtró el dataset de datos para asegurar su coherencia y calidad, quedando listo para ser analizado.

### Asimismo, se llevó a cabo un análisis de exploración que comprende:

 - Distribución de las clases.

 - Longitud de los textos y sus histogramas correspondientes.

 - Términos más comunes en cada clase.

 - Nubes de palabras para identificar patrones léxicos que diferencien.

El propósito final es identificar la metodología que ofrece el rendimiento más alto en la detección automatizada de desinformación.

## 2.Explicación de las metodologías utilizadas.

## 2.1. Preprocesamiento del texto

## 2.2. Representación vectorial del texto

- TF-TDF

- Word2Vec
- BERT (embeddings contextuales)


## 2.3. Modelos de clasificación

## 3.Resultados experimentales.



## Discusión

## 4. Conclusiones
