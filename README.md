# Photo caption generator

Este problema consiste en la generación de subtítulos de imágenes, es decir una descripción legible y concisa de los contenidos de una fotografía.

La generación de descripciones para una imagen requiere tanto los métodos de la visión por computadora para comprender el contenido de la imagen como un modelo de lenguaje del campo del procesamiento del lenguaje natural para convertir la comprensión de la imagen en palabras en el orden correcto.

El Dataset a usar conseta de 8000 imágenes, y cada imágen tiene 5 descripciones diferentes

Los pasos necesario para entrentar este problema es:

1. Preprocesar las imágenes
2. Preprocesar el texto
3. Desarrollo del modelo:
    1. Cargar datos
    2. Definir el modelo
    3. Entrenar el modelo
    4. Evaluar el modelo
4. Generación de descripciones


## 1. Preprocesamiento de las imágenes:

Se va a utilizar transfer learning para interpretar el contenido de las fotos, en este caso la arquitectura VGG16

<p align="center">
  <img src="https://www.researchgate.net/profile/Max_Ferguson/publication/322512435/figure/fig3/AS:697390994567179@1543282378794/Fig-A1-The-standard-VGG-16-network-architecture-as-proposed-in-32-Note-that-only.png" width="400">
</p>

En el entrenamiento del modelo podríamos pasar cada imagen que vamos a procesar por el modelo preentrenado y unirla a la arquitectura de image caption que se va a proponer más adelante, pero para ahorrar tiempo y recursos podemos pre calcular las "photo features" usando el modelo preentrenado y guardar estas interpretaciones por el modelo en un archivo y luego cargarlas, para meter esa interpretacion a nuestro modelo, que sería lo mismo que por cada imagen pasarla por el modelo VGG solo que lo haremos de forma anticipada, esto hará el modelo más rápido y consumirá mucha menos memoria.

Para extraer correctamente las características de la foto se remueve la última capa de la red preentrenada, que sería la parte encargada de la red que hace la clasificación, pero en este problema no estamos interesados en clasificar las imágenes, sino en la interpretación interna de la foto, que es lo que se hace justo antes de clasificarla, allí están las "características" que el modelo ha extraído de la foto.

## 2. Preprocesamiento del texto

El dataset contiene multiples desciptiones por foto y la descripciones requieren algo de limpieza,así que se pasa todo minusculas, quitar puntuaciones, palabras de una sola letras, palabras con numeros dentro de ellas, entonces una vez ya hemos limpiado el vocabulario, podemos resumir el tamaño del vocabulario, lo ideal es que el vocabulario que tenemos sea tan expresivo y pequeño como sea posible, un vocabulario pequeño resultará en un modelo más pequeño que entrenaremos más rápido

Guardamos un diccionario de las descripciones de cada imagen por identificador de cada imagen en un archivo llamado description.txt, con un identificador y descripción por línea

## 3. Construcción del modelo

El modelo que se va a desarrollar generará una descripción dada una foto, pero esta descripción se va a construir una palabra al tiempo, entonces cada palabra que se vaya generando de la descripción se le vuelve a ingresar a la entrada del modelo de forma recurrente, entonces se va a utilizar una palabra que va "igniciar" el proceso de generación y una última palabra para darle la señal de que termine la descripción, en este casó será las palabras 'startseq' y 'endseq'

                    X1,	X2 (Secuencia de palabras)                   y (palabra)
                    photo	startseq,                                       little
                    photo	startseq, little,                               girl
                    photo	startseq, little, girl, 	                running
                    photo	startseq, little, girl, running,                in
                    photo	startseq, little, girl, running, in,            field
                    photo	startseq, little, girl, running, in, field,     endseq

<p align="center">
  <img src="https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/10/Recursive-Framing-of-the-Caption-Generation-Model.png" width="400">
</p>

Entonces cuando el modelo sea usado para generar descripciones, las palabras generadas serán concatenadas, y recursivamente serán utilizadas como entrada para generar una descripción para cada imagen.

Se transforman los datos a supervisados, entonces se tiene un array de features de las fotos, y otro con el texto codificado 

El texto de entrada es codificado como entero, el cual será alimentado a una capa "word embedding", entonces las características de la foto será alimentadas directamente a otra parte del modelo, el modelo sacará una predicción la cual es una distribución de probabilidad sobre todas las palabras del vocabulario, los datos de salida será por lo tanto un one-hot encoded del vocabulario, de cada palabra, representando una probabilidad de distribution idealizada con valores de 0 para todas las posiciones de palabras.

las arquitecturas encoder-decoder son referentes en la solución de problema de este tipo (traducción, problemas de índole secuencial), en donde una red codifica datos, y otra los interpreta.

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/1000/1*1JcHGUU7rFgtXC_mydUA_Q.jpeg" width="400">
</p>

En nuestro caso se tienen 2 elementos básicos:

* Encoder: Una red que lee la fotografía y codifica el contenido en un fector de tamaño fijo, usando una representación interna

* Decoder: Una red que lee la fotografía codificada y genera la descripción

Normalmente lo que se hace es coger una red convolucional para codificar la imagen y una red recurrente como una LSTM por ejm, para bien sea codificar la secuencia de entrada y/o generar la siguiente palabra en la secuencia 

**Marc Tanti** Propuso una arquitectura, muy efectiva para generar image caption llamada **Merge-model**

El modelo de merge combina la forma codificada de la imagen de entrada con la forma codificada de la descripción de texto generada hasta ahora,la combinación de estas dos entradas codificadas es utilizada por un modelo de decodificador muy simple para generar la siguiente palabra en la secuencia,el enfoque utiliza la red neuronal recurrente solo para codificar el texto generado hasta ahora.

<p align="center">
  <img src="https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/10/Merge-Architecture-for-the-Encoder-Decoder-Model.png" width="400">
</p>

Esto separa los features de la entrada de imagen, la entrada de texto y la combinación e interpretación de las entradas codificadas.

Como se mencionó, es común usar un modelo previamente entrenado para codificar la imagen, pero de manera similar, esta arquitectura también permite usar un modelo de lenguaje previamente entrenado para codificar la entrada de texto de subtítulos.


El modelo se puede describir en 3 partes:

* El extractor de características de la foto: en este caso la VGG16 sin la capa de salida
* El procesador de secuencias: Esta es una capa de  word embedding (n dónde las palabras o frases del vocabulario son vinculadas a vectores de números reales) para manipular el texto de entrada seguida por una lstm 
* Decoder, el extractor y el procesador sacan un vector de tamaño fijo, estos son combinados y procesador por una red densa para hacer una predicción final

El modelo Photo Feature Extractor espera que las características de entrada de fotos sean un vector de 4.096 elementos. Estos son procesados por una capa densa para producir una representación de 256 elementos de la foto.

El modelo del procesador de secuencia espera secuencias de entrada con una longitud predefinida (34 palabras) que se introducen en una capa de embedding que utiliza una máscara para ignorar los valores rellenados. Esto es seguido por una capa LSTM con 256 neuronas.

Ambos modelos de entrada producen un vector de 256 elementos. Además, ambos modelos de entrada utilizan la regularización Dropout 50%. Esto es para reducir el sobreajuste del conjunto de datos de entrenamiento, ya que esta configuración de modelo aprende muy rápido.

El modelo Decoder combina los vectores de ambos modelos de entrada utilizando una operación de suma. Esto luego se alimenta a una capa de neuronas 256 densas y luego a una capa densa de salida final que hace una predicción softmax sobre todo el vocabulario de salida para la siguiente palabra en la secuencia.



**Evaluamos el modelo:**

Una vez entrenamos el modelo, podemos meterle nuestro dataset de prueba

Entonces lo primero es evaluar el modelo generando las descripciones para todas las fotos en el dataset de puebas y evaluando esas predicciones con una función de costo estandar

Las descripciones reales y pronosticadas se recopilan y evalúan colectivamente utilizando la puntuación BLEU del corpus que resume lo cerca que está el texto generado del texto esperado.

La métrica BLEU es usada para evaluar traducción de texto, evaluando el texto que debió ser, contra el predicho.


En este caso se calculo el BLEU score para 1, 2, 3 y 4 n-gramas acumulativos, un puntaje más cerca de 1 es mejor y 0 es peor




