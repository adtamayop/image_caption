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

