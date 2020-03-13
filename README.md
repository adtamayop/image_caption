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

