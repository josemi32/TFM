# TFM


En los últimos años, el avance del deep learning
mediante redes neuronales ha transformado numerosos campos,
incluyendo la mejora automática de imágenes submarinas. Este
trabajo de fin de máster propone el uso de redes neuronales para
el análisis y la mejora de la calidad de imágenes submarinas,
incorporando la subjetividad humana en su entrenamiento.
Utilizamos para ello conjuntos de datos públicos, que contienen
imágenes de buena y mala calidad evaluadas por expertos.
Proponemos un enfoque en el que primero se entrena una red
clasificadora para distinguir entre imágenes de buena y mala cali-
dad, y posteriormente se entrenan redes generativas adversariales
(GAN) incorporando diferentes criterios de mejora para procesar
las imágenes submarinas de mala calidad. Evaluamos los modelos
GAN usando métricas cuantitativas como PSNR, SSIM y UIQM,
y análisis cualitativos. Los resultados muestran que el modelo
final, propuesto, que incluye criterios como la calidad del color y
la nitidez de la imagen resultante, produce mejoras significativas
en la calidad de las imágenes, tanto numérica como visualmente.


- En la carpeta Proyectos:

   - En el archivo Entreno del Clasificador, se encuentra todo lo referente al entrenamiento correspondiente del clasificador desarrollado en este proyecto
   - ValidacionClasificador, es la validacion del clasificador sobre el conjunto de datos UIEB.
   - UsoDeClasificador, es un ejemplo de como poder usar el clasificador y asi poder evaluar imagenes marinas.
   - GanNoReference, es el entrenamiento del modelo GAN sin usar imagenes de referencia, el cual no genero buenos resultados.
   - Gan, se encuentra todo lo referente al entreno de la GAN con el uso de imagenes de referencia y su validación y calculo de diferentes metricas
   - UsoDeGan, es un ejemplo de como poder usar el model GAN para la mejora automatica de imagenes marinas.


- Pesos:

   - En esta carpeta se encuentran todos los pesos de todos los modelos desarrollados.
   - El primero es del clasificador/discriminador desarrollado para poder usarlo en cualquier momento.
   - Luego se encuentra una separacion en dos carpetas, una con el peso del modelo GAN sin referencias y la otra con los pesos de los modelos entrenados con la referencia, cada uno en su carpeta pertinente.

- ImagenesPruebas:

   - En esta carpeta se encuentra dos imagenes de prueba para el clasificador y para el generador que se muestran mas adelante, para que se pueda ver que salida dan.







Ejemplo de clasificacion por parte del Clasificador:

![im_f1000_](https://github.com/josemi32/TFM/assets/74961648/d604a73e-69c3-4a3f-bfe5-36bf01d03928)

Puntuacion: 1 , fue clasificada como de mala calidad.

![im_f1000_](https://github.com/josemi32/TFM/assets/74961648/e2c78897-9789-4ddb-8ca8-b5220fd06604)

Puntuacion:0, fue clasificada como de buena calidad.
