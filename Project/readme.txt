EL código de demo muestra una imágen (slice de volumen)  aleatoria con su respectiva segmentación y groundtruth.

El código de test corre la red sobre los datos de test y devuelve los índicea de jaccard individuales para cada volumen y su promedio.

Ambos códigos descargan la base de datos y los pesos de la red entrenada. Esta información una vez descargada no se vuelve a descargar siempre y cuando el código esté en la misma carpeta que los datos. Tener en cuenta que la descarga de la base de datos y los pesos toma bastante tempo.

