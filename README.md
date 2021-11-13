# Clasificación de la calidad del aire
Reto Data Science para la Hackaton de la II Feria de Empleo de la UOC

## Reto a desarrollar
El Acuerdo de París es un tratado internacional sobre el cambio climático que fue adoptado por 196 Partes en la COP21 de París. Su objetivo es limitar el calentamiento mundial muy por debajo de 2, preferiblemente a 1,5 grados centígrados, en comparación con los niveles preindustriales.

Para alcanzar este objetivo de temperatura a largo plazo, los países se proponen alcanzar el máximo de las emisiones de gases de efecto invernadero lo antes posible para lograr un planeta con clima neutro para mediados de siglo.

Es por ello que la Unión Europea esta destinando grandes cantidades de recursos al desarrollo de nuevas tecnologías que permitan la mejorar la lucha contra la contaminación. Una de estas es un nuevo tipo de sensor basado en tecnología láser que permita detectar la calidad del aire en función de diferentes mediciones.

## Datasets

Variables del dataset:

- Features: El dataset contiene 8 features en 8 columnas que son los parámetros medidos por los diferentes sensores. Estos corresponden a las diferentes interacciones que han tenido los haces de los láseres al travesar las partículas del aire.
- Target: El target corresponde al 'label' que clasifica la calidad del aire.

Target 0 corresponde a una calidad del aire Buena.
Target 1 corresponde a una calidad del aire Moderada.
Target 2 corresponde a una calidad del aire Peligrosa.

## Archivos:

- `uoc_train.csv`: Este dataset contiene tanto las variables predictoras como el tipo de clasificación de calidad del aire.
- `uoc_test_X.csv`: Este dataset contiene las variables predictoras con las que se tendrá que predecir el tipo de calidad de aire.

## Objetivo
El objetivo del reto será realizar un modelo predictivo basado en Random Forests que permita conocer el tipo de calidad del aire en función de las mediciones de los sensores.

Una vez se haya hecho y entrenado el modelo predictivo, éste se tendrá que emplear con los features del dataset de testing `uoc_test_X.csv`. Las predicciones se tendrán que entregar en formato `csv`. Donde tendrá que aparecer tan solo una columna en la que en la primera fila habrá un texto cualquiera y las predicciones empezarán en la fila 2.

La calidad de la predicción se medirá a partir del f1-score(macro).
