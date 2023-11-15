# EAYCD-UDEA-Monografia
# **Predicción de Ocurrencia de Accidentes Cerebrovasculares**

La idea principal de este proyecto es construir un modelo capaz de predecir los accidentes cerebro vasculares, siendo éstos la segunda causa de muertes a nivel mundial, razón por la cual despierta el interés de esta investigación. Además, cuenta con su variable objetivo desbalanceada en sus clases en un porcentaje de 95.13 % para la clase mayoritaria y 4.87 % en la clase minoritaria. Los modelos usados fueron la regresión logística, random forest, máquinas de soporte vectoriales, k nearest neighbor y árboles de decisiones. Las métricas principales fueron el f1-score, recall y AUC porque clasifican mejor los casos positivos que son la clase minoritaria. La base de datos fue encontrada en kaggle y posee 5110 registros con 12 variables. Se realizaron cuatro iteraciones; la primera se usó el parámetro class_weight = balanced sin balancear la variable objetivo. La segunda iteración se balanceó dicha variable con la técnica SMOTE y se usaron modelos con parámetros por default. La tercera iteración se usó la técnica de GridSearchCV basado en la métrica f1-score y la última iteración se redujo la dimensionalidad en dos clases más. Los principales obstáculos en este proyecto consistían en lograr mantener la clase minoritaria con la menor pérdida de información posible al aplicar el preprocesamiento y medir la capacidad de generalizar el modelo sin que haya sobreajuste. Se trazó un objetivo de lograr un f1-score del 85 % pero al final el modelo de regresión logística logró llegar hasta 80 % siendo el mejor modelo de entre los evaluados.

## **Desde Kaggle:**

Para usar el código desde kaggle se debe utilizar un API token generado por su perfil de Kaggle para descargar la base de datos directamente desde el sitio web. Deben almacenar el token dentro de su espacio en Google Drive.
En la dirección web donde se encuentra el proyecto [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) deben ir a su perfil que es el logo o foto que usas en el costado derecho en la parte de arriba de la pantalla y allí debes ir a settings y luego buscar el botón Create New Token y como se dijo anteriormente deben almacenarlo en su drive.
Corren este primer código para conectar a colab con su drive:

```python
from google.colab import drive
import os
drive.mount('/content/drive/')
```

Luego deben ejecutar este otro código:

```python
os.environ['KAGGLE_CONFIG_DIR'] = '/content/drive/MyDrive/' + input('Input the directory with your Kaggle json file: ') # Dejar input vacío en caso de que el archivo se encuentre en la raíz de Drive
!kaggle datasets download -d fedesoriano/stroke-prediction-dataset # Descarga del archivo comprimido
!unzip \*.zip && rm *.zip # Descomprensión y eliminación de cualquier archivo .zip
```

No es necesario poner nada, solo darle enter cuando salga el espacio para escribir.

## **Desde GitHub:**

Solo es necesario correr este código de abajo y luego en la parte de cargue de datos hacerlo en el que dice **Cargue de datos desde GitHub**.

```python
!git clone https://github.com/AndresEspinal/EAYCD-UDEA-Monografia.git
```

## **Desde Jupyter:**

Para abrirlo desde Jupyter solo hace falta descargar o ubicar el script en el pc y buscar en la carpeta que fue creado, allí solo debemos poner la base de datos que está localizada en el [Github](https://github.com/AndresEspinal/EAYCD-UDEA-Monografia) en la carpeta BD y luego descargarla. Cuando esté descargada es necesario ubicar la base de datos en la misma carpeta del script. Sigue los pasos normales y en la parte de cargue de datos debes hacerlo desde **Cargue de datos desde el Drive/Kaggle/Jupyter**.

## **Librerías:**

Las librerías que se van a usar o tener en cuenta son las siguientes:

```python
#Datos
import pandas as pd
import numpy as np

#Graficar
import seaborn as sns
import matplotlib.pyplot as plt

#Generador de tablas de contingencia
from pandas import crosstab
from sklearn.metrics import mutual_info_score

#Test chi-cuadrado
from scipy.stats import chi2_contingency

#Imputación
from sklearn.impute import SimpleImputer

#QQ
import statsmodels.api as sm

#Shapiro
from scipy import stats

#Escalamiento
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#K-means
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#Outliers
from sklearn.neighbors import LocalOutlierFactor
from scipy.special import entr #Entropía

#Balanceo
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

#Advertencias
import warnings

#Separación train-test
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid

#Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn import neighbors
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.tree import plot_tree
import graphviz

#Métricas
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
```
## **Bibliotecas:**

En caso de no tener todas las bibliotecas instaladas en su entorno, en la siguiente lista se mostrará las que fueron usadas o próximas a usar:
```python
!pip install pandas
!pip install numpy
!pip install seaborn
!pip install matplotlib
!pip install scikit-learn
!pip install scipy
!pip install statsmodels
!pip install imbalanced-learn
!pip install scikit-plot
!pip install graphviz
```
