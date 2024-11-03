# Prédiction de ventes via un modèle type "time series"

## Introduction

On souhaite établir des prédictions de vente moyenne à partir d'un dataset contenant des informations sur les ventes d'un site web.

Dans cet exemple, nous focaliserons notre prédiction sur une catégorie spécifique à savoir "furniture". 

Les données du dataset seront modifiées dans cette optique.

Enfin, nous afficherons notre prédiction via un graphique des ventes mensuel avec une prédiction sur l'année N+1.

**Résultats (prédictions en rouge) :**
 
![alt text](https://github.com/MathieuBerthier/ML_Time_Series-Superstore/blob/master/img/prediction_with_seasonnality.png)

Le code utilisé est détaillé ci-après.

## Description code

Dans un premier temps, on importe une partie des librairies ainsi que le dataset.

```
# Import des librairies
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd

# Import du dataset
df = pd.read_excel("superstore.xls")
```


On extrait les données de la catégorie "furniture" qui sera utilisée pour la prédiction.

```
# Préparation des données

""" Création d'un objet "furniture" qui contient toutes les lignes de la variable Category égales à "furniture" """
furniture = df.loc[df['Category'] == "Furniture"]

""" On Conserve uniquement les colonnes ‘Order Date’ et ‘Sales’ dans le dataset furniture"""
furniture = furniture.loc[:,['Order Date','Sales']]

""" Classement par ordre croissant des dates via .sort_values"""
furniture = furniture.sort_values(by=['Order Date'], ascending=True)
```

A partir des données sont calculées les ventes quotidiennes de la catégorie "furniture" puis les ventes moyennes mensuelles.

```
""" Calcul la somme des ventes par jour """
furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()

""" Transformation des dates en index du dataset """
furniture = furniture.set_index('Order Date')

""" Calcul des ventes moyenne par mois """
y = furniture['Sales'].resample('MS').mean() 
```

Première prédiction graphique avec un lissage exponentiel double.
* Comme on peut le voir, le résultat est peu adapté

```
'''Prédiction avec lissage exponentiel double'''

from statsmodels.tsa.holtwinters import ExponentialSmoothing
DES = ExponentialSmoothing(y, trend = 'add')
DES_fit = DES.fit(smoothing_level=0.5)
DES_predict = DES_fit.predict(start=0, end = 60)
sns.lineplot(x = y.index, y = y )
sns.lineplot(x = DES_predict.index, y = DES_predict )
```
 ![alt text](https://github.com/MathieuBerthier/ML_Time_Series-Superstore/blob/master/img/prediction_without_seasonnality.png)
 
On fait une nouvelle prédiction avec un lissage exponentiel + prise en compte de la saisonnalité (calculé sur une annnée).
* Ici, les résultats sont pertinents.

```
'''Prédiction avec lissage exponentiel + saisonnalité '''
from statsmodels.tsa.holtwinters import ExponentialSmoothing
TES = ExponentialSmoothing(y, trend = 'add', seasonal = 'add', seasonal_periods = 12)
TES_fit = TES.fit()
TES_predict = TES_fit.predict(start=0, end = 60)
sns.lineplot(x = y.index, y = y)
sns.lineplot(x = TES_predict.index, y = TES_predict)
```
 
![alt text](https://github.com/MathieuBerthier/ML_Time_Series-Superstore/blob/master/img/prediction_with_seasonnality.png)
