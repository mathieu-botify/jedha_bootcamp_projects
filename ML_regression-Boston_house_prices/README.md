# Prédictions de prix via des modèles de regression linéaire simple et Ridge

## Introduction

On souhaite établir des prédictions de prix à partir d'un dataset des informations relatives aux maisons à Boston.

Ce dataset est composé de 13 variables quantitaves et qualitatives distinctes (ex: proposition de résidences / sociétés dans le quartier, taux de criminalité, ...) ainsi que d'une cible "prix médian".

Pour cela, deux modèle de machine learning seront utilisés :
* regression linéaire simple : score obtenu sur échantillon test => 71,3 %  
* regression Ridge avec deux valeur d'Alpha : scores obtenus => 71,3 % pour un alpha à 0.01 / 72,3 pour un alpha à 10

Le code utilisé sera détaillé ci-après.

## Description code

Dans un premier temps, on préparer les datasets pour l'entrainement de nos modèles de ML.

```
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
boston=load_boston()

# Création des DataFrames pour l'entrainement
# 1/ Variables explicatives
boston_df=pd.DataFrame(boston.data,columns=boston.feature_names)

# 2/ Target
target = pd.DataFrame(boston.target, columns=["MEDV"])

# 3/ Séparation des données en train/test
import sklearn.cross_validation as cv
train_X, test_X, train_y, test_y = cv.train_test_split(boston_df, target, test_size = 0.3)
train_index = train_X.index
test_index = test_X.index

# 4/ Normalisation des variables explicatives

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc = sc.fit(train_X)
train_X = sc.transform(train_X)
test_X = sc.transform(test_X)

train_X = pd.DataFrame(train_X, index = train_index, columns = boston_df.columns)
test_X = pd.DataFrame(test_X, index = test_index, columns = boston_df.columns)
```

On construit les modèles de ML en commençant par la régression linéaire simple.

Scores obtenus => train: 0.740 / test: 0.713

```
from sklearn import linear_model as lm

# Régression linéaire

model_lin = lm.LinearRegression()
model_lin = model_lin.fit(train_X, train_y)
score_reglin = model_lin.score(train_X, train_y)
score_reglin_test = model_lin.score(test_X, test_y)
print(("Score train: {}, Score test: {}").format(score_reglin, score_reglin_test))
 
```

Puis la régression Ridge avec dans un premier temps un paramètre alpha fixé à 0.01

Scores obtenus =>  train: 0.740 / test: 0.713


```
# Coefficients régression ridge (alpha à 0.01)
coef_ridge001 =  pd.DataFrame(index = range(0,len(train_X.columns)), columns = ['coef','index','model'])
coef_ridge001['coef'] = model_ridge001.coef_[0]
coef_ridge001['index'] = coef_lin['index']
coef_ridge001['model'] = 'ridge001'
```


Puis un paramètre alpha fixé à 10

Scores obtenus =>  train: 0.738 / test: 0.723


```
model_ridge10 = lm.Ridge(alpha = 10)
model_ridge10 = model_ridge10.fit(train_X, train_y)
score_ridge10 = model_ridge10.score(train_X, train_y)
score_ridge10_test = model_ridge100.score(test_X, test_y)
print(("Score train: {}, Score test: {}").format(score_ridge10, score_ridge10_test))
```

** Optionnel **

On calcule les coefficients appliqués par les modèles aux variables.

```
# Coefficients régression linéaire
coef_lin = pd.DataFrame(index = range(0,len(train_X.columns)), columns = ['coef','index','model'])
coef_lin['coef'] = model_lin.coef_[0]
coef_lin['index'] = range(0,len(train_X.columns))
coef_lin['model'] = 'reg_lin'

# Coefficients régression ridge (alpha à 0.01)
coef_ridge001 =  pd.DataFrame(index = range(0,len(train_X.columns)), columns = ['coef','index','model'])
coef_ridge001['coef'] = model_ridge001.coef_[0]
coef_ridge001['index'] = coef_lin['index']
coef_ridge001['model'] = 'ridge001'

# Coefficients régression ridge (alpha à 100)
coef_ridge10 =  pd.DataFrame(index = range(0,len(train_X.columns)), columns = ['coef','index','model'])
coef_ridge10['coef'] = model_ridge100.coef_[0]
coef_ridge10['index'] = coef_lin['index']
coef_ridge10['model'] = 'ridge100'

# Ajout des coefs dans un DataFrame
compar_coef = pd.concat([coef_lin, coef_ridge001, coef_ridge10])
```

Puis, on les visualise via la librairie Seaborn

```
import seaborn as s
s.pointplot(x = 'index',y = 'coef',hue = 'model', style = 'model', data = compar_coef)
```