# Classification d'acheteurs/prospects via XG Boost + stacking

## Introduction

Une banque Portugaise a lancée une campagne de marketing direct (téléphonique) afin d’inciter des clients à faire un dépôt à terme.

Elle a collectée une quantité importante de données via cette campagne.

Notre objectif sera donc d'identifier les clients susceptibles de souscrire à une offre via un algorithme de classification.

Résultat final obtenu : 94,97% de prédiction correctes.

Le code utilisé est détaillé ci-après.


## Description code

Dans un premier temps, on importe une partie des librairies et notre dataset.

```
import pandas as pd
import numpy as np
import os

os.chdir("repertoire_contenant_le_dataset")
df = pd.read_csv("bank-additional-full.csv", sep=";")

X = df.iloc[:,0:-1]
y = df.iloc[:,-1]
```


Etape 2, preprocessing des données avec :
* vérification de présence de valeur nulles ou non
* transformation des variables catégoriques
* feature scaling
* Split du dataset en train et test

```
""" Vérification si valeur null """
null_data = X[X.isnull().any(axis=1)]
# pas de valeur null

""" Encodage des catégories avec Get_dummies """

# D'abord X
cat_columns = ['job', 'marital', 'education','default','housing','loan','contact','month','day_of_week','poutcome']
    # Remarque : j'ai récupéré les col à convertir en faisant toutes les col moins celles dans ma variable description
X_dummies = X[cat_columns]
X_dummies = pd.get_dummies(X_dummies, drop_first=True)

# Puis y
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


""" Feature scaling """

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_stand = X[description.columns]
X_stand = sc.fit_transform(X_stand)
X_stand = pd.DataFrame(X_stand, columns = description.columns)

""" Concatenation des données transformées """

X = pd.concat([X_stand, X_dummies], axis=1, join_axes=[X_stand.index])

""" Découpage dataset X train et X test """
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)

y_count = pd.Series(y)
nb_0 = y_count.where(y_count == 0).count()
nb_1 = y_count.where(y_count == 1).count()
print("nb_0 : {} soit {} % \nnb_1 : {} soit {} %".format(nb_0, (nb_0 / y_count.count()), nb_1, (nb_1 / y_count.count())))
```

***important : 88% des personnes ont refusées l'offre. Nous devrons donc obtenir plus de 88% de prédictions correctes pour être meilleur que le hasard.***



**Premier modèle de classification - "Random forest"**
* Score obtenu : 91,88 %

Résultat à améliorer.

```
""" Entrainement du modèle - avec cross validation"""

# Import modèle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Spécification des hyper paramètres
parameter_candidates = [
  {'n_estimators' : [100], 'min_samples_split': range(2, 5), 'min_samples_leaf': range(1, 5), 
   'max_depth' : range(1, 5), 'random_state':[42]}]

rf_gs = GridSearchCV(RandomForestClassifier(), param_grid=parameter_candidates, cv=5, scoring='roc_auc' ,n_jobs=-1, verbose=1)

# On entraîne le modèle créer sur les données d'apprentissage
rf_gs_fit = rf_gs.fit(X_train, y_train)

print('Best score :', rf_gs.best_score_)
# Best score 91,88 %
print('Best n_estimators:',rf_gs.best_estimator_.n_estimators) 
print('Best min_samples_split:',rf_gs.best_estimator_.min_samples_split) 
print('Best min_samples_leaf:',rf_gs.best_estimator_.min_samples_leaf)
print('Best max_depth:',rf_gs.best_estimator_.max_depth)

rf_importances = rf_gs.feature_importances_

""" Confusion matrix """
from sklearn.metrics import confusion_matrix
y_predict_rf =  rf_gs.predict(X_test)
confusion_matrix(y_test, y_predict_rf)
```

**Modèle de classification - "Regression logistique"**
* Score obtenu : 93,39 %

Gain de précision mais à améliorer.

```
# Import fonction
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

parameters_lr = [  {'C' : [0.1,1,10,100,1000],
                    'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],                 
                    'random_state':[42]}]

lr_gs = GridSearchCV(LogisticRegression(), param_grid=parameters_lr, scoring="roc_auc", cv=5, n_jobs=-1, verbose=1)
lr_gs.fit(X_train, y_train)

print('Best score: {}'.format(lr_gs.best_score_))
    # avg roc_auc : 93,39%
print('Best C: {}'.format(lr_gs.best_estimator_.C))
print('Best solver: {}'.format(lr_gs.best_estimator_.solver))


""" Confusion matrix """
from sklearn.metrics import confusion_matrix
y_predict_lr =  lr_gs.predict(X_test)
confusion_matrix(y_test, y_predict_lr)
print("error score : {}".format((800+285) / 12357))
```

**Modèle de classification - "Gradient boosting"**
* Score obtenu : 94,71 %

Gain de précision + réduction des faux négatifs

```
from sklearn.ensemble import GradientBoostingClassifier

parameters_gbc = [  {'learning_rate' : [0.1, 0.5], 'n_estimators' : [100], 
                    'min_samples_split': range(2, 5), 'min_samples_leaf': range(1, 5), 
                    'max_depth' : range(1, 5), 'random_state':[42]} ]

gbc_gs = GridSearchCV(GradientBoostingClassifier(), param_grid=parameters_gbc, scoring="roc_auc", cv=5, n_jobs=-1, verbose=1)
gbc_gs.fit(X_train, y_train)

print('Best score: {}'.format(gbc_gs.best_score_))
    # Normal : avg Accuracy : 94,71 %

y_predict_gbc =  gbc_gs.predict(X_test)
confusion_matrix(y_test, y_predict_gbc)
print("error score : {}".format((628+368) / 12357))
```


**Stacking + XG boost**

Mise en place stacking : on intègre les prédictions des 3 modèles précédents à notre dataset

```
""" Prediction des 3 modèles  """
predict_rf =  pd.Series(rf_gs.predict(X)).rename("predict_rf")
predict_lr = pd.Series(lr_gs.predict(X)).rename("predict_lr")
predict_gbc = pd.Series(gbc_gs.predict(X)).rename("predict_gbc")

""" Ajout des prédictions dans notre dataset """
X_2 = X
X_2 = pd.concat([X_2, predict_rf, predict_lr, predict_gbc], axis=1)

""" Train/test split """
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y, test_size = 0.3, random_state = 42, stratify = y)
```


Utilisation du modèle "XG Boost" sur notre nouveau dataset
* Score final obtenu : 94,97 %

```
from xgboost import XGBClassifier
xgb = XGBClassifier()

xgb_params = {'eta': [0.05, 0.1, 0.5], 'max_depth': range(2, 5),
              'n_estimators': [100], 'seed': [42]}

xgb_gs = GridSearchCV(xgb, param_grid=xgb_params, scoring="roc_auc", cv=5, n_jobs=-1, verbose=1)
xgb_gs.fit(X_train_2, y_train_2)

print('Best score: {}'.format(xgb_gs.best_score_))
    # score : 94,97 %
y_predict_xgb =  xgb_gs.predict(X_test_2)
confusion_matrix(y_test_2, y_predict_xgb)
```
