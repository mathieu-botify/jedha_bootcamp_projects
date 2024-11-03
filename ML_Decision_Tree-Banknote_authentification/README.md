# Classification via un modèle de decision tree

## Introduction

Nous souhaitons classer un jeu de données contenant des billets de banque authentiques ou faux.

Ce dataset est composé de 5 variables indépendantes distinctes ainsi que d'une cible "vrai/faux".
Les données sont téléchargeables ici : http://archive.ics.uci.edu/ml/datasets/banknote+authentication 

Pour cette classification, un modèle type "decision tree" a été utilisé.
Score obtenu sur l'échantillon de test => 99,27 %

Le code sera détaillé ci-après.

## Description code

1/ Préparation des datasets pour l'entrainement de notre modèle de ML.

```
# Import librairies
import pandas as pd
import sklearn

# Import des datasets
filename = 'data_banknote_authentication.csv'
dataset = pd.read_csv(filename, header=None, sep=";")
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]
```

2/ Découpage de notre dataset en données de train et de test 

```
# Création de datasets de train et test 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
```

3/ Entrainement du modèle.

Score obtenu => 99,27 %

```
# Entrainement du modèle de decision tree Classifier
from sklearn.tree import DecisionTreeClassifier
tree = sklearn.tree.DecisionTreeClassifier()
tree_fit = tree.fit(x_train, y_train)

# Score sur l'échantillon de test
tree_fit.score(x_test, y_test) 
```

4/ Affichage de la matrice de confusion

Remarque : pas de faux négatifs obtenus
```
# Confusion matrix
import sklearn.metrics
confusion_matrix = sklearn.metrics.confusion_matrix(y_test, predictions)
print(pd.DataFrame(confusion_matrix))
```
