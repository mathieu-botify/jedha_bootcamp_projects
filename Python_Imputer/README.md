# Fonction d'imputation

## Introduction

L'objectif de cette fonction de remplacer les valeurs manquantes d'une liste par sa valeur moyenne ou sa médiane.

## Code

Création d'une classe "Imputer" dans laquelle seront encapsulées deux fonctions "avg" et "med"

```
class Imputer:

    def __init__(self, liste):
        self.liste = liste
```

La **fonction "avg"** calcule la valeur moyenne d'une liste et remplace les valeurs manquantes par cette moyenne.

Etape 1, calcul de la moyenne via une boucle for :
* On fait la somme des valeurs non nulles
* On définit le nombre de valeurs non nulles
* Puis, on calcule la moyenne

Etape 2, imputation de la moyenne à la place des données manquantes (via une boucle for).

```
Création d'une classe "Imputer" dans laquelle sera seront encapsulées deux fonctions "avg" et "med"

    #Calcul de la moyenne
    
    def avg(self):
        sum = 0
        n = 0

        for i in range(len(self.liste)):
            if self.liste[i] != None:
                sum += self.liste[i]
                n += 1

        moyenne = round((sum / n),2)
        
        for i in range(len(self.liste)):
            if self.liste[i] == None:
                self.liste[i] = moyenne
        return self.liste
```


La deuxième **fonction "med"** calcule la valeur médiane d'une liste et remplace les valeurs manquantes par cette médiane.

Etape 1, création d'une nouvelle liste ordonnée par ordre croissant, sans les valeurs manquantes, via une boucle for.

Etape 2, calcul de la médiane (via une boucle for):
* On définit si la taille de notre liste est paire ou impaire.
* Suivant sa taille, on sélectionne la valeur centrale (longueur impaire)
* Ou, on calcule la médiane à partir des deux valeurs centrales

Etape 3, imputation de la médiane à la place des données manquantes (via une boucle for).

```
    def med(self):
    #Ajout médiane à la place des données "None" 

        n = 0
        liste_temp = []
        liste_croissante = []

        #création liste temporaire avec uniquement les data ok

        for i in range(len(self.liste)):
            if liste[i] != None:
                liste_temp.append(self.liste[i])
                n += 1

        #création d'une liste croissante
        liste_croissante = sorted(liste_temp)

        #cacul de la médiane
        if n % 2 != 0:
            mid = int(((n+1)/2) - 1)
            mediane = liste_croissante[mid]
        else:
            mid = int((n/2) - 1)
            mediane = ((liste_croissante[mid] + liste_croissante[mid + 1]) / 2)

        #insertion de la médiane à la place des None

        for i in range(len(self.liste)):
            if self.liste[i] == None:
                self.liste[i] = mediane
```
