# Caculer les intérêts d'un compte d'épargne

## Introduction

L'objectif de cette fonction est de calculer la somme générée (somme déposée + intérêts) par un compte épargne au bout de x années.

Pour cela, nous prendrons en compte 3 éléments :
* la somme déposée sur le compte
* le taux d'intérêt du compte
* le nombre d'années du placement


## Explication code

Au début de notre fonction, on définit un bloc "try" pour renvoyer des exceptions en cas d'erreur dans notre code ou dans les données entrées par l'utilisateur.

```
def interet():
    try:
```

Les lignes suivantes serviront serviront à définir les 3 éléments "somme / nb_annees / interet" utiles au calcul des intérêts. 

Pour chacune de ces variables l'utilisateur définira lui même les valeurs via la fonction input. Des exceptions seront levées si les données rentrées sont incorrectes.

```
somme = float(input("Donnez-nous la somme totale que vous souhaitez placer\n"))

if somme < 0:
	raise ValueError("Vous ne pouvez pas investir un nombre négatif d'argent")
elif somme == 0:
	raise ValueError("Vous n'avez pas investi d'argent !")

nb_annees = int(input("combien d'années allez-vous placer cet argent ?\n"))

if nb_annees < 0:
	raise ValueError("Les années ne peuvent pas prendre une valeur négative")
elif nb_annees == 0:
	raise ValueError("Vous allez bien attendre au moins un an, non ?")

interet = input("A quel taux souhaitez vous voir les intérêts ?\n ATTENTION : Vous devrez mettre une valeur décimale \n Ex: 10% --> 0.10\n")

if "%" in interet:
	raise ValueError("Merci de mettre une valeur décimale uniquement, pas de signe de pourcentage \nEx 10% ---> 0.10")

else:
	interet = float(interet)
	
if interet < 0:
	raise ValueError("Vous ne pouvez pas avoir un taux d'intérêt négatif")
elif interet == 0:
	raise ValueError("Vous devriez avoir un taux d'intérêt")
```

On calculera ensuite les intérêts générés via cette ligne :

```
total = somme*(1+interet)**(nb_annees)
```

Enfin, on affiche le résultat de notre calcul via la fonction print et le string formatting.

```
print("La somme totale dont vous disposerez après avoir déposé {} au bout de {} ans sera de {:.2f}".format(somme,nb_annees, total))
```
