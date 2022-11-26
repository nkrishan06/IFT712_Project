# -*- coding: utf-8 -*-

#####
# Paul MEIGNAN (22 140 434)
# Krishan NAVAS (22 160 938)
# Hugo FREITAS COSTINHA (22 140 773)
####

from sklearn.model_selection import train_test_split
import pandas as pd


class GestionDonnees:
    def generer_donnes(self) :
        train = pd.read_csv("train.csv")
        X = train["species"]
        y = train.drop["species"]

        print("Generation des données de test...")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

        return X_train, y_train, X_test, y_test

