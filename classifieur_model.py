# -*- coding: utf-8 -*-

#####
# Paul MEIGNAN (22 140 434)
# Krishan NAVAS (22 160 938)
# Hugo FREITAS COSTINHA (22 140 773)
####

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import gestion_donnees
import matplotlib.pyplot as plt


class Classifieur :
    def __init__(self, lamb, methode):
        self.lamb = lamb
        self.methode = methode

    def entrainement(self, X_train, y_train):
        print("Création du modèle...")
        match self.method:
            case 1 : #Regression Logistique
                clf = LogisticRegression()
                clf.fit(X_train, y_train)
                return clf

            case 2 : #Analyse de discriminant
                clf = LinearDiscriminantAnalysis()

            case 3 : #K voisins
                clf = KNeighborsClassifier()

            case 4 : #GaussianNB
                clf = GaussianNB()

            case 5 : #Arbre de décision
                clf = DecisionTreeClassifier()

            case 6 : #SVC 
                clf = SVC()

            case other :
                print("no match found")

    def prediction(self, X_test):
        match self.method:
            case 1 : #Regression Logistique
                clf.predict(X_test)

            case 2 : #Analyse de discriminant
                clf = LinearDiscriminantAnalysis()

            case 3 : #K voisins
                clf = KNeighborsClassifier()

            case 4 : #GaussianNB
                clf = GaussianNB()

            case 5 : #Arbre de décision
                clf = DecisionTreeClassifier()

            case 6 : #SVC 
                clf = SVC()

            case other :
                print("no match found")

    @staticmethod
    def erreur(t, prediction):
        """
        Retourne l'erreur de classification, i.e.
        1. si la cible ``t`` et la prédiction ``prediction``
        sont différentes, 0. sinon.
        """
        # AJOUTER CODE ICI
        if t != prediction:
            return 1
        return 0

    def afficher_donnees_et_modele(self, x_train, t_train, x_test, t_test):
        """
        afficher les donnees et le modele

        x_train, t_train : donnees d'entrainement
        x_test, t_test : donnees de test
        """
        plt.figure(0)
        plt.scatter(x_train[:, 0], x_train[:, 1], s=t_train * 100 + 20, c=t_train)

        pente = self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = -pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Training data')

        plt.figure(1)
        plt.scatter(x_test[:, 0], x_test[:, 1], s=t_test * 100 + 20, c=t_test)

        pente = self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = -pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Testing data')

        plt.show()

    def parametres(self):
        """
        Retourne les paramètres du modèle
        """
        return self.w_0, self.w
