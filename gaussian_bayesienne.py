# -*- coding: utf-8 -*-

#####
# Paul MEIGNAN (22 140 434)
# Krishan NAVAS (22 160 938)
# Hugo FREITAS COSTINHA (22 140 773)
####

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, KFold

class GaussianBayes:
    def __init__(self):
        """
        Algorithme Naïve Bayes Bernoulli
        
        """
        self.lissage = 1e-9 # Lissage des données d'entrée

    
    def recherche_hyper(self, x_tr, t_tr):
        """
        Recherche d'hyperparamètres pour Naïve Bayes Bernoulli
        
        x_train: Numpy array avec données d'entraînement
        t_train: Numpy array avec cibles pour l'entraînement

        Méthode de Grid Search: 
            prof_max: Profondeur maximale entre 10 et 50
            msf: Nombre minimal d'échantillons dans une feuille entre 2 et 10
            Mesure de la qualité de la séparation: giny et entropy
        
        Retourne un dictionnaire avec les meilleurs hyperparamètres
        """
        valeurs_liss = np.arange(0.5e-9,1e-8,0.5e-9)
        p_grid = [{'alpha': valeurs_liss}]
        
        cross_v = KFold(10, True) # validation croisée
            
        # Recherche d'hyperparamètres
        self.classif = GridSearchCV(estimator=GaussianNB(),\
                                          param_grid=p_grid, cv=cross_v)
        self.classif.fit(x_tr, t_tr)
        
        mei_param = self.classif.best_params_
        
        return mei_param
    
    def entrainement(self, x_train, t_train, cherche_hyp):
        """
        Entraînement avec Naïve Bayes Bernoulli
        
        x_train: Numpy array avec données d'entraînement
        t_train: Numpy array avec cibles pour l'entraînement
        cherche_hyp: Chercher ou non les meilleures hyperparamètres
        
        Retourne un objet avec le modèle entraîné
        """
        
        
        if cherche_hyp == True:
            print('Debut de l\'entrainement GaussNB avec recherche d\'hyperparamètres','\n')
            parametres = self.recherche_hyper(x_train, t_train)
        else:
            print('Debut de l\'entrainement GaussNB sans recherche d\'hyperparamètres','\n')
            parametres = {'alpha': self.lissage}
            
        self.classif = GaussianNB(**parametres)
        
        print('Paramètres utilisés pour l\'entraînement GaussNB :',\
              self.classif.get_params(),'\n')        

        return self.classif.fit(x_train, t_train)
    
    def prediction(self, x_p):
        """
        Prédiction avec Naïve Bayes Bernoulli
        
        x_p = Numpy array avec données pour trouver la prédiction
        
        Retourne les cibles t_p pour x_p et leur score
        """
        self.t_p = self.classif.predict(x_p)
        return self.t_p
    
    