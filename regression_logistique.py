# -*- coding: utf-8 -*-

#####
# Paul MEIGNAN (22 140 434)
# Krishan NAVAS (22 160 938)
# Hugo FREITAS COSTINHA (22 140 773)
####


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, KFold



class Regression_logistique:
    def __init__(self):
        """
        Algorithme de Forêt Aléatoire
        
        """
        self.penalty = 'l2' # Nombre d arbres à utiliser
        self.tol = 1e-4 # Profondeur maximale du l'arbre
        self.solver = 'lbfgs' # Nombre minimal de samples dans une feuille
    
    def recherche_hyper(self, x_tr, t_tr):
        """
        Recherche d'hyperparamètres pour la Forêt Aléatoire
        
        x_train: Numpy array avec données d'entraînement
        t_train: Numpy array avec cibles pour l'entraînement

        Méthode de Randomized Search: 
            n_arbres: Nombre d'arbres entre 50 et 200
            prof_max: Profondeur maximale entre 10 et 30
            msf: Nombre minimal d'échantillons dans une feuille entre 2 et 10
            Mesure de la qualité de la séparation: giny et entropy
        
        Retourne un dictionnaire avec les meilleurs hyperparamètres
        """
        valeurs_tol = np.arange(0.5e-4, 2e-3, 0.5e-4)
        p_grid = {'penalty': ['l1', 'l2', 'none', 'elasticnet'], 'tol': valeurs_tol, \
                   'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
        
        cross_v = KFold(10, True) # validation croisée
            
        # Recherche d'hyperparamètres
        self.classif = RandomizedSearchCV(estimator=LogisticRegression(), \
                                          param_distributions=p_grid, n_iter=25, cv=cross_v)
        self.classif.fit(x_tr, t_tr)
        mei_param = self.classif.best_params_
        
        return mei_param
    
    def entrainement(self, x_train, t_train, cherche_hyp):
        """
        Entraînement avec Forêt Aléatoire
        
        x_train: Numpy array avec données d'entraînement
        t_train: Numpy array avec cibles pour l'entraînement
        cherche_hyp: Chercher ou non les meilleures hyperparamètres
        
        Retourne un objet avec le modèle entraîné
        """

        if cherche_hyp == True:
            print('Debut de l\'entrainement RegLogistique avec recherche d\'hyperparamètres','\n')
            parametres = self.recherche_hyper(x_train, t_train)
        else:
            print('Debut de l\'entrainement RegLogistique sans recherche d\'hyperparamètres','\n')
            parametres = {'penalty': self.penalty, 'tol': self.tol, 'solver': self.solver}
            
        self.classif = LogisticRegression(**parametres)
        
        print('Paramètres utilisés pour l\'entraînement RegLogistique :',\
              self.classif.get_params(),'\n')

        return self.classif.fit(x_train, t_train)
    
    def prediction(self, x_p):
        """
        Prédiction avec Forêt Aléatoire
        
        x_p = Numpy array avec données pour trouver la prédiction
        
        Retourne les cibles t_p pour x_p et leur score
        """
        self.t_p = self.classif.predict(x_p)
        return self.t_p
    
    