# -*- coding: utf-8 -*-

#####
# Paul MEIGNAN (22 140 434)
# Krishan NAVAS (22 160 938)
# Hugo FREITAS COSTINHA (22 140 773)
####

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold

class ArbreDecision:
    def __init__(self):
        """
        Algorithme d'arbre de décision
        
        """
        self.prof_max = 30 # Profondeur maximale du l'arbre
        self.min_leaf = 3  # Nombre minimal d'échantillons dans une feuille
        self.max_node = 110 # Nombre maximal de nodes de feuilles
    
    def recherche_hyper(self, x_tr, t_tr):
        """
        Recherche d'hyperparamètres pour l'Arbre de décisions
        
        x_train: Numpy array avec données d'entraînement
        t_train: Numpy array avec cibles pour l'entraînement

        Méthode de Grid Search: 
            prof_max: Profondeur maximale entre 10 et 50
            msf: Nombre minimal de samples (données) dans une feuille entre 2 et 10
            Mesure de la qualité de la séparation: giny, entropy et log_loss
        
        Retourne un dictionnaire avec les meilleurs hyperparamètres
        """
        valeurs_prof = np.arange(10,50)
        valeurs_mf = np.arange(2,10, dtype ='int')
        valeur_mn = np.arange(80,140, dtype = int)
        p_grid = {'criterion': ['gini','entropy','log_loss'], 'splitter' : ['best', 'random'], \
                   'max_depth': valeurs_prof,'min_samples_leaf': valeurs_mf, 'max_leaf_nodes': valeur_mn}
        
        cross_v = KFold(n_splits= 10, shuffle= True, random_state= None) # validation croisée
            
        # Recherche d'hyperparamètres
        self.classif = RandomizedSearchCV(estimator=DecisionTreeClassifier(),\
                                          param_distributions=p_grid, n_iter=20, cv=cross_v)
        self.classif.fit(x_tr, t_tr)
        
        mei_param = self.classif.best_params_
        
        return mei_param
    
    def entrainement(self, x_train, t_train, cherche_hyp):
        """
        Entraînement avec Arbre de décision
        
        x_train: Numpy array avec données d'entraînement
        t_train: Numpy array avec cibles pour l'entraînement
        cherche_hyp: Chercher ou non le meilleures hyperparamètres
        
        Retourne un objet avec le modèle entraîné
        """
        
        
        if cherche_hyp == True:
            print('Debut de l\'entrainement AD avec recherche d\'hyperparamètres','\n')
            parametres = self.recherche_hyper(x_train, t_train)
        else:
            print('Debut de l\'entrainement AD sans recherche d\'hyperparamètres','\n')
            parametres = {'criterion': 'gini', 'splitter' : 'best', 'max_depth': self.prof_max,\
                'min_samples_leaf': self.min_leaf, 'max_leaf_nodes': self.max_node}
            
        self.classif = DecisionTreeClassifier(**parametres)
        
        print('Paramètres utilisés pour l\'entraînement AD :',\
              self.classif.get_params(),'\n')
        #arbre_fin = self.classif.fit(x_train, t_train)
        #tree.plot_tree(arbre_fin)
        return self.classif.fit(x_train, t_train)
    
    def prediction(self, x_p):
        """
        Prédiction avec Arbre de décision
        
        x_p = Numpy array avec données pour trouver la prédiction
        
        Retourne les cibles t_p pour x_p et leur score
        """
        self.t_p = self.classif.predict(x_p)
        return self.t_p
    
    