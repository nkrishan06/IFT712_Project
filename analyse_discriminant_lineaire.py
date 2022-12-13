# -*- coding: utf-8 -*-

#####
# Paul MEIGNAN (22 140 434)
# Krishan NAVAS (22 160 938)
# Hugo FREITAS COSTINHA (22 140 773)
####


import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RandomizedSearchCV, KFold

class Analyse_Discriminant_lineaire:
    def __init__(self):
        """
        Analyse_Discriminant_lineaire
        
        """
        self.solver = 'svd' # type du résolveur
        self.tol = 1e-4  # seuil pour déterminer si l'observation est pertinente
    
    def recherche_hyper(self, x_tr, t_tr):
        """
        Recherche d'hyperparamètres pour l'Analyse de Discriminant lineaire
        
        x_tr: Numpy array avec données d'entraînement
        t_tr: Numpy array avec cibles pour l'entraînement

        Méthode de Grid Search: 
            tol: seuil utilisé entre 0,00005 et 0.0001
            solver : résolveur utilisé entre 'svd', 'lsqr' et 'eigen'
        
        Retourne un dictionnaire avec les meilleurs hyperparamètres
        """
        valeurs_tol = np.arange(0.5e-4,1e-3,0.5e-4)
        p_grid = {'solver': ['svd','lsqr','eigen'], 'tol': valeurs_tol}
        
        cross_v = KFold(10, True) # validation croisée
            
        # Recherche d'hyperparamètres
        self.classif = RandomizedSearchCV(estimator=LinearDiscriminantAnalysis(),\
                                          param_distributions=p_grid, n_iter=20, cv=cross_v)
        self.classif.fit(x_tr, t_tr)
        
        mei_param = self.classif.best_params_
        
        return mei_param
    
    def entrainement(self, x_train, t_train, cherche_hyp):
        """
        Entraînement avec l'Analyse de Discriminant lineaire
        
        x_train: Numpy array avec données d'entraînement
        t_train: Numpy array avec cibles pour l'entraînement
        cherche_hyp: Chercher ou non le meilleures hyperparamètres
        
        Retourne un objet avec le modèle entraîné
        """
        
        if cherche_hyp == True:
            print('Debut de l\'entrainement ADL avec recherche d\'hyperparamètres','\n')
            parametres = self.recherche_hyper(x_train, t_train)
        else:
            print('Debut de l\'entrainement ADL sans recherche d\'hyperparamètres','\n')
            parametres = {'solver': self.solver, 'tol': self.tol}
            
        self.classif = LinearDiscriminantAnalysis(**parametres)
        
        print('Paramètres utilisés pour l\'entraînement ADL :',\
              self.classif.get_params(),'\n')
        #arbre_fin = self.classif.fit(x_train, t_train)
        #tree.plot_tree(arbre_fin)
        return self.classif.fit(x_train, t_train)
    
    def prediction(self, x_p):
        """
        Prédiction avec l'Analyse de Discriminant lineaire
        
        x_p = Numpy array avec données pour trouver la prédiction
        
        Retourne les cibles t_p pour x_p et leur score
        """
        self.t_p = self.classif.predict(x_p)
        return self.t_p
    
    