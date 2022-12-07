# -*- coding: utf-8 -*-

#####
# Paul MEIGNAN (22 140 434)
# Krishan NAVAS (22 160 938)
# Hugo FREITAS COSTINHA (22 140 773)
####


import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, KFold



class SupportVectorMachine:
    def __init__(self):
        """
        Algorithme de machines à vecteurs de support
        
        """
        self.lamb = 1
        self.degree = 3
        self.kernel = 'rbf'
        self.gamma = 'scale'
    
    def recherche_hyper(self, x_tr, t_tr):
        """
        Recherche d'hyperparamètres pour SVM, ainsi que le meilleur noyau
        
        x_train: Numpy array avec données d'entraînement
        t_train: Numpy array avec cibles pour l'entraînement

        Méthode de Randomized Search. Noyaus evalués: rbf, polynomial et sigmoïde 
        
        Retourne une dictionaire avec le meilleur noyau et ses meilleurs hyperparamètres
        """
        valeurs_lamb = np.linspace(0.00001,2,30)
        p_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': valeurs_lamb, 'gamma': ['scale','auto'], \
                        'degree': np.arange(2,7), 'coef0': np.arange(0.,10.,0.1)}
        
        cross_v = KFold(10, True) # Validation croisée
            
        # Recherche d'hyperparamètres
        self.classif = RandomizedSearchCV(estimator=SVC(), param_distributions=p_grid, n_iter=25, cv=cross_v)
        self.classif.fit(x_tr, t_tr)
        mei_param = self.classif.best_params_
        
        return mei_param
    
    def entrainement(self, x_train, t_train, cherche_hyp):
        """
        Entraînement avec SVM
        
        x_train: Numpy array avec données d'entraînement
        t_train: Numpy array avec cibles pour l'entraînement
        cherche_hyp: Chercher ou non le meilleur type de noyau et ses hyperparamètres
        
        Retourne un objet avec le modèle entraîné
        """

        if cherche_hyp == True:
            print('Debut de l\'entrainement SVM avec recherche d\'hyperparamètres','\n')
            parametres = self.recherche_hyper(x_train, t_train)
        else:
            print('Debut de l\'entrainement SVM sans recherche d\'hyperparamètres','\n')
            parametres = {'kernel': self.kernel, 'C': self.lamb, 'gamma': self.gamma, 'degree' : self.degree}
            
        self.classif = SVC(**parametres)
        
        print('Paramètres utilisés pour l\'entraînement SVM :',\
              self.classif.get_params(),'\n')
        
        return self.classif.fit(x_train, t_train)
    
    def prediction(self, x_p):
        """
        Prédiction avec SVM
        
        x_p = Numpy array avec données pour trouver la prédiction
        
        Retourne les cibles t_p pour x_p et leur score
        """
        self.t_p = self.classif.predict(x_p)
        return self.t_p
    

    
        