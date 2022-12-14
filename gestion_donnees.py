# -*- coding: utf-8 -*-

#####
# Paul MEIGNAN (22 140 434)
# Krishan NAVAS (22 160 938)
# Hugo FREITAS COSTINHA (22 140 773)
####


from sklearn.preprocessing import LabelEncoder # Gérer les noms des cibles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class GestionDonnees :
    def __init__(self, bd) :
        #self.donnees_base = donnees_base
        self.bd = bd
        
    def lecture_donnees(self, base_d) :
        """
        base_d : Matrice lu du fichier base avec l'ensemble de données d'entraînement.

        f_types : Liste avec les noms des types de feuilles.
        x_base : Contient uniquement les données d'entraînement.
        t_base : Contient uniquement les cibles (chiffres) pour l'entraînement.
        """
        # Lire les types des feuilles du fichier train
        encoder = LabelEncoder().fit(base_d.species)
        f_types = list(encoder.classes_)
        t_base = encoder.transform(base_d.species)
        
        # Séparer les identifiants de l'ensemble de données
        x_base_df = base_d.drop(['id', 'species'], axis= 1 )
        x_base = x_base_df.to_numpy()
        
        return f_types, x_base, t_base
    
    def sep_donnees(self, x_data,t_data) :
        
        # Separer les données d'entraînement et données de test
        x_tr, x_te, t_tr, t_te = train_test_split(x_data, t_data, test_size = 0.25, \
                                                  random_state = 7)
                
        return x_tr, x_te, t_tr, t_te
            
    def apply_PCA(self, X_train, X_test):
        # Standardise les données sans la cible
        scaler = StandardScaler()
        scaler.fit(X_train)

        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Applique une ACP
        pca = PCA(n_components= 'mle')
        pca.fit(X_train_scaled)
        X_train_pca = pca.transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        return X_train_pca,X_test_pca


