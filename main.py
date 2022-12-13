# -*- coding: utf-8 -*-
#####
# Paul MEIGNAN (22 140 434)
# Krishan NAVAS (22 160 938)
# Hugo FREITAS COSTINHA (22 140 773)
####

# Importer outils générals
import pandas as pd
import os
import time
from sklearn.metrics import precision_recall_fscore_support as metriques
from sklearn.metrics import accuracy_score as accu
from tabulate import tabulate

# Importer fichiers spécifiques
import gestion_donnees as gd
from classifier import gaussian_bayesienne, arbre_decision, k_proches_voisins, SVM, regression_logistique
import analyse_discriminant_lineaire


# Ignorer les warnings
from warnings import simplefilter
simplefilter(action='ignore')

# Ouvrir le csv
d_base = pd.read_csv(os.getcwd() + '\\dataset\\train.csv')

algorithme = 'Arbre_de_decisions'
cherche_hyp = True

#Utilisation de l'algo
if algorithme == 'Regression_logistique':
    classif = regression_logistique.Regression_logistique()
    
elif algorithme == 'SVM':
    classif = SVM.SupportVectorMachine()
    
elif algorithme == 'K-proches_voisins': 
    classif = k_proches_voisins.KProchesVoisins()
    
elif algorithme == 'Gaussian_Bayesienne': 
    classif = gaussian_bayesienne.GaussianBayes()
    
elif algorithme == 'Arbre_de_decisions': 
    classif = arbre_decision.ArbreDecision()
    
elif algorithme == 'analyse_discriminant_lineaire': 
    classif = analyse_discriminant_lineaire.Analyse_Discriminant_lineaire()

def main():
    
    # Séparerations données/cibles
    g_donnees = gd.GestionDonnees(d_base)
    [types, X, t] = g_donnees.lecture_donnees(d_base)
    
    # Sépareration données test/ données train
    x_tr, x_ts, t_tr, t_ts = g_donnees.sep_donnees(X, t)
        
    # Entraînement
    debut_e = time.time() # mesure le temps d'entraînement
    classif.entrainement(x_tr, t_tr, cherche_hyp)
    fin_e = time.time() 
    print('Fin de l\'entrainement. Réalisé en %.2f secondes.'% (fin_e - debut_e),'\n')
    
    # Prédictions pour les ensembles d'entraînement/ ensemble de test
    predict_tr = classif.prediction(x_tr)
    predict_ts = classif.prediction(x_ts)
    
    # Métriques pour évaluer l'entraînement et les test
    prs_tr, rec_tr, fbeta_tr, _ = metriques(t_tr, predict_tr, average='macro')
    prs_ts, rec_ts, fbeta_ts, _ = metriques(t_ts, predict_ts, average='macro')
    acc_tr = accu(t_tr, predict_tr)
    acc_ts = accu(t_ts, predict_ts)
    tab_perform = [['Accuracy', acc_tr, acc_ts],['Précision', prs_tr, prs_ts],\
                   ['Rappel', rec_tr, rec_ts],['F-Beta', fbeta_tr, fbeta_ts]]
    print(tabulate(tab_perform, headers=['Metrique', 'Train', 'Test'], floatfmt='.4f'))
    
    # phase de preprocessing
    X_train_pca, X_test_pca = g_donnees.apply_PCA(x_tr, x_ts)
    
    # Entraînement avec PCA
    debut_e = time.time() # Heure de debut pour mesurer le temps d'entraînement
    classif.entrainement(X_train_pca, t_tr, cherche_hyp)
    fin_e = time.time() # Heure de fin pour mesurer le temps d'entraînement
    print('Fin de l\'entrainement. Réalisé en %.2f secondes.'% (fin_e - debut_e),'\n')
    
    # Prédictions pour les ensembles d'entraînement et de test
    predict_tr_pca = classif.prediction(X_train_pca)
    predict_ts_pca = classif.prediction(X_test_pca)
    
    # Métriques pour évaluer l'entraînement et test
    prs_tr_pca, rec_tr_pca, fbeta_tr_pca, _ = metriques(t_tr, predict_tr, average='macro')
    prs_ts_pca, rec_ts_pca, fbeta_ts_pca, _ = metriques(t_ts, predict_ts, average='macro')
    acc_tr_pca = accu(t_tr, predict_tr_pca)
    acc_ts_pca = accu(t_ts, predict_ts_pca)
    tab_perform_pca = [['Accuracy', acc_tr_pca, acc_ts_pca],['Précision', prs_tr_pca, prs_ts_pca],\
                   ['Rappel', rec_tr_pca, rec_ts_pca],['F-Beta', fbeta_tr_pca, fbeta_ts_pca]]
    print('Entrainement avec PCA')
    print(tabulate(tab_perform_pca, headers=['Metrique', 'Train', 'Test'], floatfmt='.4f'))
   
    return tab_perform, tab_perform_pca
   
    return tab_perform

if __name__ == "__main__":
    main()
