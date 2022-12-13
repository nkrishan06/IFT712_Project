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
import SVM
import k_proches_voisins
import gaussian_bayesienne
import arbre_decision
import analyse_discriminant_lineaire
import regression_logistique

# Ignorer les warnings
from warnings import simplefilter



simplefilter(action='ignore')

# Lire la base de données
d_base = pd.read_csv(os.getcwd() + '\\dataset\\train.csv')

algorithme = 'Arbre_de_decisions'
cherche_hyp = True

# Importer l'algorithme correspondant

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
    # Caractéristique de la base de données
    d_base.head()
    d_base.describe()
    d_base.duplicated().sum()
    d_base.isnull().sum().sum()
    d_base['species'].nunique()
    d_base['species'].unique()

    # images
    id = 1
    plt.imshow(mpimg.imread(os.getcwd() + "\\dataset\\images\\" + str(id) + ".jpg"), cmap="gray")
    plt.title("Image " + str(id) + ": " + d_base[d_base["id"] == id].values[:, 1][0]);
    plt.show()

    # Correlation
    df = d_base.copy()
    correlation = df.drop("id", axis=1,
                          inplace=False).corr()  # On enlève l'id car pas pertinent et on réalise la correlation

    # Séparer les données et leur cibles
    g_donnees = gd.GestionDonnees(d_base)
    [types, X, t] = g_donnees.lecture_donnees(d_base)
    
    # Séparer les données pour test et train
    x_tr, x_ts, t_tr, t_ts = g_donnees.sep_donnees(X, t)
        
    # Entraînement
    debut_e = time.time() # Heure de debut pour mesurer le temps d'entraînement
    classif.entrainement(x_tr, t_tr, cherche_hyp)
    fin_e = time.time() # Heure de fin pour mesurer le temps d'entraînement
    print('Fin de l\'entrainement. Réalisé en %.2f secondes.'% (fin_e - debut_e),'\n')
    
    # Prédictions pour les ensembles d'entraînement et de test
    predict_tr = classif.prediction(x_tr)
    predict_ts = classif.prediction(x_ts)
    
    # Métriques pour évaluer l'entraînement et test
    prs_tr, rec_tr, fbeta_tr, _ = metriques(t_tr, predict_tr, average='macro')
    prs_ts, rec_ts, fbeta_ts, _ = metriques(t_ts, predict_ts, average='macro')
    acc_tr = accu(t_tr, predict_tr)
    acc_ts = accu(t_ts, predict_ts)
    tab_perform = [['Accuracy', acc_tr, acc_ts],['Précision', prs_tr, prs_ts],\
                   ['Rappel', rec_tr, rec_ts],['F-Beta', fbeta_tr, fbeta_ts]]
    print(tabulate(tab_perform, headers=['Metrique', 'Train', 'Test'], floatfmt='.4f'))
   
    return tab_perform

if __name__ == "__main__":
    main()