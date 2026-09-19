# Fraud Detection Project

Projet de détection de fraudes transactionnelles en Python.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

## Pipeline

1. Chargement des données
2. Nettoyage des doublons
3. Gestion des valeurs manquantes
4. Séparation X / y
5. Train / Test split
6. StandardScaler pour les variables numériques
7. OneHotEncoder pour les variables catégorielles
8. Logistic Regression
9. Évaluation du modèle
10. Sauvegarde et rechargement du modèle

## Métriques utilisées

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Matrice de confusion

## Exécution

```bash
python main.py