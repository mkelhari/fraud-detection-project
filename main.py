from src.preprocess import (
    load_data,
    inspect_data,
    remove_duplicates,
    handle_missing_values,
    split_features_target,
    split_train_val_test,
    create_preprocessor
)
from src.evaluate import (
    evaluate_model,
    get_confusion_matrix,
    get_roc_auc
)

df = load_data("data/raw/transactions.csv")

print(df.head())
print(df.columns)
print(df.shape)

inspect_data(df)

df = remove_duplicates(df)

print("\nAprès suppression des doublons :")
print(df.shape)

df = handle_missing_values(df)

print("\nValeurs manquantes après traitement :")
print(df.isnull().sum())

x, y = split_features_target(df, "Class")

print("\nShape de x :", x.shape)
print("Shape de y :", y.shape)

x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(x, y)

print("x_train :", x_train.shape)
print("x_test :", x_test.shape)
print("y_train :", y_train.shape)
print("y_test :", y_test.shape)

print("\nRépartition des classes :")
print(y.value_counts())

print("\nRépartition en pourcentage :")
print(y.value_counts(normalize=True) * 100)

preprocessor = create_preprocessor(x_train)

x_train_prepared = preprocessor.fit_transform(x_train)
X_val_prepared = preprocessor.transform(x_val)
x_test_prepared = preprocessor.transform(x_test)

from src.train import train_logistic_regression
model = train_logistic_regression(
    x_train_prepared,
    y_train
)

print("Modèle entraîné avec succès.")

y_val_proba = model.predict_proba(X_val_prepared)[:, 1]
for threshold in [0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98]:
    y_val_pred = (y_val_proba >= threshold).astype(int)

    accuracy, precision, recall, f1 = evaluate_model(
        y_val,
        y_val_pred
    )

    print(f"\n--- Seuil {threshold} ---")
    print("Precision :", precision)
    print("Recall :", recall)
    print("F1-score :", f1)

from src.predict import make_predictions
y_pred, y_proba = make_predictions(
    model,
    x_test_prepared
)
final_threshold = 0.98

y_test_pred = (y_proba >= final_threshold).astype(int)

accuracy, precision, recall, f1 = evaluate_model(
    y_test,
    y_test_pred
)

print("\n--- Évaluation finale ---")
print("Seuil :", final_threshold)
print("Accuracy :", accuracy)
print("Precision :", precision)
print("Recall :", recall)
print("F1-score :", f1)

print("\nMatrice de confusion :")
print(get_confusion_matrix(y_test, y_test_pred))

y_pred_threshold = (y_proba >= threshold).astype(int)

from src.evaluate import evaluate_model, get_confusion_matrix

accuracy, precision, recall, f1 = evaluate_model(
    y_test,
    y_pred_threshold
)

print("\n--- Seuil 0.8 ---")
print("Accuracy :", accuracy)
print("Precision :", precision)
print("Recall :", recall)
print("F1-score :", f1)

print("Matrice de confusion :")
print(get_confusion_matrix(y_test, y_pred_threshold))
print("Prédictions :", y_pred[:10])
print("Probabilités de fraude :", y_proba[:10])
model.predict(x_test_prepared)

from src.evaluate import evaluate_model
accuracy, precision, recall, f1 = evaluate_model(
    y_test,
    y_pred
)

print("Accuracy :", accuracy)
print("Precision :", precision)
print("Recall :", recall)
print("F1-score :", f1)

from src.evaluate import evaluate_model, get_confusion_matrix
cm = get_confusion_matrix(y_test, y_pred)

print("\nMatrice de confusion :")
print(cm)

from src.evaluate import evaluate_model, get_confusion_matrix, get_roc_auc

roc_auc = get_roc_auc(y_test, y_proba)

print("ROC-AUC :", roc_auc)

from src.train import train_logistic_regression, save_model

save_model(
    model,
    "models/logistic_regression.pkl"
)

print("Modèle sauvegardé.")

from src.preprocess import save_preprocessor

save_preprocessor(
    preprocessor,
    "models/preprocessor.pkl"
)

print("Préprocesseur sauvegardé.")

from src.predict import load_artifacts, predict_new_transaction

loaded_model, loaded_preprocessor = load_artifacts(
    "models/logistic_regression.pkl",
    "models/preprocessor.pkl"
)

new_transaction = x_test.iloc[[0]]

prediction, probability = predict_new_transaction(
    loaded_model,
    loaded_preprocessor,
    new_transaction
)

print("Prédiction :", prediction[0])
print("Probabilité de fraude :", probability[0])

print("\n--- Vérification finale ---")

print("Valeur réelle :", y_test.iloc[0])
print("Prédiction :", prediction[0])
print("Probabilité de fraude :", probability[0])
