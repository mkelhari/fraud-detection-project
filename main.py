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
from sklearn.model_selection import train_test_split
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
from src.train import (
    train_logistic_regression,
    train_random_forest,
    save_model
)
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

random_forest = train_random_forest(
    x_train_prepared,
    y_train
)

print("Random Forest entraîné avec succès.")

rf_val_proba = random_forest.predict_proba(X_val_prepared)[:, 1]

import numpy as np

best_threshold = 0
best_f1 = 0

for threshold in np.arange(0.01, 1.00, 0.01):
    rf_val_pred = (rf_val_proba >= threshold).astype(int)

    accuracy, precision, recall, f1 = evaluate_model(
        y_val,
        rf_val_pred
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("\n--- Meilleur seuil Random Forest ---")
print("Seuil :", best_threshold)
print("F1-score :", best_f1)

rf_test_proba = random_forest.predict_proba(x_test_prepared)[:, 1]

rf_test_pred = (rf_test_proba >= best_threshold).astype(int)

accuracy, precision, recall, f1 = evaluate_model(
    y_test,
    rf_test_pred
)

print("\n--- Évaluation finale Random Forest ---")
print("Seuil :", round(best_threshold, 2))
print("Accuracy :", accuracy)
print("Precision :", precision)
print("Recall :", recall)
print("F1-score :", f1)

print("\nMatrice de confusion :")
print(get_confusion_matrix(y_test, rf_test_pred))

save_model(
    random_forest,
    "models/random_forest.pkl"
)

print("Random Forest sauvegardé.")

from src.anomaly import train_isolation_forest, get_anomaly_scores
from src.reinforce_state import build_states
print("\n--- Isolation Forest ---")

isolation_forest = train_isolation_forest(
    x_train_prepared
)

print("Isolation Forest entraîné avec succès.")

test_anomaly_scores = get_anomaly_scores(
    isolation_forest,
    x_test_prepared
)
# Probabilités Random Forest sur validation
rf_val_proba = random_forest.predict_proba(
    X_val_prepared
)[:, 1]

# Scores Isolation Forest sur validation
val_anomaly_scores = get_anomaly_scores(
    isolation_forest,
    X_val_prepared
)

# Construction des états de validation
val_states = build_states(
    rf_val_proba,
    val_anomaly_scores,
    X_val_prepared[:, -1]
)

print("\n--- États pour apprentissage RL ---")
print("Shape :", val_states.shape)
print("Premier état :", val_states[0])
print("Classe réelle :", y_val.iloc[0])
print("Premiers scores d'anomalie :")
print(test_anomaly_scores[:10])


from src.train import train_xgboost

xgboost_model = train_xgboost(
    x_train_prepared,
    y_train
)

print("XGBoost entraîné avec succès.")


xgb_val_proba = xgboost_model.predict_proba(
    X_val_prepared
)[:, 1]

best_xgb_threshold = 0
best_xgb_f1 = 0

for threshold in np.arange(0.01, 1.00, 0.01):
    xgb_val_pred = (xgb_val_proba >= threshold).astype(int)

    accuracy, precision, recall, f1 = evaluate_model(
        y_val,
        xgb_val_pred
    )

    if f1 > best_xgb_f1:
        best_xgb_f1 = f1
        best_xgb_threshold = threshold

print("\n--- Meilleur seuil XGBoost ---")
print("Seuil :", round(best_xgb_threshold, 2))
print("F1-score :", best_xgb_f1)

xgb_test_proba = xgboost_model.predict_proba(
    x_test_prepared
)[:, 1]

xgb_test_pred = (
    xgb_test_proba >= best_xgb_threshold
).astype(int)

accuracy, precision, recall, f1 = evaluate_model(
    y_test,
    xgb_test_pred
)

print("\n--- Évaluation finale XGBoost ---")
print("Seuil :", round(best_xgb_threshold, 2))
print("Accuracy :", accuracy)
print("Precision :", precision)
print("Recall :", recall)
print("F1-score :", f1)

print("\nMatrice de confusion :")
print(
    get_confusion_matrix(
        y_test,
        xgb_test_pred
    )
)

xgb_test_proba = xgboost_model.predict_proba(
    x_test_prepared
)[:, 1]

xgb_test_pred = (
    xgb_test_proba >= best_xgb_threshold
).astype(int)

accuracy, precision, recall, f1 = evaluate_model(
    y_test,
    xgb_test_pred
)

print("\n--- Évaluation finale XGBoost ---")
print("Seuil :", round(best_xgb_threshold, 2))
print("Accuracy :", accuracy)
print("Precision :", precision)
print("Recall :", recall)
print("F1-score :", f1)

print("\nMatrice de confusion :")
print(
    get_confusion_matrix(
        y_test,
        xgb_test_pred
    )
)

save_model(
    xgboost_model,
    "models/xgboost.pkl"
)

print("XGBoost sauvegardé.")

from src.anomaly import train_isolation_forest

isolation_forest = train_isolation_forest(
    x_train_prepared
)

print("Isolation Forest entraîné avec succès.")

isolation_forest = train_isolation_forest(
    x_train_prepared
)

print("Isolation Forest entraîné avec succès.")

from src.anomaly import train_isolation_forest, get_anomaly_scores

anomaly_scores_val = get_anomaly_scores(
    isolation_forest,
    X_val_prepared
)

y_val_array = y_val.to_numpy()

normal_scores = anomaly_scores_val[y_val_array == 0]
fraud_scores = anomaly_scores_val[y_val_array == 1]

print("\n--- Isolation Forest ---")
print("Score moyen transactions normales :", normal_scores.mean())
print("Score moyen fraudes :", fraud_scores.mean())

print("Score médian transactions normales :", np.median(normal_scores))
print("Score médian fraudes :", np.median(fraud_scores))

y_val_array = y_val.to_numpy()

normal_scores = anomaly_scores_val[y_val_array == 0]
fraud_scores = anomaly_scores_val[y_val_array == 1]

print("\n--- Isolation Forest ---")
print("Score moyen transactions normales :", normal_scores.mean())
print("Score moyen fraudes :", fraud_scores.mean())

print(
    "Score médian transactions normales :",
    np.median(normal_scores)
)

print(
    "Score médian fraudes :",
    np.median(fraud_scores)
)

from src.reinforce_state import build_states


amount_mean = x_train["Amount"].mean()
amount_std = x_train["Amount"].std()

amount_val = (
    x_val["Amount"].to_numpy() - amount_mean
) / amount_std

states_val = build_states(
    rf_val_proba,
    anomaly_scores_val,
    amount_val
)

print("Shape des états :", states_val.shape)
print("Premier état :", states_val[0])

from src.reward import calculate_reward

print("\n--- Test récompenses ---")
print("Normale acceptée :", calculate_reward(0, 0))
print("Fraude bloquée :", calculate_reward(1, 1))
print("Fraude acceptée :", calculate_reward(0, 1))
print("Normale bloquée :", calculate_reward(1, 0))

from src.reinforce_state import build_states

states = build_states(
    rf_test_proba,
    test_anomaly_scores,
    x_test_prepared[:, -1]
)

print("\nShape des états :", states.shape)
print("Premier état :", states[0])

from src.environment import FraudEnvironment

print("\n--- Test environnement ---")

# Une partie pour apprendre, une autre pour régler le seuil
rl_train_states, tune_states, y_rl_train, y_tune = train_test_split(
    val_states,
    y_val.values,
    test_size=0.5,
    random_state=42,
    stratify=y_val.values
)

print("\n--- Séparation des données RL ---")
print("États pour apprentissage :", rl_train_states.shape)
print("États pour réglage :", tune_states.shape)
print("Fraudes pour apprentissage :", int(y_rl_train.sum()))
print("Fraudes pour réglage :", int(y_tune.sum()))

train_env = FraudEnvironment(
    rl_train_states,
    y_rl_train
)

state = train_env.reset()

print("Premier état :", state)

next_state, reward, done = train_env.step(0)

print("État suivant :", next_state)
print("Récompense :", reward)
print("Terminé :", done)

from src.q_agent import QLearningAgent

print("\n--- Test agent Q-Learning ---")

agent = QLearningAgent()

state = train_env.reset()

print("État :", state)

print(
    "État discrétisé :",
    agent.discretize_state(state)
)

action = agent.choose_action(state)

print("Action choisie :", action)

print(
    "Action :",
    "ACCEPTER" if action == 0 else "BLOQUER"
)

print("\n--- Entraînement Q-Learning ---")

# Environnement d'apprentissage :
# IMPORTANT : on utilise validation, pas test
train_env = FraudEnvironment(
    val_states,
    y_val.values
)

# Nouvel agent qui va réellement apprendre
agent = QLearningAgent(
    learning_rate=0.1,
    discount_factor=0.0,
    epsilon=1.0,
    epsilon_decay=0.90,
    epsilon_min=0.05
)

n_episodes = 30

for episode in range(n_episodes):

    state = train_env.reset()

    done = False
    total_reward = 0

    while not done:

        # L'agent choisit accepter ou bloquer
        action = agent.choose_action(state)

        # L'environnement répond
        next_state, reward, done = train_env.step(action)

        # L'agent apprend
        agent.update(
            state,
            action,
            reward,
            next_state,
            done
        )

        total_reward += reward

        state = next_state

    # On diminue progressivement l'exploration
    agent.decay_epsilon()

    print(
        f"Episode {episode + 1}/{n_episodes} "
        f"- Reward : {total_reward} "
        f"- Epsilon : {agent.epsilon:.4f}"
    )

print("\nEntraînement terminé.")
print("Nombre d'états dans la Q-table :", len(agent.q_table))
unknown_tune = sum(
    agent.discretize_state(state) not in agent.q_table
    for state in tune_states
)

print("États de réglage inconnus :", unknown_tune)
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Décisions de l'agent, sans exploration ni apprentissage
hybrid_tune_predictions = [
    agent.predict_action(state)
    for state in tune_states
]

# Même ensemble de transactions, avec le Random Forest seul
rf_tune_predictions = (
    tune_states[:, 0] >= 0.21
).astype(int)

for name, predictions in [
    ("Random Forest — RÉGLAGE", rf_tune_predictions),
    ("Agent hybride — RÉGLAGE", hybrid_tune_predictions)
]:
    print(f"\n--- {name} ---")

    print("Precision :", precision_score(
        y_tune, predictions, zero_division=0
    ))
    print("Recall :", recall_score(
        y_tune, predictions, zero_division=0
    ))
    print("F1-score :", f1_score(
        y_tune, predictions, zero_division=0
    ))

    print("Matrice de confusion :")
    print(confusion_matrix(
        y_tune, predictions, labels=[0, 1]
    ))
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
test_states = build_states(
    rf_test_proba,
    test_anomaly_scores,
    x_test_prepared[:, -1]
)
print("\n--- Analyse couverture Q-table AVANT évaluation ---")

known_states = 0
unknown_states = 0

known_frauds = 0
unknown_frauds = 0

for state, real_class in zip(test_states, y_test.values):

    discrete_state = agent.discretize_state(state)

    if discrete_state in agent.q_table:

        known_states += 1

        if real_class == 1:
            known_frauds += 1

    else:

        unknown_states += 1

        if real_class == 1:
            unknown_frauds += 1


print("États TEST connus :", known_states)
print("États TEST inconnus :", unknown_states)

print(
    "Pourcentage connu :",
    known_states / len(test_states) * 100
)

print("\nFraudes dans états connus :", known_frauds)
print("Fraudes dans états inconnus :", unknown_frauds)

print("\n--- Vraie couverture Q-table avant TEST ---")

known = 0
unknown = 0
known_frauds = 0
unknown_frauds = 0

for state, real_class in zip(test_states, y_test.values):

    discrete_state = agent.discretize_state(state)

    if discrete_state in agent.q_table:

        known += 1

        if real_class == 1:
            known_frauds += 1

    else:

        unknown += 1

        if real_class == 1:
            unknown_frauds += 1


print("États connus :", known)
print("États inconnus :", unknown)

print(
    "Couverture :",
    known / len(test_states) * 100,
    "%"
)

print("Fraudes dans états connus :", known_frauds)
print("Fraudes dans états inconnus :", unknown_frauds)

print("\n--- Évaluation Q-Learning sur TEST ---")

# On recrée clairement les états TEST
test_states = build_states(
    rf_test_proba,
    test_anomaly_scores,
    x_test_prepared[:, -1]
)

# IMPORTANT :
# plus d'exploration pendant l'évaluation
agent.epsilon = 0.0

rl_predictions = []

for state in test_states:

    action = agent.predict_action(state)

    # 0 = transaction normale
    # 1 = fraude
    rl_predictions.append(action)

# Calcul des métriques
accuracy = accuracy_score(
    y_test,
    rl_predictions
)

precision = precision_score(
    y_test,
    rl_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    rl_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    rl_predictions,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    rl_predictions
)

print("Accuracy :", accuracy)
print("Precision :", precision)
print("Recall :", recall)
print("F1-score :", f1)

print("\nMatrice de confusion :")
print(cm)

print("\n--- Analyse couverture Q-table ---")

known_states = 0
unknown_states = 0

unknown_frauds = 0
known_frauds = 0

for state, real_class in zip(test_states, y_test.values):

    discrete_state = agent.discretize_state(state)

    if discrete_state in agent.q_table:

        known_states += 1

        if real_class == 1:
            known_frauds += 1

    else:

        unknown_states += 1

        if real_class == 1:
            unknown_frauds += 1


print("États TEST connus :", known_states)
print("États TEST inconnus :", unknown_states)

print(
    "Pourcentage connu :",
    known_states / len(test_states) * 100
)

print("\nFraudes dans états connus :", known_frauds)
print("Fraudes dans états inconnus :", unknown_frauds)

print("\n--- Analyse des états contenant des fraudes ---")

fraud_state_stats = {}

for state, real_class in zip(val_states, y_val.values):

    discrete_state = agent.discretize_state(state)

    if discrete_state not in fraud_state_stats:
        fraud_state_stats[discrete_state] = {
            "total": 0,
            "frauds": 0
        }

    fraud_state_stats[discrete_state]["total"] += 1

    if real_class == 1:
        fraud_state_stats[discrete_state]["frauds"] += 1


fraud_states = []

for discrete_state, stats in fraud_state_stats.items():

    if stats["frauds"] > 0:

        fraud_rate = (
            stats["frauds"]
            / stats["total"]
        )

        q_values = agent.q_table.get(
            discrete_state,
            [0, 0]
        )

        fraud_states.append(
            (
                discrete_state,
                stats["total"],
                stats["frauds"],
                fraud_rate,
                q_values
            )
        )


fraud_states.sort(
    key=lambda x: x[3],
    reverse=True
)


for item in fraud_states[:20]:

    state_id, total, frauds, rate, q_values = item

    print(
        f"État {state_id} "
        f"- Total : {total} "
        f"- Fraudes : {frauds} "
        f"- Taux fraude : {rate:.3f} "
        f"- Q : {q_values}"
    )

print("\n--- Analyse des fraudes ratées par le Q-Learning ---")

missed_frauds = 0

for state, real_class, prediction in zip(
    test_states,
    y_test.values,
    rl_predictions
):

    # vraie fraude mais l'agent accepte
    if real_class == 1 and prediction == 0:

        discrete_state = agent.discretize_state(state)

        # Statistiques de cet état dans VALIDATION
        stats = fraud_state_stats.get(
            discrete_state,
            {"total": 0, "frauds": 0}
        )

        total = stats["total"]
        frauds = stats["frauds"]

        if total > 0:
            fraud_rate = frauds / total
        else:
            fraud_rate = 0

        q_values = agent.q_table.get(
            discrete_state,
            [0, 0]
        )

        print(
            f"État {discrete_state} "
            f"- Validation total : {total} "
            f"- Fraudes : {frauds} "
            f"- Taux fraude : {fraud_rate:.4f} "
            f"- Q : {q_values}"
        )

        missed_frauds += 1


print("\nNombre total de fraudes ratées :", missed_frauds)

rates = []

for state, real_class, prediction in zip(
    test_states,
    y_test.values,
    rl_predictions
):

    if real_class == 1 and prediction == 0:

        discrete_state = agent.discretize_state(state)

        stats = fraud_state_stats.get(
            discrete_state,
            {"total": 0, "frauds": 0}
        )

        total = stats["total"]
        frauds = stats["frauds"]

        if total > 0:
            fraud_rate = frauds / total
            rates.append(fraud_rate)


print("\n--- Résumé fraudes ratées ---")

print("Taux moyen de fraude des états :", np.mean(rates))
print("Taux minimum :", np.min(rates))
print("Taux maximum :", np.max(rates))

print(
    "États avec taux < 10% :",
    sum(rate < 0.10 for rate in rates)
)

print(
    "États avec taux entre 10% et 20% :",
    sum(0.10 <= rate < 0.20 for rate in rates)
)

print(
    "États avec taux >= 20% :",
    sum(rate >= 0.20 for rate in rates)
)