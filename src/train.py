from sklearn.linear_model import LogisticRegression


def train_logistic_regression(x_train, y_train):
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    )

    model.fit(x_train, y_train)

    return model

import joblib


def save_model(model, path):
    joblib.dump(model, path)

from sklearn.ensemble import RandomForestClassifier


def train_random_forest(x_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(x_train, y_train)

    return model

from xgboost import XGBClassifier


def train_xgboost(x_train, y_train):
    n_normal = (y_train == 0).sum()
    n_fraud = (y_train == 1).sum()

    scale_pos_weight = n_normal / n_fraud

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )

    model.fit(x_train, y_train)

    return model