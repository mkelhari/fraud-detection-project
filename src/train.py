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

    