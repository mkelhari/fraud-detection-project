def make_predictions(model, x_test):
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    return predictions, probabilities

import joblib


def make_predictions(model, X_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return predictions, probabilities


def load_artifacts(model_path, preprocessor_path):
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    return model, preprocessor


def predict_new_transaction(model, preprocessor, transaction):
    transaction_prepared = preprocessor.transform(transaction)

    prediction = model.predict(transaction_prepared)
    probability = model.predict_proba(transaction_prepared)[:, 1]

    return prediction, probability