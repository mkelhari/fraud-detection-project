from sklearn.ensemble import IsolationForest


def train_isolation_forest(x_train):
    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    model.fit(x_train)

    return model

def get_anomaly_scores(model, X):
    scores = -model.decision_function(X)
    return scores

