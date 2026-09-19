import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def inspect_data(df):
    print("\nTypes des colonnes :")
    print(df.dtypes)

    print("\nValeurs manquantes :")
    print(df.isnull().sum())

    print("\nLignes dupliquées :")
    print(df.duplicated().sum())

def remove_duplicates(df):
    df = df.drop_duplicates()
    return df

def handle_missing_values(df):
    numeric_columns = df.select_dtypes(include=["number"]).columns
    categorical_columns = df.select_dtypes(include=["object"]).columns

    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].median())

    for column in categorical_columns:
        df[column] = df[column].fillna(df[column].mode()[0])

    return df

def split_features_target(df, target_column):
    x = df.drop(columns=[target_column])
    y = df[target_column]

    return x, y

from sklearn.model_selection import train_test_split

def split_train_val_test(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def create_preprocessor(X):
    numeric_columns = X.select_dtypes(include=["number"]).columns
    categorical_columns = X.select_dtypes(include=["object", "string"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
        ]
    )

    return preprocessor

import joblib


def save_preprocessor(preprocessor, path):
    joblib.dump(preprocessor, path)