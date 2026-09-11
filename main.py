from src.preprocess import (
    load_data,
    inspect_data,
    remove_duplicates,
    handle_missing_values,
    split_features_target
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

x, y = split_features_target(df, "is_fraud")

print("\nShape de x :", x.shape)
print("Shape de y :", y.shape)

x_train, x_test, y_train, y_test = split_train_test(X, y)

print("x_train :", x_train.shape)
print("x_test :", x_test.shape)
print("y_train :", y_train.shape)
print("y_test :", y_test.shape)