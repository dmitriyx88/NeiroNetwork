import numpy as np
import pandas as pd
from config import exclude_col


def extract_features(df: pd.DataFrame):
    excluded_cols = ["Time", "Target1", "Target2"] + exclude_col
    feature_cols = [col for col in df.columns if col not in excluded_cols]
    if len(feature_cols) == 0:
        raise ValueError("Нет колонок признаков.")
    X = df[feature_cols].astype("float32").values
    return X, feature_cols



def time_train_test_split(X: np.ndarray, y: np.ndarray, train_part: float):
    """
    Делит данные по времени: первые train_part строк идут в train, остаток в test.

    Почему не train_test_split(..., shuffle=True):
    на рынке это часто создает утечку будущего, когда модель обучается на данных,
    которые хронологически находятся после тестового участка.
    """

    if not 0.0 < train_part < 1.0:
        raise ValueError("TRAIN_PART должен быть между 0 и 1.")

    split_index = int(len(X) * train_part)

    if split_index <= 0 or split_index >= len(X):
        raise ValueError("Слишком мало строк для train/test split.")

    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test



