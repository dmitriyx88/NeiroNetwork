import numpy as np
import pandas as pd
from config import exclude_col



def split_features_and_target(df: pd.DataFrame, task: str):
    """
    Разделяет DataFrame на:
    - X: матрицу признаков индикаторов;
    - y: целевую переменную;
    - feature_cols: имена колонок-признаков.

    Колонки Time, Target1 и Target2 не должны попадать в X:
    - Time не является обычным числовым признаком;
    - Target1/Target2 являются ответами, их попадание в X даст утечку цели.
    """

    excluded_cols = ["Time", "Target1", "Target2"] + exclude_col

    feature_cols = [
        col for col in df.columns
        if col not in excluded_cols
    ]

    if len(feature_cols) == 0:
        raise ValueError("Нет колонок признаков. Проверь включение индикаторов в config.mqh.")

    # astype("float32") уменьшает расход памяти и хорошо подходит TensorFlow/Keras.
    X = df[feature_cols].astype("float32").values

    if task == "classification":
        # Target1: бинарная цель.
        # 1.0 означает, что следующий ZigZag-экстремум выше текущего close.
        # 0.0 означает, что следующий ZigZag-экстремум ниже текущего close.
        y = df["Target1"].astype("float32").values
    
    elif task == "multiclass":
        # Target1: многоклассовая цель.
        # -1 означает продажу, 0 - не торговать, 1 - покупку.
        y = prepare_multiclass_target(df["Target1"].values)
    
    elif task == "regression":
        # Target2: непрерывная цель.
        # Это расстояние от текущего close до будущего ZigZag-экстремума.
        y = df["Target2"].astype("float32").values
    else:
        raise ValueError('TASK должен быть "classification", "multiclass" или "regression".')

    return X, y, feature_cols



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



def prepare_multiclass_target(y: np.ndarray) -> np.ndarray:
    """Преобразует y из [-1, 0, 1] в индексы [0, 1, 2]."""
    y_mapped = np.zeros_like(y, dtype="int32")
    y_mapped[y == -1] = 0  # продажа -> класс 0
    y_mapped[y == 0] = 1   # не торговать -> класс 1
    y_mapped[y == 1] = 2   # покупка -> класс 2
    return y_mapped