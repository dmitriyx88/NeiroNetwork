from keras import layers, models, optimizers, regularizers
from keras.metrics import AUC

import numpy as np




def create_normalization_layer(X_train: np.ndarray) -> layers.Normalization:
    normalizer = layers.Normalization(axis=-1)
    normalizer.adapt(X_train)
    return normalizer



def build_classification_model(input_size: int, X_train: np.ndarray) -> models.Model:
    """
    Создает простую модель для бинарной классификации Target1.

    Последний слой:
        Dense(1, activation="sigmoid")

    Sigmoid возвращает вероятность класса 1 в диапазоне 0..1.
    """
    normalizer = create_normalization_layer(X_train)

    model = models.Sequential([
        layers.Input(shape=(input_size,)),
        normalizer,

        # Первый плотный слой ищет нелинейные комбинации индикаторов.
        layers.Dense(48, activation="relu"),

        # Dropout случайно выключает часть нейронов во время обучения.
        # Это помогает снизить переобучение.
        #layers.Dropout(0.20),

        layers.Dense(24, activation="relu"),

        # Один выход: вероятность Target1 == 1.
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            AUC(name="auc"),
        ],
    )

    return model

def build_multiclassification_model(input_size: int, X_train: np.ndarray) -> models.Model:
    """
    Создает простую модель для многоклассовой классификации Target1.

    Последний слой:
        Dense(3, activation="softmax")

    Softmax возвращает вероятность каждого класса в диапазоне 0..1.
    """
    normalizer = create_normalization_layer(X_train)

    model = models.Sequential([
        layers.Input(shape=(input_size,)),
        normalizer,

        # Первый плотный слой ищет нелинейные комбинации индикаторов.
        layers.Dense(48, activation="relu"),

        # Dropout случайно выключает часть нейронов во время обучения.
        # Это помогает снизить переобучение.
        layers.Dropout(0.20),

        layers.Dense(24, activation="relu"),

        # Один выход: вероятность Target1 == 1.
        layers.Dense(3, activation="softmax"),
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            AUC(name="auc"),
        ],
    )

    return model

def build_regression_model(input_size: int, X_train: np.ndarray) -> models.Model:
    """
    Создает простую модель для регрессии Target2.

    Последний слой без activation:
        Dense(1)

    Это значит, что модель может вернуть любое число:
    отрицательное, положительное или около нуля.
    """
    normalizer = create_normalization_layer(X_train)
    
    model = models.Sequential([
        layers.Input(shape=(input_size,)),
        normalizer,
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.0005)),
        layers.Dropout(0.20),
        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.0005)),
        layers.Dropout(0.15),
        layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.0005)),
        layers.Dropout(0.10),
        #layers.Dense(1),
        layers.Dense(1),
    ])

    model.compile(
        #optimizer=optimizers.Adam(learning_rate=1e-3),
        optimizer=optimizers.Adam(learning_rate=5e-4),
        loss="mse",
        metrics=["mae"],
    )

    return model

