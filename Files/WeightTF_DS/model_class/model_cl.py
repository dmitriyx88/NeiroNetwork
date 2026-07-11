from keras import layers, models, optimizers, regularizers
from keras.metrics import AUC

import numpy as np




def normalization_layer_class(X_train: np.ndarray) -> layers.Normalization:
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
    normalizer = normalization_layer_class(X_train)

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