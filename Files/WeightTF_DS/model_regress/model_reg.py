from keras import layers, models, optimizers, regularizers
from keras.metrics import AUC

import numpy as np




def normalization_layer_regression(X_train: np.ndarray) -> layers.Normalization:
    normalizer = layers.Normalization(axis=-1)
    normalizer.adapt(X_train)
    return normalizer


def build_regression_model(input_size: int, X_train: np.ndarray) -> models.Model:
    """
    Создает простую модель для регрессии Target2.

    Последний слой без activation:
        Dense(1)

    Это значит, что модель может вернуть любое число:
    отрицательное, положительное или около нуля.
    """
    normalizer = normalization_layer_regression(X_train)
    
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