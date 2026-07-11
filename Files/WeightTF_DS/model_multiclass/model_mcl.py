from keras import layers, models, optimizers, regularizers



import numpy as np




def normalization_layer_multiclass(X_train: np.ndarray) -> layers.Normalization:
    normalizer = layers.Normalization(axis=-1)
    normalizer.adapt(X_train)
    return normalizer



def build_multiclassification_model(input_size: int, X_train: np.ndarray) -> models.Model:
    """
    Создает модель для 3-классовой классификации Target1/Target2.

    Классы (после prepare_multiclass_target):
        0 -> Sell   (исходный -1)
        1 -> NoTrade (исходный  0)
        2 -> Buy    (исходный +1)

    Последний слой:
        Dense(3, activation="softmax")

    Softmax возвращает вероятности трёх классов, сумма = 1.
    Таргет должен быть int32: 0 / 1 / 2.

    Примечание: AUC не используется — keras.metrics.AUC рассчитан на бинарный вывод
    и даёт некорректный результат при softmax с 3 выходами.
    """
    normalizer = normalization_layer_multiclass(X_train)


    model = models.Sequential([
        layers.Input(shape=(input_size,)),
        normalizer,
        layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.0005)),
        layers.Dropout(0.20),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.0005)),
        layers.Dropout(0.15),
        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.0005)),
        layers.Dropout(0.10),
        layers.Dense(3, activation="softmax"),
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model