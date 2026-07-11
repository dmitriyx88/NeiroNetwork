from config import BATCH_SIZE, EPOCHS, TASK, TRAIN_PART
from import_ds import get_default_dataset_path, load_dataset
from preparation_ds import split_features_and_target, time_train_test_split
from model_NN import build_regression_model, build_classification_model, build_multiclassification_model
from graphic_hist import plot_probability_distribution, plot_training_history
import numpy as np


def main() -> None:
    """
    Главная функция примера:
    1. Находит датасет.
    2. Загружает его через pandas.
    3. Отделяет признаки от target.
    4. Делит данные по времени.
    5. Создает Keras-модель со встроенной нормализацией признаков.
    6. Обучает модель.
    7. Проверяет качество на test-участке.
    """

    dataset_path = get_default_dataset_path()
    print(f"Dataset path: {dataset_path}")

    df = load_dataset(dataset_path)

    print("Первые строки датасета:")
    print(df.head())
    print()

    print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} колонок")
    print(f"Период: {df['Time'].min()} -> {df['Time'].max()}")
    print()

    X, y, feature_cols = split_features_and_target(df, TASK)

    print(f"Количество признаков: {len(feature_cols)}")
    print("Признаки:")
    for name in feature_cols:
        print(f"  - {name}")
    print()

    global X_train, X_test, y_train, y_test, model 
    X_train, X_test, y_train, y_test = time_train_test_split(X, y, TRAIN_PART)

    print(f"Train rows: {len(X_train)}")
    print(f"Test rows:  {len(X_test)}")
    print()

    if TASK == "classification":
        model = build_classification_model(input_size=X_train.shape[1], X_train=X_train)
    
    elif TASK == "multiclass":
        model = build_multiclassification_model(input_size=X_train.shape[1], X_train=X_train)
    else:
        model = build_regression_model(input_size=X_train.shape[1], X_train=X_train)

    model.summary()

    # shuffle=False сохраняет порядок временного ряда внутри эпохи.
    # validation_split берет последний кусок train-выборки под validation.
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.20,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=1,
    )

    test_result = model.evaluate(X_test, y_test, verbose=0)
    print()
    print("Test metrics:")
    print(dict(zip(model.metrics_names, test_result)))

    if TASK == "classification":
        # Для классификации predict возвращает вероятность Target1 == 1.
        probabilities = model.predict(X_test, verbose=0).ravel()

        # Порог 0.5 - стартовая точка. Для торговли его обычно подбирают отдельно.
        signals = (probabilities >= 0.5).astype(int)

        print()
        print("Первые 10 вероятностей Target1 == 1:")
        print(probabilities[:10])

        print("Первые 10 сигналов по порогу 0.5:")
        print(signals[:10])

        plot_probability_distribution(probabilities)
    
    elif TASK == "multiclass":
        # Для многоклассовой классификации predict возвращает вероятности для каждого класса.
        probabilities = model.predict(X_test, verbose=0)
        predictions = np.argmax(probabilities, axis=1)

        print()
        print("Первые 10 прогнозов Target1 (многоклассовая):")
        print(predictions[:10])

    else:
        # Для регрессии predict возвращает ожидаемое Target2.
        predictions = model.predict(X_test, verbose=0).ravel()

        print()
        print("Первые 10 прогнозов Target2:")
        print(predictions[:10])

    # history.history можно использовать для графика loss/accuracy/auc.
    return history


if __name__ == "__main__":
    
    history = main() 
    plot_training_history(history)
