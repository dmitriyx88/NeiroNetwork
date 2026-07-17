from config import BATCH_SIZE, EPOCHS, TASK, TRAIN_PART
from import_ds import get_default_dataset_path, load_dataset
from graphic_hist import plot_training_history
from export_model import export_model
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, balanced_accuracy_score

from preparation_ds import extract_features, time_train_test_split

from model_class.model_cl import build_classification_model
from model_class.prepare_ds_cl import get_target_cl
from model_class.plot_results_cl import plot_roc_curve, plot_confusion_matrix_cl, plot_probability_distribution_cl

from model_multiclass.model_mcl import build_multiclassification_model
from model_multiclass.prepare_ds_mcl import get_target_mcl
from model_multiclass.plot_results_mcl import plot_confusion_matrix_mcl, plot_class_distribution, plot_softmax_confidence

from model_regress.model_reg import build_regression_model
from model_regress.prepare_ds_reg import get_target_reg
from model_regress.plot_results_reg import plot_scatter, plot_residuals

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

    #X, y, feature_cols = split_features_and_target(df, TASK)
    X, feature_cols = extract_features(df)
    if TASK == "classification":
        y = get_target_cl(df)
    elif TASK == "multiclass":
        y = get_target_mcl(df)
    else:
        y = get_target_reg(df)

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

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.summary()

    # shuffle=False сохраняет порядок временного ряда внутри эпохи.
    # validation_split берет последний кусок train-выборки под validation.
    fit_params = dict(
        validation_split=0.20,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=1,
        callbacks=[early_stop],
    )
    #if TASK == "multiclass":
    #    fit_params["class_weight"] = {0: 1.2, 1: 1.0, 2: 1.2}
    
    history = model.fit(
        X_train,
        y_train,
        **fit_params
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

        # Графики для бинарной классификации.
        plot_probability_distribution_cl(probabilities)
        plot_roc_curve(y_test, probabilities)
        plot_confusion_matrix_cl(y_test, signals)
    
    elif TASK == "multiclass":
        # Для 3-классовой классификации predict возвращает вероятности [Sell, NoTrade, Buy].
        probabilities = model.predict(X_test, verbose=0)   # shape (N, 3)
        class_indices = np.argmax(probabilities, axis=1)   # 0=Sell, 1=NoTrade, 2=Buy
        signals = class_indices                       # обратно в -1 / 0 / +1

        print()
        print("Balanced accuracy:")
        print(balanced_accuracy_score(y_test, class_indices))

        print()
        print("Classification report:")
        print(classification_report(
            y_test,
            class_indices,
            labels=list(range(6)),
            target_names=["Buy TP",
    "Sell TP",
    "Buy SL",
    "Sell SL",
    "Timeout",
    "No Trade",],
            digits=4,
            zero_division=0,
        ))

        print()
        print("Первые 10 вероятностей [Sell, NoTrade, Buy]:")
        print(probabilities[:10].round(3))

        print("Первые 10 предсказанных классов (0=BuyTP, 1=SellTP, 2=BuySL, 3=SellSL, 4=Timeout, 5=NoTrade):")
        print(signals[:10])

        # Графики для 3-классовой классификации.
        plot_confusion_matrix_mcl(y_test, class_indices)
        plot_class_distribution(y_test, class_indices)
        plot_softmax_confidence(probabilities)

    else:
        # Для регрессии predict возвращает ожидаемое Target2.
        predictions = model.predict(X_test, verbose=0).ravel()

        print()
        print("Первые 10 прогнозов Target2:")
        print(predictions[:10])

        # Графики для регрессии.
        plot_scatter(y_test, predictions)
        plot_residuals(y_test, predictions)

    # history.history можно использовать для графика loss/accuracy/auc.
    return history

#export_model(save_keras=True, save_ONNX=True, model_name="model")

if __name__ == "__main__":
    
    history = main() 
    plot_training_history(history)
