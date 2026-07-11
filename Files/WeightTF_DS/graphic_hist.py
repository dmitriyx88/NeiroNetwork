from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def finish_plot(file_name: str) -> None:
    """Сохраняет график в файл и показывает окно, если backend поддерживает GUI."""

    PLOTS_DIR.mkdir(exist_ok=True)
    path = PLOTS_DIR / file_name

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"График сохранен: {path}")

    if matplotlib.get_backend().lower() != "agg":
        plt.show()

    plt.close()


def plot_training_history(history) -> None:
    """Строит графики обучения и валидации для текущей модели."""

    history_dict = history.history
    epochs = range(1, len(history_dict["loss"]) + 1)

    # График ошибки: показывает, как меняется loss на обучении и валидации.
    plt.plot(epochs, history_dict["loss"], "bo", label="Ошибка на обучении")
    plt.plot(epochs, history_dict["val_loss"], "b", label="Ошибка на валидации")
    plt.title("Ошибка на обучении и валидации")
    plt.xlabel("Эпохи")
    plt.ylabel("Ошибка")
    plt.legend()
    finish_plot("loss.png")

    # График точности: показывает долю правильных ответов для Target1.
    if "accuracy" in history_dict and "val_accuracy" in history_dict:
        plt.plot(epochs, history_dict["accuracy"], "bo", label="Точность на обучении")
        plt.plot(epochs, history_dict["val_accuracy"], "b", label="Точность на валидации")
        plt.title("Точность на обучении и валидации")
        plt.xlabel("Эпохи")
        plt.ylabel("Accuracy")
        plt.legend()
        finish_plot("accuracy.png")

    # График AUC: полезен для бинарной классификации, особенно если классы несбалансированы.
    if "auc" in history_dict and "val_auc" in history_dict:
        plt.plot(epochs, history_dict["auc"], "bo", label="AUC на обучении")
        plt.plot(epochs, history_dict["val_auc"], "b", label="AUC на валидации")
        plt.title("AUC на обучении и валидации (бинарная классификация)")
        plt.xlabel("Эпохи")
        plt.ylabel("AUC")
        plt.legend()
        finish_plot("auc.png")

    # График MAE: для регрессии показывает среднюю абсолютную ошибку.
    if "mae" in history_dict and "val_mae" in history_dict:
        plt.plot(epochs, history_dict["mae"], "bo", label="MAE на обучении")
        plt.plot(epochs, history_dict["val_mae"], "b", label="MAE на валидации")
        plt.title("MAE на обучении и валидации")
        plt.xlabel("Эпохи")
        plt.ylabel("MAE")
        plt.legend()
        finish_plot("mae.png")

