import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from graphic_hist import finish_plot

CLASS_NAMES = ["Buy TP",
    "Sell TP",
    "Buy SL",
    "Sell SL",
    "Timeout",
    "No Trade",]


def plot_confusion_matrix_mcl(y_true, y_pred_classes) -> None:
    """Матрица ошибок 3×3. Принимает классы 0/1/2."""
    cm = confusion_matrix(y_true, y_pred_classes, labels=list(range(len(CLASS_NAMES))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues")
    plt.title("Матрица ошибок (мультикласс)")
    finish_plot("confusion_matrix_mcl.png")


def plot_class_distribution(y_true, y_pred_classes) -> None:
    """Столбчатый график: реальное vs предсказанное распределение классов."""
    x = np.arange(len(CLASS_NAMES))
    width = 0.35
    true_counts = [np.sum(y_true == i) for i in range(len(CLASS_NAMES))]
    pred_counts  = [np.sum(y_pred_classes == i) for i in range(len(CLASS_NAMES))]
    plt.bar(x - width / 2, true_counts, width, label="Реальные",      color="steelblue")
    plt.bar(x + width / 2, pred_counts, width, label="Предсказанные", color="orange")
    plt.xticks(x, CLASS_NAMES)
    plt.ylabel("Количество")
    plt.title("Распределение классов: реальные vs предсказанные")
    plt.legend()
    finish_plot("class_distribution_mcl.png")


def plot_softmax_confidence(proba) -> None:
    """Гистограмма максимальной уверенности softmax."""
    confidence = proba.max(axis=1)
    plt.hist(confidence, bins=50, edgecolor="black", color="steelblue")
    for thresh in [0.40, 0.50, 0.60, 0.70]:
        plt.axvline(thresh, linestyle="--", label=f"> {thresh:.0%}")
    plt.title("Распределение уверенности softmax")
    plt.xlabel("Максимальная вероятность класса")
    plt.ylabel("Количество прогнозов")
    plt.legend()
    finish_plot("softmax_confidence_mcl.png")