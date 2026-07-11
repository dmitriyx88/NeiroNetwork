import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from graphic_hist import finish_plot


def plot_roc_curve(y_true, probabilities) -> None:
    """ROC-кривая с площадью AUC."""
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color="blue", label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Случайный классификатор")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривая (бинарная классификация)")
    plt.legend()
    finish_plot("roc_curve_cl.png")


def plot_confusion_matrix_cl(y_true, y_pred) -> None:
    """Матрица ошибок 2×2."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Sell (0)", "Buy (1)"])
    disp.plot(cmap="Blues")
    plt.title("Матрица ошибок (бинарная классификация)")
    finish_plot("confusion_matrix_cl.png")


def plot_probability_distribution_cl(probabilities) -> None:
    """Гистограмма вероятностей с торговыми порогами."""
    plt.hist(probabilities, bins=50, edgecolor="black")
    plt.axvline(0.50, color="black", linestyle="--", label="Порог 0.50")
    plt.axvline(0.60, color="green", linestyle="--", label="Buy > 0.60")
    plt.axvline(0.40, color="red", linestyle="--", label="Sell < 0.40")
    plt.title("Распределение вероятностей (бинарная классификация)")
    plt.xlabel("Вероятность")
    plt.ylabel("Количество прогнозов")
    plt.legend()
    finish_plot("probabilities_cl.png")