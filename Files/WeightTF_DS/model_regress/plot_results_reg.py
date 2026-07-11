import numpy as np
import matplotlib.pyplot as plt
from graphic_hist import finish_plot


def plot_scatter(y_true, y_pred) -> None:
    """Scatter: предсказание vs реальный таргет."""
    plt.scatter(y_true, y_pred, alpha=0.3, s=5, color="steelblue")
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], "r--", label="Идеальное предсказание")
    plt.xlabel("Реальный Target2")
    plt.ylabel("Предсказанный Target2")
    plt.title("Предсказание vs Реальный таргет (регрессия)")
    plt.legend()
    finish_plot("scatter_reg.png")


def plot_residuals(y_true, y_pred) -> None:
    """Гистограмма остатков (y_pred - y_true)."""
    residuals = y_pred - y_true
    plt.hist(residuals, bins=60, edgecolor="black", color="steelblue")
    plt.axvline(0,                color="red",    linestyle="--", label="Нулевая ошибка")
    plt.axvline(residuals.mean(), color="orange", linestyle="--", label=f"Среднее = {residuals.mean():.3f}")
    plt.title("Распределение остатков (y_pred - y_true)")
    plt.xlabel("Остаток")
    plt.ylabel("Количество")
    plt.legend()
    finish_plot("residuals_reg.png")