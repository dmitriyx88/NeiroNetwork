import numpy as np
import pandas as pd

def get_target_mcl(df: pd.DataFrame):
    # Target1: многоклассовая цель.
    # -1 → 0 (Sell),  0 → 1 (NoTrade),  1 → 2 (Buy)
    
    #return prepare_multiclass_target(df["Target1"].values)
    
    return df["Target1"].astype("int32").values


def prepare_multiclass_target(y: np.ndarray) -> np.ndarray:
    """Преобразует y из [-1, 0, 1] в индексы [0, 1, 2]."""
    y_mapped = np.zeros_like(y, dtype="int32")
    y_mapped[y == -1] = 0  # продажа -> класс 0
    y_mapped[y == 0] = 1   # не торговать -> класс 1
    y_mapped[y == 1] = 2   # покупка -> класс 2
    return y_mapped