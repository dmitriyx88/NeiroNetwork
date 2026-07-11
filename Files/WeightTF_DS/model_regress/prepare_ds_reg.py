import pandas as pd

def get_target_reg(df: pd.DataFrame):
    # Target2: непрерывная цель.
    # Расстояние от текущего close до будущего ZigZag-экстремума.
    return df["Target2"].astype("float32").values