import pandas as pd

def get_target_cl(df: pd.DataFrame):
    # Target1: бинарная цель.
    # 1.0 означает, что следующий ZigZag-экстремум выше текущего close.
    # 0.0 означает, что следующий ZigZag-экстремум ниже текущего close.
    return df["Target1"].astype("float32").values