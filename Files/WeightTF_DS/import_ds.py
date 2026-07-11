import pandas as pd
from pathlib import Path
from config import DATASET_FILE_NAME

def get_default_dataset_path() -> Path:
    """
    Возвращает путь к датасету из текущей папки ноутбука
    или из подпапки MyNN при запуске из корня проекта.
    """

    candidates = [
        Path(DATASET_FILE_NAME),
        Path("MyNN") / DATASET_FILE_NAME,
    ]

    for path in candidates:
        if path.exists():
            return path.resolve()

    return candidates[0].resolve()


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Загружает CSV/TAB-файл в pandas DataFrame.

    Важно:
    - sep="\\t" нужен, потому что MQL5 v2 пишет колонки через TAB.
    - Time превращается в datetime, чтобы дальше можно было делать split по времени,
      графики, фильтры по датам и контроль утечки будущего.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Датасет не найден: {path}\n"
            "Сначала запусти MQL5-скрипт GetDataSet v2 или укажи путь вручную."
        )

    df = pd.read_csv(path, sep="\t")

    if "Time" not in df.columns:
        raise ValueError("В датасете нет колонки Time. Проверь, что файл создан v2-скриптом.")

    if "Target1" not in df.columns or "Target2" not in df.columns:
        raise ValueError("В датасете нет Target1/Target2. Проверь структуру CSV.")

    df["Time"] = pd.to_datetime(df["Time"])

    return df