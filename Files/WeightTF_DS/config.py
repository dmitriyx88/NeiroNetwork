PATH_DATASET = "DS_IN/"
DATASET_FILE_NAME = f"{PATH_DATASET}dataset_EURUSD_M5.csv"
#DATASET_FILE_NAME = f"{PATH_DATASET}dataset_EURUSD_H1.csv"


# Основной режим обучения:
#   "classification" -> обучаем модель предсказывать Target1, то есть направление 0/1.
#   "regression"     -> обучаем модель предсказывать Target2, то есть расстояние до будущего экстремума.
#TASK = "classification"
# TASK = "regression"
TASK = "multiclass"


# Доля данных, которая пойдет в train.
# Для рынка нельзя случайно перемешивать прошлое и будущее перед split,
# поэтому ниже используется обычное деление по времени: сначала train, потом test.
TRAIN_PART = 0.80
#TRAIN_PART = 0.50

# Размер batch. Для табличных финансовых датасетов часто удобно начинать с 128-512.
#BATCH_SIZE = 256
BATCH_SIZE = 512

# Количество проходов по train-выборке.
# Если модель быстро переобучается, уменьшай EPOCHS или добавляй Dropout/регуляризацию.
EPOCHS = 50


# Исключить признаки:

exclude_col= []


