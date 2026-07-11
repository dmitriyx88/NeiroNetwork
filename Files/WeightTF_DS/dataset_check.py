#!/usr/bin/env python3
"""
Диагностический скрипт для проверки экспортированного датасета H1.
Сравнивает текущий датасет с эталонными параметрами.
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("ДИАГНОСТИКА ДАТАСЕТА H1")
print("=" * 70)

# 1. СТРУКТУРА ФАЙЛА
print("\n1️⃣  СТРУКТУРА ФАЙЛА:")
path = Path('Files/WeightTF_DS/DS_IN/dataset_EURUSD_M5.csv')
df = pd.read_csv(path, sep='\t')
print(f"   ✓ Размер: {df.shape[0]} строк × {df.shape[1]} колонок")
print(f"   ✓ Ожидалось: ~52,035 строк × 25 колонок")
print(f"   ✓ Признаков: {len([c for c in df.columns if c not in ['Time','Target1','Target2']])}")
if df.shape[0] < 50000 or df.shape[0] > 53000:
    print("   ⚠️  ВНИМАНИЕ: Датасет меньше/больше ожидаемого!")

# 2. ПРОПУСКИ В ДАННЫХ
print("\n2️⃣  ПРОПУСКИ В ДАННЫХ:")
missing = df.isna().sum().sum()
print(f"   ✓ Всего пропусков: {missing}")
if missing > 0:
    print("   ⚠️  ВНИМАНИЕ: Есть пропуски! Детали:")
    print(df.isna().sum()[df.isna().sum() > 0])

# 3. ЦЕЛЕВАЯ ПЕРЕМЕННАЯ
print("\n3️⃣  ЦЕЛЕВАЯ ПЕРЕМЕННАЯ (Target2):")
t2 = df['Target2']
print(f"   ✓ Тип: {t2.dtype}")
print(f"   ✓ Min: {t2.min():.6f}, Max: {t2.max():.6f}")
print(f"   ✓ Mean: {t2.mean():.6f}, Std: {t2.std():.6f}")
print(f"   ✓ Медиана: {t2.median():.6f}")
print(f"   ✓ Распределение значений:")
print(f"       -1.0: {(t2 == -1.0).sum()} ({100*(t2 == -1.0).sum()/len(t2):.1f}%)")
print(f"        0.0: {(t2 == 0.0).sum()} ({100*(t2 == 0.0).sum()/len(t2):.1f}%)")
print(f"        1.0: {(t2 == 1.0).sum()} ({100*(t2 == 1.0).sum()/len(t2):.1f}%)")
print(f"      другие: {((t2 != -1.0) & (t2 != 0.0) & (t2 != 1.0)).sum()}")

# Проверка: если почти все значения это -1/0/1, то это "ступенчатая" цель
discrete_pct = 100 * ((t2 == -1.0) | (t2 == 0.0) | (t2 == 1.0)).sum() / len(t2)
if discrete_pct > 95:
    print(f"   ⚠️  ВНИМАНИЕ: {discrete_pct:.1f}% Target2 это только -1/0/1!")
    print("       Это 'ступенчатая' цель, а не гладкая регрессия.")
    print("       Это НОРМАЛЬНО для вашей задачи!")

# 4. НОВЫЕ ПРИЗНАКИ (должны быть в файле)
print("\n4️⃣  НОВЫЕ ПРИЗНАКИ (22 шт):")
expected_new = [
    'H1_StochasticSignal', 'H1_MACDSignal',
    'H1_Close-BBLow', 'H1_BBUp-Close',
    'Close-Open', 'High-Close', 'Close-Low',
    'H1_MFI'
]
for feat in expected_new:
    if feat in df.columns:
        s = df[feat]
        print(f"   ✓ {feat}: mean={s.mean():.6f}, std={s.std():.6f}")
    else:
        print(f"   ❌ {feat}: ОТСУТСТВУЕТ!")

# 5. СТАТИСТИКА ПРИЗНАКОВ
print("\n5️⃣  СТАТИСТИКА ОСНОВНЫХ ПРИЗНАКОВ:")
feature_cols = [c for c in df.columns if c not in ['Time','Target1','Target2']]
X = df[feature_cols].astype('float32').values
print(f"   ✓ Признаков всего: {len(feature_cols)}")
print(f"   ✓ Среднее значение всех признаков: {X.mean():.6f}")
print(f"   ✓ Std всех признаков: {X.std():.6f}")
print(f"   ✓ Min: {X.min():.6f}, Max: {X.max():.6f}")

# 6. КОРРЕЛЯЦИИ С ЦЕЛЬЮ
print("\n6️⃣  ТОП КОРРЕЛЯЦИЙ С TARGET2:")
cor = pd.Series(index=feature_cols, dtype=float)
for i, c in enumerate(feature_cols):
    cor[c] = np.corrcoef(X[:, i], t2.values)[0, 1]
cor_abs = cor.abs().sort_values(ascending=False)
for feat, val in cor_abs.head(10).items():
    print(f"   • {feat}: {val:.4f}")

# 7. НОВЫЕ ПРИЗНАКИ - КОРРЕЛЯЦИИ
print("\n7️⃣  КОРРЕЛЯЦИИ НОВЫХ ПРИЗНАКОВ С TARGET2:")
for feat in expected_new:
    if feat in df.columns:
        val = cor[feat]
        print(f"   • {feat}: {val:.4f}")

# 8. ПОСЛЕДОВАТЕЛЬНОСТЬ ВРЕМЕНИ
print("\n8️⃣  ПРОВЕРКА ВРЕМЕННОЙ ПОСЛЕДОВАТЕЛЬНОСТИ:")
df['Time'] = pd.to_datetime(df['Time'])
print(f"   ✓ Первая дата: {df['Time'].min()}")
print(f"   ✓ Последняя дата: {df['Time'].max()}")
print(f"   ✓ Промежуток: {(df['Time'].max() - df['Time'].min()).days} дней")
# Проверка, что dates возрастают
is_sorted = (df['Time'].diff().dt.total_seconds() > 0).all()
print(f"   ✓ Даты отсортированы: {is_sorted}")
if not is_sorted:
    print("   ⚠️  ВНИМАНИЕ: Даты не отсортированы или есть пропуски по времени!")

# 9. СРАВНЕНИЕ С ЭТАЛОНОМ
print("\n9️⃣  СРАВНЕНИЕ С ЭТАЛОННЫМИ ЗНАЧЕНИЯМИ:")
print("   Эталон (что мы ожидали):")
print("   • Loss: 0.35, MAE: 0.45")
print("   • Target2 mean: близко к 0, std: ~0.79")
print("\n   Текущие значения:")
print(f"   • Target2 mean: {t2.mean():.6f}, std: {t2.std():.6f}")

if abs(t2.mean() - (-0.006)) < 0.05 and abs(t2.std() - 0.79) < 0.1:
    print("   ✓ TARGET2 ПО ЭТАЛОНУ - похоже на правильный датасет!")
else:
    print("   ⚠️  TARGET2 ОТЛИЧАЕТСЯ - проверьте формулу целевой переменной!")

print("\n" + "=" * 70)
print("✅ ДИАГНОСТИКА ЗАВЕРШЕНА")
print("=" * 70)
