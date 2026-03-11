#!/usr/bin/env python3
"""
Зменшує датасет видаляючи 10% benign трейсів
"""

import pandas as pd
import numpy as np

# Читаємо датасет
print("📖 Читання датасету...")
df = pd.read_csv("../CSV/all_sequences_w1000_s250_2020.csv")

print(f"   Початкова кількість вікон: {len(df):,}")

# Отримуємо trace_ids та labels
trace_ids = df['trace_id'].values
labels = df['is_attack'].values

# Конвертуємо лейбли
labels_binary = np.where(labels == True, 1, 0)

# Знаходимо унікальні трейси для кожного класу
trace_labels = {}
for trace_id, label in zip(trace_ids, labels_binary):
    if trace_id not in trace_labels:
        trace_labels[trace_id] = label

# Розділяємо на benign та malicious трейси
benign_traces = [tid for tid, label in trace_labels.items() if label == 0]
malicious_traces = [tid for tid, label in trace_labels.items() if label == 1]

print(f"\n📊 Початковий розподіл:")
print(f"   Benign трейсів: {len(benign_traces)}")
print(f"   Malicious трейсів: {len(malicious_traces)}")

# Випадково вибираємо 40% benign трейсів для видалення
np.random.seed(42)
n_to_remove = int(len(benign_traces) * 0.4)
traces_to_remove = set(np.random.choice(benign_traces, size=n_to_remove, replace=False))

print(f"\n🗑️  Видаляємо {n_to_remove} benign трейсів ({len(traces_to_remove)} унікальних)")

# Фільтруємо датасет - залишаємо тільки вікна які НЕ належать видаленим трейсам
df_reduced = df[~df['trace_id'].isin(traces_to_remove)]

print(f"\n✅ Результат:")
print(f"   Вікон після видалення: {len(df_reduced):,}")
print(f"   Видалено вікон: {len(df) - len(df_reduced):,}")

# Перевіряємо що не видалили malicious трейси
remaining_trace_ids = df_reduced['trace_id'].unique()
remaining_labels = df_reduced['is_attack'].values
remaining_labels_binary = np.where(remaining_labels == True, 1, 0)

remaining_trace_labels = {}
for trace_id, label in zip(df_reduced['trace_id'].values, remaining_labels_binary):
    if trace_id not in remaining_trace_labels:
        remaining_trace_labels[trace_id] = label

remaining_benign = sum(1 for label in remaining_trace_labels.values() if label == 0)
remaining_malicious = sum(1 for label in remaining_trace_labels.values() if label == 1)

print(f"\n📊 Фінальний розподіл:")
print(f"   Benign трейсів: {remaining_benign} (було {len(benign_traces)})")
print(f"   Malicious трейсів: {remaining_malicious} (було {len(malicious_traces)})")

# Перевірка
if remaining_malicious != len(malicious_traces):
    print(f"\n❌ ПОМИЛКА: Видалено {len(malicious_traces) - remaining_malicious} malicious трейсів!")
else:
    print(f"\n✅ Всі malicious трейси збережено!")

# Зберігаємо зменшений датасет
output_file = "../CSV/all_sequences_w1000_s250_2020_reduced.csv"
df_reduced.to_csv(output_file, index=False)
print(f"\n💾 Збережено в: {output_file}")

# Оцінка пам'яті
n_windows = len(df_reduced)
vocab_size = 54
memory_gb = n_windows * 1000 * vocab_size * 1 / 1024 / 1024 / 1024

print(f"\n💡 Оцінка пам'яті:")
print(f"   Приблизно {memory_gb:.2f} GB (було 13.44 GB)")
print(f"   Зменшення: {(1 - memory_gb/13.44)*100:.1f}%")
