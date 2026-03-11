#!/usr/bin/env python3
"""
Перевірка розподілу класів у test set
"""

import sys
sys.path.insert(1, 'Utils')

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# Читаємо датасет
df = pd.read_csv("../CSV/all_syscalls_w1000_s250.csv")

trace_ids = df.iloc[:, 0].values
labels = df.iloc[:, -1].values

# Конвертуємо лейбли
if labels[0] in ['True', 'False']:
    labels_binary = np.where(labels == 'True', 1, 0)
else:
    labels_binary = labels.astype(int)

# Отримуємо унікальні трейси
unique_traces = np.unique(trace_ids)

# Створюємо mapping: trace_id -> label
trace_labels = {}
for trace_id, label in zip(trace_ids, labels_binary):
    if trace_id not in trace_labels:
        trace_labels[trace_id] = label

# Масив лейблів для кожного унікального трейсу
trace_labels_array = np.array([trace_labels[tid] for tid in unique_traces])

print(f"📊 Загальний розподіл трейсів:")
unique_labels, counts = np.unique(trace_labels_array, return_counts=True)
for label, count in zip(unique_labels, counts):
    percentage = (count / len(unique_traces)) * 100
    class_name = "Malware" if label == 1 else "Benign"
    print(f"   {class_name}: {count} трейсів ({percentage:.1f}%)")

# Перевіряємо розподіл у кожному фолді
kfold = KFold(n_splits=2, shuffle=True, random_state=0)

print(f"\n🔍 Розподіл по фолдам:")

for fold_num, (train_idx, test_idx) in enumerate(kfold.split(unique_traces), 1):
    print(f"\n   Fold {fold_num}:")
    
    # Train
    train_labels = trace_labels_array[train_idx]
    train_unique, train_counts = np.unique(train_labels, return_counts=True)
    print(f"   Train:")
    for label, count in zip(train_unique, train_counts):
        percentage = (count / len(train_labels)) * 100
        class_name = "Malware" if label == 1 else "Benign"
        print(f"      {class_name}: {count} трейсів ({percentage:.1f}%)")
    
    # Test
    test_labels = trace_labels_array[test_idx]
    test_unique, test_counts = np.unique(test_labels, return_counts=True)
    print(f"   Test:")
    for label, count in zip(test_unique, test_counts):
        percentage = (count / len(test_labels)) * 100
        class_name = "Malware" if label == 1 else "Benign"
        print(f"      {class_name}: {count} трейсів ({percentage:.1f}%)")
    
    # Якщо в test є тільки один клас - це проблема!
    if len(test_unique) == 1:
        print(f"   ⚠️  УВАГА: Test set містить тільки один клас!")
        print(f"   Це може пояснити 100% accuracy!")

print(f"\n💡 Аналіз:")
print(f"   Якщо test set дуже незбалансований (наприклад 99% одного класу),")
print(f"   то модель може просто завжди передбачати мажоритарний клас")
print(f"   і отримувати ~100% accuracy.")
