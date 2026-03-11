#!/usr/bin/env python3
"""
Тестування читання windowed датасету
"""

import sys
sys.path.insert(1, 'Utils')

import AWSCTDReadDataWindowed
import numpy as np

def test_windowed_data():
    """Тестує читання windowed датасету"""
    
    print("🧪 Тестування windowed датасету\n")
    
    dataset_path = "../CSV/all_syscalls_w1000_s250.csv"
    
    # Читаємо дані
    Xtr, Ytr, trace_ids, nParametersCount, nClassCount, nWordCount = \
        AWSCTDReadDataWindowed.ReadDataWindowedImpl(dataset_path, bCategorical=False)
    
    print(f"\n📊 Результати:")
    print(f"   X shape: {Xtr.shape}")
    print(f"   Y shape: {Ytr.shape}")
    print(f"   Trace IDs shape: {trace_ids.shape}")
    print(f"   Parameters count: {nParametersCount}")
    print(f"   Class count: {nClassCount}")
    print(f"   Word count: {nWordCount}")
    
    # Аналіз trace_ids
    unique_traces = np.unique(trace_ids)
    print(f"\n🔍 Аналіз трейсів:")
    print(f"   Унікальних трейсів: {len(unique_traces)}")
    print(f"   Перші 5 trace_ids: {unique_traces[:5]}")
    print(f"   Останні 5 trace_ids: {unique_traces[-5:]}")
    
    # Підрахунок вікон на трейс
    from collections import Counter
    trace_counts = Counter(trace_ids)
    windows_per_trace = list(trace_counts.values())
    
    print(f"\n📈 Статистика вікон на трейс:")
    print(f"   Мінімум: {min(windows_per_trace)}")
    print(f"   Максимум: {max(windows_per_trace)}")
    print(f"   Середнє: {np.mean(windows_per_trace):.1f}")
    print(f"   Медіана: {np.median(windows_per_trace):.1f}")
    
    # Приклад трейсу з найбільшою кількістю вікон
    max_trace = max(trace_counts, key=trace_counts.get)
    print(f"\n🔝 Трейс з найбільшою кількістю вікон:")
    print(f"   Trace ID: {max_trace}")
    print(f"   Кількість вікон: {trace_counts[max_trace]}")
    
    # Перевірка розподілу лейблів
    print(f"\n🏷️  Розподіл лейблів:")
    unique_labels, label_counts = np.unique(Ytr, return_counts=True)
    for label, count in zip(unique_labels, label_counts):
        percentage = (count / len(Ytr)) * 100
        print(f"   Клас {label}: {count:,} вікон ({percentage:.1f}%)")
    
    # Розподіл по трейсам
    trace_labels = {}
    for trace_id, label in zip(trace_ids, Ytr.flatten()):
        if trace_id not in trace_labels:
            trace_labels[trace_id] = label
    
    trace_label_values = list(trace_labels.values())
    unique_trace_labels, trace_label_counts = np.unique(trace_label_values, return_counts=True)
    
    print(f"\n🎯 Розподіл лейблів по трейсам:")
    for label, count in zip(unique_trace_labels, trace_label_counts):
        percentage = (count / len(unique_traces)) * 100
        print(f"   Клас {label}: {count:,} трейсів ({percentage:.1f}%)")
    
    print(f"\n✅ Тест завершено успішно!")
    
    return Xtr, Ytr, trace_ids

if __name__ == "__main__":
    test_windowed_data()
