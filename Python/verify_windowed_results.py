#!/usr/bin/env python3
"""
Детальна перевірка результатів windowed тренування
"""

import sys
sys.path.insert(1, 'Utils')

import numpy as np
import pandas as pd
from collections import defaultdict

def verify_results():
    """Перевіряє чи правильно працює агрегація"""
    
    print("🔍 Детальна перевірка результатів windowed тренування\n")
    
    # Читаємо датасет
    print("📖 Читання датасету...")
    df = pd.read_csv("../CSV/all_syscalls_w1000_s250.csv")
    
    trace_ids = df.iloc[:, 0].values
    labels = df.iloc[:, -1].values
    
    # Конвертуємо лейбли
    if labels[0] in ['True', 'False']:
        labels_binary = np.where(labels == 'True', 1, 0)
    else:
        labels_binary = labels.astype(int)
    
    print(f"   Всього вікон: {len(df):,}")
    print(f"   Унікальних трейсів: {len(np.unique(trace_ids)):,}")
    
    # Групуємо по trace_id
    trace_data = defaultdict(lambda: {'windows': [], 'label': None})
    
    for i, (trace_id, label) in enumerate(zip(trace_ids, labels_binary)):
        trace_data[trace_id]['windows'].append(i)
        if trace_data[trace_id]['label'] is None:
            trace_data[trace_id]['label'] = label
        elif trace_data[trace_id]['label'] != label:
            print(f"⚠️  УВАГА: Трейс {trace_id} має різні лейбли!")
    
    print(f"\n📊 Статистика по трейсам:")
    
    # Розподіл лейблів по трейсам
    trace_labels = [data['label'] for data in trace_data.values()]
    unique_labels, counts = np.unique(trace_labels, return_counts=True)
    
    for label, count in zip(unique_labels, counts):
        percentage = (count / len(trace_labels)) * 100
        class_name = "Malware" if label == 1 else "Benign"
        print(f"   {class_name} (клас {label}): {count:,} трейсів ({percentage:.1f}%)")
    
    # Статистика вікон на трейс
    windows_per_trace = [len(data['windows']) for data in trace_data.values()]
    print(f"\n📈 Вікна на трейс:")
    print(f"   Мінімум: {min(windows_per_trace)}")
    print(f"   Максимум: {max(windows_per_trace)}")
    print(f"   Середнє: {np.mean(windows_per_trace):.1f}")
    print(f"   Медіана: {np.median(windows_per_trace):.1f}")
    
    # Перевірка розподілу вікон по класам
    print(f"\n🔍 Розподіл вікон по класам:")
    window_labels = labels_binary
    unique_window_labels, window_counts = np.unique(window_labels, return_counts=True)
    
    for label, count in zip(unique_window_labels, window_counts):
        percentage = (count / len(window_labels)) * 100
        class_name = "Malware" if label == 1 else "Benign"
        print(f"   {class_name} (клас {label}): {count:,} вікон ({percentage:.1f}%)")
    
    # Перевірка чи всі вікна одного трейсу мають однаковий лейбл
    print(f"\n🧪 Перевірка консистентності лейблів:")
    inconsistent_traces = 0
    
    for trace_id, data in trace_data.items():
        trace_label = data['label']
        window_indices = data['windows']
        window_labels_for_trace = labels_binary[window_indices]
        
        if not np.all(window_labels_for_trace == trace_label):
            inconsistent_traces += 1
            print(f"   ⚠️  Трейс {trace_id}: очікується {trace_label}, але є {np.unique(window_labels_for_trace)}")
    
    if inconsistent_traces == 0:
        print(f"   ✅ Всі трейси мають консистентні лейбли")
    else:
        print(f"   ❌ Знайдено {inconsistent_traces} трейсів з неконсистентними лейблами!")
    
    # Симуляція простої агрегації
    print(f"\n🎯 Симуляція агрегації (якщо всі вікна правильно класифіковані):")
    
    # Якщо window accuracy = 95%, скільки трейсів будуть правильні?
    window_acc = 0.95
    
    correct_traces_simulation = 0
    total_traces = len(trace_data)
    
    for trace_id, data in trace_data.items():
        num_windows = len(data['windows'])
        true_label = data['label']
        
        # Симулюємо що 95% вікон правильні
        # Якщо більшість вікон правильні, то MEAN буде правильний
        expected_correct_windows = int(num_windows * window_acc)
        
        # Якщо більше половини вікон правильні, trace prediction буде правильний
        if expected_correct_windows > num_windows / 2:
            correct_traces_simulation += 1
    
    simulated_trace_acc = (correct_traces_simulation / total_traces) * 100
    print(f"   При window accuracy {window_acc*100}%:")
    print(f"   Очікувана trace accuracy: ~{simulated_trace_acc:.1f}%")
    
    # Перевірка train/test split
    print(f"\n🔀 Перевірка train/test split:")
    
    train_traces = [tid for tid in trace_ids if 'train' in str(tid)]
    test_traces = [tid for tid in trace_ids if 'test' in str(tid)]
    
    print(f"   Train вікон: {len(train_traces):,}")
    print(f"   Test вікон: {len(test_traces):,}")
    
    unique_train = len(np.unique(train_traces))
    unique_test = len(np.unique(test_traces))
    
    print(f"   Унікальних train трейсів: {unique_train:,}")
    print(f"   Унікальних test трейсів: {unique_test:,}")
    
    # КРИТИЧНА ПЕРЕВІРКА: чи є перетин між train та test?
    train_trace_ids = set(np.unique(train_traces))
    test_trace_ids = set(np.unique(test_traces))
    overlap = train_trace_ids.intersection(test_trace_ids)
    
    if len(overlap) > 0:
        print(f"\n   ❌ КРИТИЧНА ПОМИЛКА: Знайдено {len(overlap)} трейсів які є і в train, і в test!")
        print(f"   Приклади: {list(overlap)[:5]}")
        print(f"\n   🚨 ЦЕ ПОЯСНЮЄ 100% ACCURACY!")
        print(f"   Модель бачила ці трейси під час тренування!")
    else:
        print(f"   ✅ Немає перетину між train та test трейсами")
    
    # Перевірка KFold split
    print(f"\n🔄 Аналіз KFold split:")
    print(f"   KFold робить split на рівні ВІКОН, а не ТРЕЙСІВ!")
    print(f"   Це означає що вікна одного трейсу можуть потрапити:")
    print(f"   - Частина в train")
    print(f"   - Частина в test")
    print(f"\n   🚨 ЦЕ ГОЛОВНА ПРОБЛЕМА!")
    print(f"   Модель бачить частину вікон трейсу під час тренування,")
    print(f"   а потім тестується на інших вікнах ТОГО Ж трейсу!")
    
    return trace_data

if __name__ == "__main__":
    trace_data = verify_results()
    
    print(f"\n" + "="*60)
    print(f"💡 ВИСНОВОК:")
    print(f"="*60)
    print(f"100% trace accuracy - це НЕ помилка в коді, але є")
    print(f"МЕТОДОЛОГІЧНА ПРОБЛЕМА:")
    print(f"")
    print(f"KFold.split() розділяє ВІКНА, а не ТРЕЙСИ.")
    print(f"Тому вікна одного трейсу потрапляють і в train, і в test.")
    print(f"")
    print(f"Модель вчиться на частині вікон трейсу, а тестується")
    print(f"на інших вікнах того ж трейсу - звідси 100% accuracy.")
    print(f"")
    print(f"РІШЕННЯ: Потрібно робити split на рівні ТРЕЙСІВ, а не вікон!")
    print(f"="*60)
