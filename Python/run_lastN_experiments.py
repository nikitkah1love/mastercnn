#!/usr/bin/env python3
"""
Скрипт для запуску тренінгів на всіх lastN датасетах та збору метрик
"""
import subprocess
import re
import csv
import os

# Датасети для тестування
datasets = ['n10', 'n20', 'n40', 'n80', 'n100', 'n200', 'n400', 'n600', 'n800', 'n1000']

results = []

print("🚀 Запуск експериментів на lastN датасетах\n")

for dataset in datasets:
    train_file = f"CSV/output_last_n_datasets/dataset_last_{dataset}_train.csv"
    test_file = f"CSV/output_last_n_datasets/dataset_last_{dataset}_test.csv"
    
    print(f"\n{'='*70}")
    print(f"📊 Датасет: last_{dataset}")
    print(f"{'='*70}")
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"⚠️  Файли не знайдено, пропускаємо {dataset}")
        continue
    
    cmd = [
        'python', 'Python/AWSCTD_embedding.py',
        train_file,
        test_file,
        'AWSCTD-CNN-S-EMBEDDING',
        '--no-trace-aggregation'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        output = result.stdout + result.stderr
        
        accuracy_match = re.search(r'Accuracy:\s+([\d.]+)%', output)
        precision_match = re.search(r'Precision:\s+([\d.]+)\s+\(macro\)', output)
        recall_match = re.search(r'Recall:\s+([\d.]+)\s+\(macro\)', output)
        f1_match = re.search(r'F1-Score:\s+([\d.]+)\s+\(macro\)', output)
        loss_match = re.search(r'Loss:\s+([\d.]+)', output)
        train_time_match = re.search(r'Training time\s+:\s+([\d.]+)s', output)
        test_time_match = re.search(r'Testing time\s+:\s+([\d.]+)s', output)
        
        if accuracy_match and precision_match and recall_match and f1_match:
            accuracy = float(accuracy_match.group(1))
            precision = float(precision_match.group(1))
            recall = float(recall_match.group(1))
            f1 = float(f1_match.group(1))
            loss = float(loss_match.group(1)) if loss_match else 0.0
            train_time = float(train_time_match.group(1)) if train_time_match else 0.0
            test_time = float(test_time_match.group(1)) if test_time_match else 0.0
            fnr = 1.0 - recall
            fpr = 1.0 - precision
            
            results.append({
                'Dataset': f'last_{dataset}',
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'FPR': fpr,
                'FNR': fnr,
                'Loss': loss,
                'Train_Time': train_time,
                'Test_Time': test_time
            })
            
            print(f"✅ Успішно: Accuracy={accuracy:.2f}%, F1={f1:.4f}")
        else:
            print(f"❌ Не вдалося розпарсити метрики")
            print(f"Output: {output[-500:]}")
            
    except subprocess.TimeoutExpired:
        print(f"⏱️  Таймаут для {dataset}")
    except Exception as e:
        print(f"❌ Помилка: {e}")

output_file = 'Python/lastN_results.csv'
if results:
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['Dataset', 'Accuracy', 'Precision', 'Recall', 'F1', 'FPR', 'FNR', 'Loss', 'Train_Time', 'Test_Time']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print(f"\n{'='*70}")
    print(f"{'Dataset':<12} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1':<10} {'FPR':<10} {'FNR':<10} {'Loss':<10} {'Train(s)':<10} {'Test(s)':<10}")
    print(f"{'='*70}")
    for row in results:
        print(f"{row['Dataset']:<12} {row['Accuracy']:<10.2f} {row['Precision']:<12.4f} {row['Recall']:<10.4f} {row['F1']:<10.4f} {row['FPR']:<10.4f} {row['FNR']:<10.4f} {row['Loss']:<10.4f} {row['Train_Time']:<10.2f} {row['Test_Time']:<10.2f}")
    print(f"\n✅ Результати збережено у {output_file}")
else:
    print("\n❌ Немає результатів для збереження")
